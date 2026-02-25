#!/usr/bin/env python3
"""Export PocketTTS subgraphs to ONNX and write a manifest.

Exports:
- text_conditioner
- flow_lm_main
- flow_lm_flow
- latent_to_mimi
- mimi_encoder
- mimi_decoder
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    import onnx
    from onnx import TensorProto
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "onnx package is required for export_onnx.py. "
        "Install it in the selected python environment."
    ) from exc

# Disable beartype claw instrumentation during export/tracing.
# pocket_tts enables beartype at package import time, which can break ONNX tracing
# when symbolic tensor shapes flow through functions expecting plain ints.
try:  # pragma: no cover - best effort compatibility shim
    import beartype.claw as _beartype_claw

    _beartype_claw.beartype_this_package = lambda *args, **kwargs: None
except Exception:
    pass

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states


OPSET_VERSION = 17


@dataclass
class ExportSpec:
    name: str
    filename: str
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]]
    example_inputs: tuple[torch.Tensor, ...]
    module: torch.nn.Module


def clone_model_state(state: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for module_name, module_state in state.items():
        out[module_name] = {k: v.clone() for k, v in module_state.items()}
    return out


def extract_kv_tensors(
    flow_lm: "torch.nn.Module",
    state: "dict[str, dict[str, torch.Tensor]]",
    t_written: int,
) -> "tuple[list[torch.Tensor], torch.Tensor]":
    """Extract per-layer KV tensors and offset from model_state after prefill.

    Returns (kv_list, offset_tensor) where:
    - kv_list[i] is the [2, B, t_written, H, Dh] slice of layer i's cache
    - offset_tensor is int64[1] = t_written
    """
    kv_list = []
    for _module_name, module in flow_lm.named_modules():
        if not hasattr(module, "_cache_backend"):
            continue
        layer_state = state[module._module_absolute_name]
        # cache shape: [2, B, max_seq, H, Dh]; slice to written portion
        kv = layer_state["cache"][:, :, :t_written, :, :]
        kv_list.append(kv)
    offset = torch.tensor([t_written], dtype=torch.long)
    return kv_list, offset


def rebuild_state_from_kv(
    flow_lm: "torch.nn.Module",
    kv_list: "list[torch.Tensor]",
    offset: "torch.Tensor",
    max_seq: int,
) -> "dict[str, dict[str, torch.Tensor]]":
    """Reconstruct a model_state dict from per-layer KV tensors.

    Pads the cache back to [2, B, max_seq, H, Dh] with NaN.
    """
    state: dict[str, dict[str, torch.Tensor]] = {}
    kv_iter = iter(kv_list)
    for _module_name, module in flow_lm.named_modules():
        if not hasattr(module, "_cache_backend"):
            continue
        kv = next(kv_iter)  # [2, B, t_written, H, Dh]
        t_written_local = kv.shape[2]
        b, h, dh = kv.shape[1], kv.shape[3], kv.shape[4]
        cache = torch.full((2, b, max_seq, h, dh), float("nan"), dtype=kv.dtype)
        cache[:, :, :t_written_local, :, :] = kv
        abs_name = module._module_absolute_name
        state[abs_name] = {
            "cache": cache,
            "offset": offset.expand(b).clone(),
        }
    return state


class TextConditionerWrapper(torch.nn.Module):
    def __init__(self, model: TTSModel):
        super().__init__()
        self.conditioner = model.flow_lm.conditioner

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.conditioner(TokenizedText(tokens=tokens))


class FlowLMMainWrapper(torch.nn.Module):
    def __init__(self, model: TTSModel, max_sequence_length: int = 256):
        super().__init__()
        self.flow_lm = model.flow_lm
        self.base_state = init_states(self.flow_lm, batch_size=1, sequence_length=max_sequence_length)
        # Register bos_emb as a buffer so it is baked into the ONNX graph as a constant.
        # The Go caller signals BOS positions by passing NaN; we replace them here so that
        # the torch.isnan() branch is always traced (example input contains NaN).
        self.register_buffer("bos_emb", model.flow_lm.bos_emb.detach().clone())

    def forward(self, sequence: torch.Tensor, text_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = clone_model_state(self.base_state)
        # Replace NaN BOS positions with the learned bos_emb embedding.
        # bos_emb is [ldim]; broadcast to match sequence shape [B, S, ldim].
        sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)
        projected = self.flow_lm.input_linear(sequence)
        hidden = self.flow_lm.backbone(projected, text_embeddings, sequence, model_state=state)
        last_hidden = hidden[:, -1, :]
        eos_logits = self.flow_lm.out_eos(last_hidden)
        return last_hidden, eos_logits


class FlowLMPrefillWrapper(torch.nn.Module):
    """Runs text embeddings through the FlowLM transformer backbone once and
    returns per-layer KV-cache tensors for use in incremental AR generation.

    Called once per synthesis chunk before the AR loop. Returns kv_0..kv_{N-1}
    (each [2, 1, T, H, Dh]) and offset (int64[1]=T).
    """

    def __init__(self, model: TTSModel, max_sequence_length: int = 256):
        super().__init__()
        self.flow_lm = model.flow_lm
        self.max_sequence_length = max_sequence_length
        self._num_kv_layers = sum(
            1 for _, m in model.flow_lm.named_modules() if hasattr(m, "_cache_backend")
        )

    def forward(self, text_embeddings: torch.Tensor) -> tuple:
        """
        Args:
            text_embeddings: [1, T, 1024]
        Returns:
            kv_0, kv_1, ..., kv_{N-1}: [2, 1, T, H, Dh] each
            offset: int64[1] = T
        """
        T = text_embeddings.shape[1]
        state = init_states(self.flow_lm, batch_size=1, sequence_length=self.max_sequence_length)

        # Run backbone with text-only (empty sequence input).
        # backbone() does: input_ = cat([text_embeddings, sequence_input], dim=1)
        # then transformer, then strips the sequence prefix from output.
        # With empty sequence the stripped portion is empty, so no output is needed.
        empty_seq = torch.zeros(1, 0, self.flow_lm.ldim, dtype=text_embeddings.dtype)
        projected = self.flow_lm.input_linear(empty_seq)
        self.flow_lm.backbone(projected, text_embeddings, empty_seq, model_state=state)

        kv_list, offset = extract_kv_tensors(self.flow_lm, state, T)
        return tuple(kv_list) + (offset,)


class FlowLMStepWrapper(torch.nn.Module):
    """Runs a single autoregressive step with explicit KV-cache I/O.

    Accepts sequence_frame [1, 1, 32], per-layer KV tensors, and offset as inputs.
    Returns last_hidden [1, 1024], eos_logits [1, 1], updated KV tensors, and
    updated offset. The Go caller maintains the KV state between steps.
    """

    def __init__(self, model: TTSModel, max_sequence_length: int = 256):
        super().__init__()
        self.flow_lm = model.flow_lm
        self.max_sequence_length = max_sequence_length
        self.register_buffer("bos_emb", model.flow_lm.bos_emb.detach().clone())
        self._num_kv_layers = sum(
            1 for _, m in model.flow_lm.named_modules() if hasattr(m, "_cache_backend")
        )

    def forward(self, sequence_frame: torch.Tensor, *args: torch.Tensor) -> tuple:
        """
        Args:
            sequence_frame: [1, 1, 32] — NaN for BOS, latent frame thereafter
            *args: kv_0, kv_1, ..., kv_{N-1}, offset
                   kv_i: [2, 1, S, H, Dh]
                   offset: int64[1]
        Returns:
            last_hidden: [1, 1024]
            eos_logits: [1, 1]
            kv_0, ..., kv_{N-1}: updated [2, 1, S+1, H, Dh]
            offset: updated int64[1]
        """
        kv_list = list(args[:-1])
        offset = args[-1]

        # Reconstruct state dict from KV tensors + offset.
        state = rebuild_state_from_kv(
            self.flow_lm, kv_list, offset, self.max_sequence_length
        )

        # Replace NaN BOS positions with the learned bos_emb embedding.
        frame = torch.where(torch.isnan(sequence_frame), self.bos_emb, sequence_frame)

        # Run single AR step: empty text embeddings (already in KV cache from prefill).
        projected = self.flow_lm.input_linear(frame)
        empty_text = torch.zeros(1, 0, self.flow_lm.dim, dtype=frame.dtype)
        hidden = self.flow_lm.backbone(projected, empty_text, frame, model_state=state)

        if self.flow_lm.out_norm:
            hidden = self.flow_lm.out_norm(hidden)
        last_hidden = hidden[:, -1, :]
        eos_logits = self.flow_lm.out_eos(last_hidden)

        # Extract updated KV (offset is now offset+1).
        new_t = int(offset.item()) + 1
        new_kv_list, new_offset = extract_kv_tensors(self.flow_lm, state, new_t)
        return (last_hidden, eos_logits) + tuple(new_kv_list) + (new_offset,)


class FlowLMFlowWrapper(torch.nn.Module):
    def __init__(self, model: TTSModel):
        super().__init__()
        self.flow_net = model.flow_lm.flow_net

    def forward(self, condition: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.flow_net(condition, s, t, x)


class LatentToMimiWrapper(torch.nn.Module):
    def __init__(self, model: TTSModel):
        super().__init__()
        self.register_buffer("emb_std", model.flow_lm.emb_std.detach().clone())
        self.register_buffer("emb_mean", model.flow_lm.emb_mean.detach().clone())
        self.quantizer_proj = model.mimi.quantizer.output_proj

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: [B, T, ldim]
        # Apply flow normalization stats, then quantizer projection to Mimi decoder dim.
        denorm = latent * self.emb_std + self.emb_mean
        transposed = denorm.transpose(-1, -2)  # [B, ldim, T]
        return self.quantizer_proj(transposed)  # [B, mimi_dim, T]


class MimiEncoderWrapper(torch.nn.Module):
    def __init__(self, model: TTSModel):
        super().__init__()
        self.mimi = model.mimi

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.mimi.encode_to_latent(audio)


class MimiDecoderWrapper(torch.nn.Module):
    def __init__(self, model: TTSModel, max_sequence_length: int = 256):
        super().__init__()
        self.mimi = model.mimi
        self.base_state = init_states(self.mimi, batch_size=1, sequence_length=max_sequence_length)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        state = clone_model_state(self.base_state)
        return self.mimi.decode_from_latent(latent, state)


def export_one(spec: ExportSpec, out_dir: Path) -> Path:
    out_path = out_dir / spec.filename
    spec.module.eval()

    with torch.no_grad():
        torch.onnx.export(
            spec.module,
            spec.example_inputs,
            out_path.as_posix(),
            input_names=spec.input_names,
            output_names=spec.output_names,
            dynamic_axes=spec.dynamic_axes,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"exported {spec.name} -> {out_path}")
    return out_path


def quantize_int8(path: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "--int8 requested but onnxruntime quantization is unavailable; "
            "install onnxruntime in the selected python environment"
        ) from exc

    tmp = path.with_suffix(".int8.tmp.onnx")
    quantize_dynamic(path.as_posix(), tmp.as_posix(), weight_type=QuantType.QInt8)
    shutil.move(tmp.as_posix(), path.as_posix())
    print(f"quantized INT8 -> {path}")


def tensor_shape_to_json(tensor_type: onnx.TypeProto.Tensor) -> list[Any]:
    dims = []
    for d in tensor_type.shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        elif d.dim_value:
            dims.append(int(d.dim_value))
        else:
            dims.append("?")
    return dims


def inspect_onnx(path: Path) -> dict[str, Any]:
    model = onnx.load(path.as_posix())
    graph = model.graph

    def to_entries(values: list[onnx.ValueInfoProto]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for v in values:
            tt = v.type.tensor_type
            entries.append(
                {
                    "name": v.name,
                    "dtype": TensorProto.DataType.Name(tt.elem_type).lower(),
                    "shape": tensor_shape_to_json(tt),
                }
            )
        return entries

    return {
        "filename": path.name,
        "inputs": to_entries(list(graph.input)),
        "outputs": to_entries(list(graph.output)),
    }


def build_specs(model: TTSModel, max_sequence_length: int = 256) -> list[ExportSpec]:
    # Determine KV-cache layer count and dimensions for prefill/step specs.
    _num_kv_layers = sum(
        1 for _, m in model.flow_lm.named_modules() if hasattr(m, "_cache_backend")
    )
    _num_heads = model.flow_lm.transformer.layers[0].self_attn.num_heads
    _head_dim = model.flow_lm.transformer.layers[0].self_attn.dim_per_head
    _T_ex = 8  # example text token count for tracing
    _example_kv = [
        torch.zeros(2, 1, _T_ex, _num_heads, _head_dim) for _ in range(_num_kv_layers)
    ]
    _example_offset = torch.tensor([_T_ex], dtype=torch.long)
    _kv_names = [f"kv_{i}" for i in range(_num_kv_layers)]

    return [
        ExportSpec(
            name="text_conditioner",
            filename="text_conditioner.onnx",
            input_names=["tokens"],
            output_names=["text_embeddings"],
            dynamic_axes={"tokens": {1: "text_tokens"}, "text_embeddings": {1: "text_tokens"}},
            example_inputs=(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long),),
            module=TextConditionerWrapper(model),
        ),
        ExportSpec(
            name="flow_lm_main",
            filename="flow_lm_main.onnx",
            input_names=["sequence", "text_embeddings"],
            output_names=["last_hidden", "eos_logits"],
            dynamic_axes={
                "sequence": {1: "sequence_steps"},
                "text_embeddings": {1: "text_tokens"},
            },
            example_inputs=(
                # First position is NaN (BOS sentinel); rest are normal latents.
                # This ensures torch.isnan() is always traced into the ONNX graph.
                torch.cat([
                    torch.full((1, 1, 32), float("nan"), dtype=torch.float32),
                    torch.randn(1, 7, 32, dtype=torch.float32),
                ], dim=1),
                torch.randn(1, 8, 1024, dtype=torch.float32),
            ),
            module=FlowLMMainWrapper(model, max_sequence_length=max_sequence_length),
        ),
        ExportSpec(
            name="flow_lm_prefill",
            filename="flow_lm_prefill.onnx",
            input_names=["text_embeddings"],
            output_names=_kv_names + ["offset"],
            dynamic_axes={
                "text_embeddings": {1: "text_tokens"},
                **{f"kv_{i}": {2: "text_tokens"} for i in range(_num_kv_layers)},
            },
            example_inputs=(torch.randn(1, _T_ex, 1024),),
            module=FlowLMPrefillWrapper(model, max_sequence_length=max_sequence_length),
        ),
        ExportSpec(
            name="flow_lm_step",
            filename="flow_lm_step.onnx",
            input_names=["sequence_frame"] + _kv_names + ["offset"],
            output_names=["last_hidden", "eos_logits"] + _kv_names + ["offset"],
            dynamic_axes={
                **{f"kv_{i}": {2: "seq_len"} for i in range(_num_kv_layers)},
            },
            example_inputs=(
                torch.full((1, 1, 32), float("nan")),
                *_example_kv,
                _example_offset,
            ),
            module=FlowLMStepWrapper(model, max_sequence_length=max_sequence_length),
        ),
        ExportSpec(
            name="flow_lm_flow",
            filename="flow_lm_flow.onnx",
            input_names=["condition", "s", "t", "x"],
            output_names=["flow_direction"],
            dynamic_axes={},
            example_inputs=(
                torch.randn(1, 1024, dtype=torch.float32),
                torch.zeros(1, 1, dtype=torch.float32),
                torch.ones(1, 1, dtype=torch.float32),
                torch.randn(1, 32, dtype=torch.float32),
            ),
            module=FlowLMFlowWrapper(model),
        ),
        ExportSpec(
            name="latent_to_mimi",
            filename="latent_to_mimi.onnx",
            input_names=["latent"],
            output_names=["mimi_latent"],
            dynamic_axes={"latent": {1: "latent_steps"}, "mimi_latent": {2: "latent_steps"}},
            example_inputs=(torch.randn(1, 13, 32, dtype=torch.float32),),
            module=LatentToMimiWrapper(model),
        ),
        ExportSpec(
            name="mimi_encoder",
            filename="mimi_encoder.onnx",
            input_names=["audio"],
            output_names=["latent"],
            dynamic_axes={"audio": {2: "audio_samples"}, "latent": {2: "latent_steps"}},
            example_inputs=(torch.randn(1, 1, 24000, dtype=torch.float32),),
            module=MimiEncoderWrapper(model),
        ),
        ExportSpec(
            name="mimi_decoder",
            filename="mimi_decoder.onnx",
            input_names=["latent"],
            output_names=["audio"],
            dynamic_axes={"latent": {2: "latent_steps"}, "audio": {2: "audio_samples"}},
            example_inputs=(torch.randn(1, 512, 13, dtype=torch.float32),),
            module=MimiDecoderWrapper(model, max_sequence_length=max_sequence_length),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PocketTTS subgraphs to ONNX")
    parser.add_argument("--models-dir", default="models", help="Directory containing downloaded checkpoints")
    parser.add_argument("--out-dir", default="models/onnx", help="Output directory for ONNX files")
    parser.add_argument("--variant", default="b6369a24", help="PocketTTS model variant/config")
    parser.add_argument("--int8", action="store_true", help="Apply dynamic INT8 quantization to exported ONNX files")
    parser.add_argument("--max-seq", type=int, default=256, help="KV-cache max sequence length for flow_lm_main and mimi_decoder (default: 256; use 512+ when using voice conditioning)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise SystemExit(f"models-dir does not exist: {models_dir}")

    # PocketTTS itself resolves downloaded files from HF cache/references; this ensures callers
    # can override environment/model placement and still keep CLI contract explicit.
    os.environ.setdefault("POCKETTTS_MODELS_DIR", models_dir.as_posix())

    print(f"loading pocket-tts model variant={args.variant}")
    model = TTSModel.load_model(args.variant)

    specs = build_specs(model, max_sequence_length=args.max_seq)
    manifest: dict[str, Any] = {
        "variant": args.variant,
        "int8": bool(args.int8),
        "graphs": [],
    }

    for spec in specs:
        out_path = export_one(spec, out_dir)
        if args.int8:
            quantize_int8(out_path)
        manifest["graphs"].append(
            {
                "name": spec.name,
                "size_bytes": int(out_path.stat().st_size),
                **inspect_onnx(out_path),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote ONNX manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
