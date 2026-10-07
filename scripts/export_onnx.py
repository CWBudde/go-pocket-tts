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
import contextlib
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

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules import attention as pocket_attention
from pocket_tts.modules import transformer as pocket_transformer
from pocket_tts.modules.stateful_module import init_states


OPSET_VERSION = 17
LEGACY_DEFAULT_VARIANT = "b6369a24"
DEFAULT_LANGUAGE = "english_2026-01"


@dataclass
class ExportSpec:
    name: str
    filename: str
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]]
    example_inputs: tuple[torch.Tensor, ...]
    module: torch.nn.Module


def uncached_causal_mask(t: int, context: int | None, device: torch.device) -> torch.Tensor:
    """Build the stateless attention mask on every call.

    Upstream caches the mask in a module-global dict at the largest t seen, so
    a trace would bake that tensor in as a constant sized to the trace input.
    """
    pos = torch.arange(t, device=device, dtype=torch.long).view(1, -1)
    return pocket_attention._build_attention_mask(pos, pos, context)


def patch_for_tracing() -> None:
    # transformer imports the function by name, so patch both modules.
    pocket_attention._cached_causal_mask = uncached_causal_mask
    pocket_transformer._cached_causal_mask = uncached_causal_mask


def clone_model_state(state: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for module_name, module_state in state.items():
        out[module_name] = {k: v.clone() for k, v in module_state.items()}
    return out


def extract_kv_tensors(
    flow_lm: "torch.nn.Module",
    state: "dict[str, dict[str, torch.Tensor]]",
    t_written: int,
) -> "list[torch.Tensor]":
    """Extract per-layer KV tensors from model_state after prefill.

    kv_list[i] is the [2, B, t_written, H, Dh] slice of layer i's cache.
    """
    kv_list = []
    for _module_name, module in flow_lm.named_modules():
        if not hasattr(module, "_cache_backend"):
            continue
        layer_state = state[module._module_absolute_name]
        # cache shape: [2, B, max_seq, H, Dh]; slice to written portion
        kv = layer_state["cache"][:, :, :t_written, :, :]
        kv_list.append(kv)
    return kv_list


def append_kv_by_concat(
    self: "pocket_attention._LinearKVCacheBackend",
    k: torch.Tensor,
    v: torch.Tensor,
    state: "dict[str, torch.Tensor] | None",
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Trace-friendly _LinearKVCacheBackend.append_and_get for flow_lm_step.

    Upstream writes k/v into a preallocated cache at int(offset.item()), which
    a trace freezes at the example offset. Here the cache holds exactly the
    written positions and grows by concatenation, so no Python int is taken
    from a tensor.
    """
    cache = torch.cat([state["cache"], torch.stack([k, v])], dim=2)
    state["cache"] = cache
    k_attn = cache[0].permute(0, 2, 1, 3)
    v_attn = cache[1].permute(0, 2, 1, 3)
    pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long)
    pos_k = pos_k.view(1, -1).expand(k_attn.shape[0], -1)
    return k_attn, v_attn, pos_k, state["offset"]


@contextlib.contextmanager
def kv_append_by_concat():
    backend = pocket_attention._LinearKVCacheBackend
    original = backend.append_and_get
    backend.append_and_get = append_kv_by_concat
    try:
        yield
    finally:
        backend.append_and_get = original


class TextConditionerWrapper(torch.nn.Module):
    def __init__(self, model: TTSModel):
        super().__init__()
        self.conditioner = model.flow_lm.conditioner

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.conditioner(tokens)


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

        kv_list = extract_kv_tensors(self.flow_lm, state, T)
        # T is a Python int, which a trace freezes; count the rows instead so
        # the offset follows the input length.
        offset = torch.ones_like(text_embeddings[0, :, 0], dtype=torch.long).sum().view(1)
        return tuple(kv_list) + (offset,)


class FlowLMStepWrapper(torch.nn.Module):
    """Runs a single autoregressive step with explicit KV-cache I/O.

    Accepts sequence_frame [1, 1, 32], per-layer KV tensors, and offset as inputs.
    Returns last_hidden [1, 1024], eos_logits [1, 1], updated KV tensors, and
    updated offset. The Go caller maintains the KV state between steps.
    """

    def __init__(self, model: TTSModel):
        super().__init__()
        self.flow_lm = model.flow_lm
        self.register_buffer("bos_emb", model.flow_lm.bos_emb.detach().clone())
        self.kv_module_names = [
            m._module_absolute_name for _, m in model.flow_lm.named_modules() if hasattr(m, "_cache_backend")
        ]

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

        # Each layer's cache is exactly its kv input; kv_append_by_concat
        # appends this step's k/v to it.
        state = {name: {"cache": kv, "offset": offset} for name, kv in zip(self.kv_module_names, kv_list)}

        # Replace NaN BOS positions with the learned bos_emb embedding.
        frame = torch.where(torch.isnan(sequence_frame), self.bos_emb, sequence_frame)

        # Run single AR step: empty text embeddings (already in KV cache from prefill).
        projected = self.flow_lm.input_linear(frame)
        empty_text = torch.zeros(1, 0, self.flow_lm.dim, dtype=frame.dtype)
        # backbone applies out_norm itself, as in flow_lm_main.
        with kv_append_by_concat():
            hidden = self.flow_lm.backbone(projected, empty_text, frame, model_state=state)
        last_hidden = hidden[:, -1, :]
        eos_logits = self.flow_lm.out_eos(last_hidden)

        new_kv_list = [state[name]["cache"] for name in self.kv_module_names]
        return (last_hidden, eos_logits) + tuple(new_kv_list) + (offset + 1,)


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
        # encode_to_latent returns [B, T, 512]; the graph keeps [B, 512, T].
        return self.mimi.encode_to_latent(audio).transpose(-1, -2)


class MimiDecoderWrapper(torch.nn.Module):
    def __init__(self, model: TTSModel, max_latent_steps: int = 256):
        super().__init__()
        self.mimi = model.mimi
        mimi_steps_per_latent = int(round(model.mimi.encoder_frame_rate / model.mimi.frame_rate))
        decoder_sequence_length = max_latent_steps * mimi_steps_per_latent
        self.base_state = init_states(
            self.mimi, batch_size=1, sequence_length=decoder_sequence_length
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent is latent_to_mimi's output, already through the quantizer
        # projection, so this is decode_from_latent without its first step.
        state = clone_model_state(self.base_state)
        emb = self.mimi._to_encoder_framerate(latent, state)
        (emb,) = self.mimi.decoder_transformer(emb, state)
        return self.mimi.decoder(emb, state)


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
    _kv_out_names = [f"kv_out_{i}" for i in range(_num_kv_layers)]

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
            output_names=["last_hidden", "eos_logits"] + _kv_out_names + ["offset_out"],
            dynamic_axes={
                **{f"kv_{i}": {2: "seq_len"} for i in range(_num_kv_layers)},
                **{f"kv_out_{i}": {2: "seq_len_plus_one"} for i in range(_num_kv_layers)},
            },
            example_inputs=(
                torch.full((1, 1, 32), float("nan")),
                *_example_kv,
                _example_offset,
            ),
            module=FlowLMStepWrapper(model),
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
            # Mimi contains shape-dependent streaming/padding branches that are
            # traced as constants by the legacy ONNX exporter. Trace at the
            # configured maximum latent length, while the wrapper sizes Mimi's
            # internal state to max_latents * mimi_steps_per_latent.
            example_inputs=(torch.randn(1, 512, max_sequence_length, dtype=torch.float32),),
            module=MimiDecoderWrapper(model, max_latent_steps=max_sequence_length),
        ),
    ]


def resolve_model_source(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve CLI model selection into current upstream TTSModel.load_model kwargs.

    The Go tooling historically passed --variant=b6369a24. Latest pocket-tts
    selects models by language or explicit config path, so keep the legacy flag
    as a compatibility alias while exposing the new API directly.
    """
    if args.config and args.language:
        raise SystemExit("--language and --config are mutually exclusive")

    if args.config:
        return {"config": args.config}, {"config": args.config}

    if args.language:
        return {"language": args.language}, {"language": args.language}

    variant = args.variant or LEGACY_DEFAULT_VARIANT
    if variant == LEGACY_DEFAULT_VARIANT:
        return {"language": DEFAULT_LANGUAGE}, {
            "language": DEFAULT_LANGUAGE,
            "legacy_variant": variant,
        }

    if variant.endswith((".yaml", ".yml")):
        return {"config": variant}, {
            "config": variant,
            "legacy_variant": variant,
        }

    return {"language": variant}, {
        "language": variant,
        "legacy_variant": variant,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PocketTTS subgraphs to ONNX")
    parser.add_argument("--models-dir", default="models", help="Directory containing downloaded checkpoints")
    parser.add_argument("--out-dir", default="models/onnx", help="Output directory for ONNX files")
    parser.add_argument("--language", help=f"PocketTTS language/config name (default: {DEFAULT_LANGUAGE})")
    parser.add_argument("--config", help="Path to an upstream PocketTTS config .yaml")
    parser.add_argument(
        "--variant",
        default=LEGACY_DEFAULT_VARIANT,
        help="Deprecated compatibility alias; b6369a24 maps to english_2026-01",
    )
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

    load_kwargs, manifest_source = resolve_model_source(args)
    source_label = ", ".join(f"{key}={value}" for key, value in load_kwargs.items())
    print(f"loading pocket-tts model {source_label}")
    model = TTSModel.load_model(**load_kwargs)
    if model.flow_lm.flow_type != "lsd":
        # flow_lm_flow takes (condition, s, t, x): two time conditions, as only
        # the lsd flow head has.
        raise SystemExit(f"flow type {model.flow_lm.flow_type!r} is not supported; only lsd configs export")
    patch_for_tracing()

    specs = build_specs(model, max_sequence_length=args.max_seq)
    manifest: dict[str, Any] = {
        "variant": args.variant,
        **manifest_source,
        "int8": bool(args.int8),
        "sample_rate": int(model.mimi.sample_rate),
        "graphs": [],
    }
    if model.flow_lm.insert_bos_before_voice:
        # Upstream puts this learned embedding in front of a voice prompt; the
        # graphs never see the voice, so the Go caller prepends it.
        manifest["bos_before_voice"] = model.flow_lm.bos_before_voice.detach().reshape(-1).tolist()

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
