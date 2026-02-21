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

    def forward(self, sequence: torch.Tensor, text_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = clone_model_state(self.base_state)
        projected = self.flow_lm.input_linear(sequence)
        hidden = self.flow_lm.backbone(projected, text_embeddings, sequence, model_state=state)
        last_hidden = hidden[:, -1, :]
        eos_logits = self.flow_lm.out_eos(last_hidden)
        return last_hidden, eos_logits


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


def build_specs(model: TTSModel) -> list[ExportSpec]:
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
                torch.randn(1, 8, 32, dtype=torch.float32),
                torch.randn(1, 8, 1024, dtype=torch.float32),
            ),
            module=FlowLMMainWrapper(model),
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
            module=MimiDecoderWrapper(model),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PocketTTS subgraphs to ONNX")
    parser.add_argument("--models-dir", default="models", help="Directory containing downloaded checkpoints")
    parser.add_argument("--out-dir", default="models/onnx", help="Output directory for ONNX files")
    parser.add_argument("--variant", default="b6369a24", help="PocketTTS model variant/config")
    parser.add_argument("--int8", action="store_true", help="Apply dynamic INT8 quantization to exported ONNX files")
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

    specs = build_specs(model)
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
                **inspect_onnx(out_path),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote ONNX manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
