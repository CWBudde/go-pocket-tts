#!/usr/bin/env python3
"""Dump upstream PocketTTS tensors for native Go runtime parity tests.

Run this from the Go repo root after installing the upstream checkout:

    cd original/pockettts
    uv sync --all-extras
    cd ../..
    original/pockettts/.venv/bin/python scripts/dump_python_parity.py \
      --output tests/parity/native_runtime.json

Then run:

    POCKETTTS_NATIVE_PY_FIXTURE=tests/parity/native_runtime.json go test ./internal/native
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    args = parse_args()
    upstream = args.upstream.resolve()
    if not (upstream / "pocket_tts").is_dir():
        print(f"upstream checkout not found at {upstream}", file=sys.stderr)
        return 2

    sys.path.insert(0, upstream.as_posix())

    try:
        import torch
        from pocket_tts.conditioners.base import TokenizedText
        from pocket_tts.models.tts_model import TTSModel
        from pocket_tts.modules.stateful_module import increment_steps, init_states
    except ModuleNotFoundError as exc:
        print(
            f"missing Python dependency {exc.name!r}; run `uv sync --all-extras` in {upstream}",
            file=sys.stderr,
        )
        return 2

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    if args.config is not None:
        model = TTSModel.load_model(config=args.config)
        source_config = args.config
    else:
        model = TTSModel.load_model(language=args.language)
        source_config = args.language
    model.eval()

    fixture: dict[str, Any] = {
        "source": {
            "upstream": upstream.as_posix(),
            "config": source_config,
            "seed": args.seed,
        }
    }
    fixture["flow_lm_prefill_step"] = dump_flow_lm_prefill_step(
        torch,
        TokenizedText,
        init_states,
        increment_steps,
        model,
        parse_ints(args.flow_tokens),
        args.flow_cache_length,
    )
    fixture["mimi"] = [
        dump_mimi_case(torch, init_states, model, frames, args.mimi_cache_length)
        for frames in parse_ints(args.mimi_frames)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(args.output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("original/pockettts"))
    parser.add_argument("--language", default="english_2026-01")
    parser.add_argument("--config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--flow-tokens", default="10,20,30")
    parser.add_argument("--flow-cache-length", type=int, default=64)
    parser.add_argument("--mimi-frames", default="1,2,4")
    parser.add_argument("--mimi-cache-length", type=int, default=64)
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def dump_flow_lm_prefill_step(
    torch: Any,
    tokenized_text_type: Any,
    init_states: Any,
    increment_steps: Any,
    model: Any,
    tokens: list[int],
    cache_length: int,
) -> dict[str, Any]:
    flow = model.flow_lm
    text_tokens = torch.tensor([tokens], dtype=torch.int64, device=flow.device)
    text_embeddings = flow.conditioner(tokenized_text_type(text_tokens))
    state = init_states(flow, batch_size=1, sequence_length=cache_length)

    with torch.no_grad():
        _ = flow.transformer(text_embeddings, state)
        increment_steps(flow, state, increment=text_embeddings.shape[1])
        prompt_offsets = state_offsets(state)

        step_latent = deterministic_tensor(torch, (1, 1, flow.ldim), scale=0.05)
        step_input = flow.input_linear(step_latent)
        step_out = flow.transformer(step_input, state)
        increment_steps(flow, state, increment=step_input.shape[1])
        step_offsets = state_offsets(state)

        step_out = flow.out_norm(step_out.to(torch.float32))
        last_hidden = step_out[:, -1]
        eos_logits = flow.out_eos(last_hidden)

    return {
        "tokens": tokens,
        "step_latent": tensor_to_json(step_latent),
        "prompt_layer_offsets": prompt_offsets,
        "step_layer_offsets": step_offsets,
        "step_last_hidden": tensor_to_json(last_hidden),
        "step_eos_logits": tensor_to_json(eos_logits),
    }


def dump_mimi_case(
    torch: Any,
    init_states: Any,
    model: Any,
    frames: int,
    cache_length: int,
) -> dict[str, Any]:
    flow = model.flow_lm
    mimi = model.mimi
    latent = deterministic_tensor(torch, (1, frames, flow.ldim), scale=0.03)
    with torch.no_grad():
        mimi_input = latent * flow.emb_std + flow.emb_mean
        mimi_input = mimi_input.transpose(-1, -2)
        quantized = mimi.quantizer(mimi_input)

        mimi_steps_per_latent = int(mimi.encoder_frame_rate / mimi.frame_rate)
        sequence_length = max(cache_length, frames * mimi_steps_per_latent)
        mimi_state = init_states(mimi, batch_size=1, sequence_length=sequence_length)
        audio = mimi.decode_from_latent(quantized, mimi_state)

    return {
        "name": f"{frames}_frames",
        "latent": tensor_to_json(latent),
        "latent_to_mimi": tensor_to_json(quantized),
        "mimi_decode": tensor_to_json(audio),
    }


def deterministic_tensor(torch: Any, shape: tuple[int, ...], scale: float) -> Any:
    count = 1
    for dim in shape:
        count *= dim
    values = torch.arange(count, dtype=torch.float32)
    values = ((values % 23) - 11) * scale
    return values.reshape(shape)


def state_offsets(state: dict[str, dict[str, Any]]) -> list[int]:
    offsets: list[int] = []
    for _, module_state in sorted(state.items()):
        offset = module_state.get("offset")
        if offset is not None:
            offsets.append(int(offset.reshape(-1)[0].item()))
    return offsets


def tensor_to_json(tensor: Any) -> dict[str, Any]:
    cpu = tensor.detach().to(dtype=tensor.float().dtype).cpu().contiguous()
    return {
        "shape": list(cpu.shape),
        "data": [float(x) for x in cpu.reshape(-1).tolist()],
    }


if __name__ == "__main__":
    raise SystemExit(main())
