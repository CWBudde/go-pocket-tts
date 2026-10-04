#!/usr/bin/env python3
"""Dump upstream Mimi encoder tensors and the voice model state for the native Go
voice-encoder and model-state export parity tests.

The encoder weights are zeroed in the ungated kyutai/pocket-tts-without-voice-cloning
checkpoints, so this needs the gated kyutai/pocket-tts weights, e.g.

    pockettts model download --language german --out-dir models/gated/german --no-voices

Then, from the Go repo root:

    original/pockettts/.venv/bin/python scripts/dump_voice_encoder_parity.py \
      --language german \
      --weights models/gated/german/model.safetensors \
      --output internal/native/testdata/python_parity/encoder_german.json
    prettier -w internal/native/testdata/python_parity/encoder_german.json

The prompt WAV (internal/native/testdata/python_parity/voice_prompt.wav) is
synthetic and deterministic; it is written when missing and reused otherwise, so
every language's fixture encodes the same file.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

from dump_python_parity import local_config, tensor_to_json, upstream_revision

SAMPLE_RATE = 24000

# 2.03 s of signal is 406 encoder frames at 200 Hz, past the encoder transformer's
# attention context of 250; the quiet 0.4 s after it is cut by end_on_pause. The
# prepared prompt is then no multiple of the 1920-sample frame, so the encoder's
# right padding runs too.
SIGNAL_SECONDS = 2.03
QUIET_SECONDS = 0.4

# Prepared samples stored at each end of the prompt.
PREPARED_EDGE_SAMPLES = 16

# safetensors dtype names of the torch dtypes in a voice model state.
SAFETENSORS_DTYPES = {"torch.float32": "F32", "torch.int64": "I64"}


def main() -> int:
    args = parse_args()
    upstream = args.upstream.resolve()
    if not (upstream / "pocket_tts").is_dir():
        print(f"upstream checkout not found at {upstream}", file=sys.stderr)
        return 2

    sys.path.insert(0, upstream.as_posix())

    try:
        import numpy as np
        import torch
        import yaml
        from pocket_tts.data.audio import audio_read
        from pocket_tts.data.audio_utils import convert_audio, end_on_pause
        from pocket_tts.models.tts_model import TTSModel
    except ModuleNotFoundError as exc:
        print(
            f"missing Python dependency {exc.name!r}; run `uv sync --no-dev` in {upstream}",
            file=sys.stderr,
        )
        return 2

    torch.set_num_threads(1)

    if not args.prompt.exists():
        write_prompt(np, args.prompt, args.seed)

    config_path = upstream / "pocket_tts" / "config" / f"{args.language}.yaml"
    with tempfile.TemporaryDirectory() as tmp:
        model = TTSModel.load_model(
            config=local_config(yaml, config_path, args.weights, None, Path(tmp))
        )
    model.eval()

    # get_state_for_audio_prompt up to _encode_audio (prompts here are < 30 s).
    audio, rate = audio_read(args.prompt)
    audio = convert_audio(audio, rate, model.config.mimi.sample_rate, 1)
    audio = end_on_pause(audio, model.config.mimi.sample_rate)

    with torch.no_grad():
        latent = model.mimi.encode_to_latent(audio.unsqueeze(0))
        conditioning = model._encode_audio(audio.unsqueeze(0))
        # What `pocket-tts export-voice` saves (export_model_state).
        model_state = model.get_state_for_audio_prompt(args.prompt, truncate=True)

    if not torch.any(latent != 0):
        print("encoder output is all zero: --weights must be the gated checkpoint", file=sys.stderr)
        return 2

    frames = latent.shape[1]
    rows = sorted({0, frames // 2, frames - 1})
    prepared = audio.reshape(-1)

    fixture = {
        "source": {
            "upstream": upstream_revision(upstream),
            "config": args.language,
            "prompt": args.prompt.name,
        },
        "prepared": {
            "length": prepared.shape[0],
            "sum": float(f"{prepared.double().sum().item():.9g}"),
            "head": tensor_to_json(prepared[:PREPARED_EDGE_SAMPLES])["data"],
            "tail": tensor_to_json(prepared[-PREPARED_EDGE_SAMPLES:])["data"],
        },
        "latent": tensor_to_json(latent),
        "conditioning_rows": {"frames": rows, **tensor_to_json(conditioning[0, rows])},
        "model_state": dump_model_state(model_state),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(args.output)
    return 0


def dump_model_state(model_state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per stateful module: the layout of every tensor export_model_state writes,
    the offset and pad, and the K and V cache rows at the first two and the last
    cached position."""
    modules = []
    positions = None
    for name, state in model_state.items():
        cache = state["cache"]
        steps = cache.shape[2]
        if positions is None:
            positions = sorted({0, 1, steps - 1})
        modules.append(
            {
                "module": name,
                "tensors": {
                    key: {
                        "dtype": SAFETENSORS_DTYPES[str(value.dtype)],
                        "shape": list(value.shape),
                    }
                    for key, value in sorted(state.items())
                },
                "offset": int(state["offset"].reshape(-1)[0]),
                "pad": int(state["pad"].reshape(-1)[0]),
                "k_rows": tensor_to_json(cache[0, 0, positions]),
                "v_rows": tensor_to_json(cache[1, 0, positions]),
            }
        )
    return {"positions": positions, "modules": modules}


def write_prompt(np: Any, path: Path, seed: int) -> None:
    """Writes a voice-like mono PCM16 prompt: amplitude-modulated harmonics with
    noise, then a quiet tail."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SIGNAL_SECONDS * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    signal = envelope * (
        0.3 * np.sin(2 * np.pi * 150 * t)
        + 0.15 * np.sin(2 * np.pi * 310 * t + 0.5)
        + 0.08 * np.sin(2 * np.pi * 1200 * t)
    )
    signal += 0.02 * rng.standard_normal(signal.shape)
    quiet = 0.002 * rng.standard_normal(int(QUIET_SECONDS * SAMPLE_RATE))
    pcm = np.clip(np.concatenate([signal, quiet]) * 32767, -32768, 32767).astype("<i2")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("original/pockettts"))
    parser.add_argument("--language", default="english_2026-01")
    parser.add_argument(
        "--weights", type=Path, required=True, help="gated checkpoint with real encoder weights"
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("internal/native/testdata/python_parity/voice_prompt.wav"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
