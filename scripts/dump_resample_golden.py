#!/usr/bin/env python3
"""Dump upstream convert_audio (scipy resample_poly) outputs for Go tests.

Run this from the Go repo root with the upstream checkout installed at the sync
target (see docs/INSTALL.md, "Parity setup"):

    original/pockettts/.venv/bin/python scripts/dump_resample_golden.py \
      --output internal/audio/testdata/resample_poly.json
    prettier -w internal/audio/testdata/resample_poly.json

Inputs are not stored: TestResamplePoly rebuilds them with the same formula
(promptTestSignal in internal/audio/resample_test.go).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy
import torch

from pocket_tts.data.audio_utils import convert_audio

TARGET_RATE = 24000

# (name, source rate, input samples)
CASES = [
    ("8k_up", 8000, 240),
    ("16k_up", 16000, 480),
    ("22050_down", 22050, 662),
    ("44100_down", 44100, 1323),
    ("48k_down", 48000, 1440),
    ("48k_odd_length", 48000, 1001),
    ("44100_short", 44100, 5),
    ("24k_identity", 24000, 64),
]


def signal(rate: int, n: int) -> np.ndarray:
    """Chirp + tone + deterministic pseudo-noise, computed in float64 then
    rounded to float32 like a decoded prompt."""
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        t = i / rate
        chirp = 0.5 * math.sin(2 * math.pi * 200 * t + math.pi * 40000 * t * t)
        tone = 0.25 * math.sin(2 * math.pi * 3100 * t)
        noise = 0.1 * ((i * 7919 % 1000) / 500 - 1)
        out[i] = chirp + tone + noise
    return out.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = []
    for name, rate, n in CASES:
        wav = torch.from_numpy(signal(rate, n)).unsqueeze(0)
        out = convert_audio(wav, rate, TARGET_RATE, 1)
        assert out.dtype == torch.float32, out.dtype
        cases.append(
            {
                "name": name,
                "from_rate": rate,
                "to_rate": TARGET_RATE,
                "n_in": n,
                # 9 significant digits round-trip every float32 exactly.
                "output": [float(f"{v:.9g}") for v in out[0].tolist()],
            }
        )

    fixture = {
        "source": f"pocket_tts.data.audio_utils.convert_audio, scipy {scipy.__version__}",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
