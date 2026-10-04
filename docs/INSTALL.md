# Installation

## Runtime vs tooling dependencies

- Runtime commands (`synth`, `export-voice`, `serve`, `doctor`, `model verify`) can run without Python when using backend `native`.
- Tooling command (`pockettts-tools model export`) requires Python tooling.

## ONNX Runtime shared library

`pockettts` needs an ONNX Runtime shared library for native ONNX execution paths.

You can provide it in either way:

1. System package manager (recommended)

- Linux: install `libonnxruntime` from your distro packages, then verify files such as
  - `/usr/lib/libonnxruntime.so`
  - `/usr/local/lib/libonnxruntime.so`
- macOS (Homebrew, Apple Silicon):

  ```bash
  brew install onnxruntime
  export POCKETTTS_ORT_LIB=/opt/homebrew/lib/libonnxruntime.dylib
  ```

  ONNX Runtime is optional: the default `native-safetensors` backend does not
  need it. Without it, tests that need ONNX Runtime skip.

2. Manual download

- Download ONNX Runtime release binaries from Microsoft.
- Place the shared library in a stable location and pass its path to `pockettts`.

## How to point `pockettts` to the library

Priority order used by runtime bootstrap:

1. `--ort-lib` (alias for `--runtime-ort-library-path`)
2. `POCKETTTS_ORT_LIB`
3. `ORT_LIBRARY_PATH`
4. built-in platform path candidates

Examples:

```bash
pockettts doctor --ort-lib /usr/local/lib/libonnxruntime.so
```

```bash
export POCKETTTS_ORT_LIB=/usr/local/lib/libonnxruntime.so
pockettts doctor
```

## Tooling prerequisites (Python)

Only needed for export tooling commands:

- `pockettts-tools model export`:
  - Python `>=3.10,<3.15`
  - importable modules: `pocket_tts`, `torch`, `onnx`
  - optional for `--int8`: `onnxruntime`
    To force compatibility mode that uses the Python CLI for synthesis:

```bash
pockettts synth --backend cli --text "Hello" --out out.wav
```

## Parity setup (Python reference)

The native parity fixtures (`scripts/dump_python_parity.py`) and the
model-state voice export fallback run against a local checkout of upstream
PocketTTS in `original/pockettts` (gitignored). Check out the sync target
(`41cbc84`, upstream 3.3.0 plus 13 commits; see `PLAN.md`), not upstream `main`:

```bash
git clone https://github.com/kyutai-labs/pocket-tts original/pockettts
git -C original/pockettts checkout 41cbc84af539ea78a804ffca5f9c6edc1a22ce44
cd original/pockettts && UV_PYTHON=3.12 uv sync --no-dev && cd ../..
```

The committed fixtures in `internal/native/testdata/python_parity/` run with
`go test ./internal/native` (each skips when its local model is missing).
Regenerate one from the local model, tokenizer and default voice, so Python and
Go read the same files:

```bash
original/pockettts/.venv/bin/python scripts/dump_python_parity.py \
  --language german \
  --weights models/german/model.safetensors \
  --tokenizer models/german/tokenizer.json \
  --voice voices/german/juergen.safetensors \
  --output internal/native/testdata/python_parity/german.json
prettier -w internal/native/testdata/python_parity/german.json
```

For `english_2026-01` use `models/tts_b6369a24.safetensors`,
`models/tokenizer.json` and `voices/alba.safetensors`. A fixture outside
`testdata/` runs too when `POCKETTTS_NATIVE_PY_FIXTURE` points at it.

Notes:

- Upstream's `.python-version` says 3.10, but its scipy wheel for 3.10 does not
  load on macOS 27 (`__DATA/__thread_bss` dyld error); `UV_PYTHON=3.12` avoids it.
- Upstream's `pyproject.toml` sets `[tool.uv] exclude-newer = "7 days"`, so
  `uv sync` ignores package releases from the last week.
- `--no-dev` skips the `dev` dependency group, which in 3.x pulls in training
  and evaluation packages (torchaudio, transformers, UTMOS scoring). The parity
  script only needs inference.
