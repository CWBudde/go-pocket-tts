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
PocketTTS in `original/pockettts` (gitignored). Check out the commit the port
is synced to (see "Upstream alignment" in the README), not upstream `main`:

```bash
git clone https://github.com/kyutai-labs/pocket-tts original/pockettts
git -C original/pockettts checkout 2dff8a2d1b3b21bf44ecf0084cc8ce79ab6d6bba
cd original/pockettts && uv sync --all-extras && cd ../..
original/pockettts/.venv/bin/python scripts/dump_python_parity.py \
  --output tests/parity/native_runtime.json
POCKETTTS_NATIVE_PY_FIXTURE=tests/parity/native_runtime.json go test ./internal/native
```

Notes:

- Upstream's `pyproject.toml` sets `[tool.uv] exclude-newer = "7 days"`, so
  `uv sync` ignores package releases from the last week.
- `uv sync` installs the `dev` dependency group by default. In newer upstream
  versions (3.x) it pulls in training and evaluation packages (torchaudio,
  transformers, UTMOS scoring); add `--no-dev` if you only need inference.
