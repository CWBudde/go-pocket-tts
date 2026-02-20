# Installation

## Runtime vs tooling dependencies

- Runtime commands (`synth`, `serve`, `doctor`, `model verify`) can run without Python when using backend `native`.
- Tooling commands (`pockettts-tools model export`, `pockettts-tools export-voice`) require Python tooling.

## ONNX Runtime shared library

`pockettts` needs an ONNX Runtime shared library for native ONNX execution paths.

You can provide it in either way:

1. System package manager (recommended)

- Linux: install `libonnxruntime` from your distro packages, then verify files such as
  - `/usr/lib/libonnxruntime.so`
  - `/usr/local/lib/libonnxruntime.so`
- macOS (Homebrew): install ONNX Runtime and verify
  - `/opt/homebrew/lib/libonnxruntime.dylib`

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
- `pockettts-tools export-voice`:
  - `pocket-tts` CLI installed (Python package)

To force compatibility mode that uses the Python CLI for synthesis:

```bash
pockettts synth --backend cli --text "Hello" --out out.wav
```
