# Installation

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
