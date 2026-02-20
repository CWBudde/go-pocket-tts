# go-pocket-tts

Go CLI (and a small HTTP server skeleton) for working with [PocketTTS](https://github.com/kyutai-labs/pocket-tts):

- Run text-to-speech via the `pocket-tts` executable
- Download PocketTTS model checkpoints from Hugging Face
- Export PocketTTS PyTorch checkpoints to ONNX subgraphs + manifest
- Smoke-verify exported ONNX graphs
- Export voice embeddings (`.safetensors`) from a WAV prompt

## Status

- CLI commands (`synth`, `model *`, `export-voice`, `doctor`, `health`) are implemented.
- The HTTP server currently only exposes `/healthz`. The `/v1/synth` route is present but **returns 501 (not implemented)**.

## Requirements

- Go `1.25+`
- For CLI synthesis and voice export:
  - `pocket-tts` installed and available in `PATH`, or set `--tts-cli-path` / `POCKETTTS_TTS_CLI_PATH`
- For ONNX export:
  - Python environment that can import: `torch`, `onnx`, and `pocket_tts` (from PocketTTS)
  - Optional for `--int8`: Python `onnxruntime` with quantization support
- For ONNX verify:
  - Python environment that can import: `numpy` and `onnxruntime`
- For native ONNX Runtime usage in the Go runtime bootstrap:
  - An ONNX Runtime shared library (see [docs/INSTALL.md](docs/INSTALL.md))

## Build

```bash
go build -o pockettts ./cmd/pockettts
```

Or run without building a binary:

```bash
go run ./cmd/pockettts --help
```

## Quickstart (CLI)

### 1) Download model files

Downloads into `./models` by default.

```bash
./pockettts model download
```

If the Hugging Face repo is gated, provide a token:

```bash
export HF_TOKEN=...  # or use --hf-token
./pockettts model download --hf-repo kyutai/pocket-tts
```

### 2) Sanity-check your setup

Checks:

- `pocket-tts` executable is available
- configured model path exists
- configured voice path exists (if set)

```bash
./pockettts doctor
```

If ONNX Runtime can’t be found automatically, point to it (see [docs/INSTALL.md](docs/INSTALL.md)):

```bash
./pockettts doctor --ort-lib /usr/local/lib/libonnxruntime.so
```

### 3) Synthesize audio

```bash
./pockettts synth --text "Hello from PocketTTS" --out out.wav
```

Override voice for a single request:

```bash
./pockettts synth --text "Hello" --voice mimi --out out.wav
```

Write the WAV to stdout:

```bash
./pockettts synth --text "Hello" --out - > out.wav
```

## Model export + verify (ONNX)

### Export

```bash
./pockettts model export --models-dir models --out-dir models/onnx
```

Optional INT8 quantization:

```bash
./pockettts model export --models-dir models --out-dir models/onnx --int8
```

### Verify

Runs a small Python smoke inference for each graph in the exported `manifest.json`.

```bash
./pockettts model verify --manifest models/onnx/manifest.json
```

If you need to provide the ONNX Runtime shared library path to Python:

```bash
export ORT_LIBRARY_PATH=/usr/local/lib/libonnxruntime.so
./pockettts model verify --manifest models/onnx/manifest.json
```

## Export a voice embedding

Exports a `.safetensors` voice embedding from a speaker WAV prompt and prints a suggested `voices/manifest.json` entry.

```bash
./pockettts export-voice --audio speaker.wav --out voices/my_voice.safetensors --id my-voice --license "CC-BY-4.0"
```

See [voices/README.md](voices/README.md) for licensing guidance.

## Server

Start the server (HTTP):

```bash
./pockettts serve
```

Health check:

```bash
curl -s http://localhost:8080/healthz
```

The CLI also has a probe command:

```bash
./pockettts health
./pockettts health --addr localhost:8080
```

## Configuration

Configuration is loaded in this order:

1. Flags (see `./pockettts --help`)
2. Environment variables with prefix `POCKETTTS_`
3. Config file passed with `--config`
4. Optional local config file named `pockettts.(yaml|yml|toml|json)` in the working directory

### Example `pockettts.yaml`

```yaml
paths:
  model_path: models/model.onnx
  voice_path: models/voice.bin

runtime:
  threads: 4
  inter_op_threads: 1
  # ort_library_path: /usr/local/lib/libonnxruntime.so
  # ort_version: "1.18.0"

server:
  listen_addr: ":8080"
  grpc_addr: ":9090"

tts:
  # Voice name/ID (or a .safetensors path, depending on your PocketTTS setup)
  voice: "mimi"
  # Path to the pocket-tts executable (leave empty to use PATH)
  cli_path: ""
  # Optional PocketTTS config path
  cli_config_path: ""
  concurrency: 1
  quiet: true
```

### Useful environment variables

- `POCKETTTS_TTS_CLI_PATH` (points to `pocket-tts`)
- `POCKETTTS_TTS_VOICE`
- `POCKETTTS_SERVER_LISTEN_ADDR`
- `POCKETTTS_RUNTIME_ORT_LIBRARY_PATH` (or `POCKETTTS_ORT_LIB`, or `ORT_LIBRARY_PATH`)

## Development

This repo uses `just` for common workflows:

```bash
just fmt
just test
just lint
just ci
```
