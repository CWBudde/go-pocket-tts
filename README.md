# go-pocket-tts

A pure-Go CLI and HTTP server for [PocketTTS](https://github.com/kyutai-labs/pocket-tts) text-to-speech synthesis. The default backend runs inference directly from safetensors weights — no Python, no ONNX Runtime required. An optional ONNX backend is available as a fallback.

- Synthesize speech from text with the native Go backend (AVX2/FMA optimized)
- Serve TTS over HTTP with concurrent worker pool and graceful shutdown
- Download PocketTTS model checkpoints from Hugging Face
- Export voice embeddings (`.safetensors`) from a WAV/PCM prompt
- Optionally export and run ONNX subgraphs via ONNX Runtime
- Run in-browser via experimental WASM kernel

## Status

All core features are implemented and functional.

- Runtime binary: `pockettts` (`synth`, `export-voice`, `serve`, `doctor`, `health`, `model download`, `model verify`).
- Tooling binary: `pockettts-tools` (`model export`).
- HTTP server endpoints: `GET /health`, `GET /voices`, `POST /tts`.
- Experimental browser app via Go WASM kernel (GitHub Pages deployment).

## Get Started

This path gets you from zero to a first `hello.wav` using the native safetensors backend (default).
No Python, no ONNX Runtime required.

1. Build the runtime CLI:

```bash
go build -o pockettts ./cmd/pockettts
```

2. Download model files (ungated repo, no HF token required):

```bash
./pockettts model download \
  --hf-repo kyutai/pocket-tts-without-voice-cloning \
  --out-dir models
```

3. Run local checks:

```bash
./pockettts doctor
```

4. Generate your first WAV:

```bash
./pockettts synth --text "Hello world" --out hello.wav
```

5. Confirm the file exists:

```bash
ls -lh hello.wav
```

## Requirements

### Runtime (no Python required)

- Go `1.25+`
- **native-safetensors** (default): no external dependencies
- **native-onnx**: ONNX Runtime shared library — see [docs/INSTALL.md](docs/INSTALL.md)

### Backend comparison

|                | native-safetensors (default)                  | native-onnx                                    |
| -------------- | --------------------------------------------- | ---------------------------------------------- |
| Required files | `tts_b6369a24.safetensors`, `tokenizer.model` | ONNX models + `manifest.json`                  |
| External deps  | None                                          | ONNX Runtime shared library                    |
| Setup          | `pockettts model download`                    | `pockettts model download` + `model export`    |
| Verify         | `pockettts model verify`                      | `pockettts model verify --backend native-onnx` |
| Performance    | Optimized Go (AVX2/FMA)                       | ONNX Runtime                                   |

### Tooling (Python required)

- `pockettts-tools model export`
  - Python environment with `pocket_tts`, `torch`, `onnx`
  - optional for `--int8`: Python `onnxruntime`

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

- native runtime preflight checks
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

Force CLI compatibility backend:

```bash
./pockettts synth --backend cli --text "Hello from PocketTTS" --out out.wav
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
./pockettts-tools model export --models-dir models --out-dir models/onnx
```

Optional INT8 quantization:

```bash
./pockettts-tools model export --models-dir models --out-dir models/onnx --int8
```

### Download prebuilt ONNX bundle (no Python)

Download and verify a prebuilt ONNX archive directly:

```bash
./pockettts-tools model download-onnx \
  --bundle-url https://example.com/pockettts-onnx-b6369a24.tar.gz \
  --sha256 <sha256> \
  --out-dir models/onnx
```

Or resolve a pinned bundle from lock file (`bundles/onnx-bundles.lock.json`):

```bash
./pockettts-tools model download-onnx --variant b6369a24 --out-dir models/onnx
```

### Verify

Runs a native Go smoke inference for each graph in the exported `manifest.json`
using ONNX Runtime (`onnxruntime-purego`).

```bash
./pockettts model verify --manifest models/onnx/manifest.json
```

If you need to provide a custom ONNX Runtime shared library path:

```bash
export ORT_LIBRARY_PATH=/usr/local/lib/libonnxruntime.so
./pockettts model verify --manifest models/onnx/manifest.json
```

## Export a voice embedding

Exports a `.safetensors` voice embedding from a speaker WAV/PCM prompt using the native ONNX `mimi_encoder` + speaker projection path and prints a suggested `voices/manifest.json` entry.

```bash
./pockettts export-voice --input speaker.wav --out voices/my_voice.safetensors --id my-voice --license "CC-BY-4.0"
```

See [voices/README.md](voices/README.md) for licensing guidance.

## Server

Start the server (HTTP):

```bash
./pockettts serve
```

Health check:

```bash
curl -s http://localhost:8080/health
```

The CLI also has a probe command:

```bash
./pockettts health
./pockettts health --addr localhost:8080
```

## Web WASM App (Experimental)

This repo now includes a GitHub Action that builds a browser app artifact with:

- Go wasm kernel: `web/dist/pockettts-kernel.wasm`
- Go runtime JS shim: `web/dist/wasm_exec.js`
- Static app: `web/dist/index.html`, `web/dist/main.js`
- Native safetensors model: `web/dist/models/tts_b6369a24.safetensors`
- Voice embeddings: `web/dist/voices/*.safetensors` + `web/dist/voices/manifest.json`

Run/deploy workflow:

- GitHub Actions -> `Deploy Web App to GitHub Pages` -> `Run workflow`
- Deployment is handled by `.github/workflows/deploy-pages.yml`.
  - Pushes to `main` build and deploy automatically.
  - Build includes direct safetensors model download (no ONNX export/conversion step).

The deployed page provides a single synthesis path:

- `Go WASM kernel` orchestration (`PocketTTSKernel.loadModel` + `PocketTTSKernel.synthesize`) for model boot, text preprocessing/chunking, autoregressive generation, and WAV encoding.
- Native safetensors inference runs directly in Go/wasm (no `onnxruntime-web` graph bridge).
- Optional voice conditioning by passing `.safetensors` embeddings into the Go kernel.

At startup the app runs capability checks and only enables synthesis when kernel + model are ready.

Current browser constraints:

- Model download/bundling remains a CI/tooling step (not in-browser conversion).

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
  # Backend: native (default) or cli
  backend: "native"
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
- `POCKETTTS_BACKEND` (`native` or `cli`)
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

## Acknowledgements

This project is based heavily on the work of [Kyutai Labs](https://github.com/kyutai-labs) and their [PocketTTS](https://github.com/kyutai-labs/pocket-tts) text-to-speech model. The native Go inference engine reimplements the model architecture and generation loop originally developed by the Kyutai team. All model weights are downloaded from their official Hugging Face repositories. Full credit for the underlying research and model design belongs to them.
