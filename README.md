# go-pocket-tts

Go CLI (and a small HTTP server skeleton) for working with [PocketTTS](https://github.com/kyutai-labs/pocket-tts):

- Run text-to-speech with native Go backend by default (`--backend native`)
- Download PocketTTS model checkpoints from Hugging Face
- Export PocketTTS PyTorch checkpoints to ONNX subgraphs + manifest
- Smoke-verify exported ONNX graphs
- Export voice embeddings (`.safetensors`) from a WAV prompt (optional tooling)

## Status

- CLI commands are implemented.
- Runtime binary: `pockettts` (`synth`, `serve`, `doctor`, `health`, `model download`, `model verify`).
- Tooling binary: `pockettts-tools` (`model export`, `export-voice`).
- HTTP server endpoints: `GET /health`, `GET /voices`, `POST /tts`.

## Get Started (No Python)

This path gets you from zero to a first `hello.wav` using the native backend only.

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

3. (Optional but recommended) point `pockettts` to ONNX Runtime explicitly:

```bash
export POCKETTTS_ORT_LIB=/usr/local/lib/libonnxruntime.so
```

4. Run local checks in native mode:

```bash
./pockettts doctor --backend native
```

5. Generate your first WAV:

```bash
./pockettts synth --backend native --text "Hello world" --out hello.wav
```

6. Confirm the file exists:

```bash
ls -lh hello.wav
```

## Requirements

### Runtime (no Python required)

- Go `1.25+`
- ONNX Runtime shared library for native backend
  - see [docs/INSTALL.md](docs/INSTALL.md)

### Tooling (Python required)

- `pockettts-tools model export`
  - Python environment with `pocket_tts`, `torch`, `onnx`
  - optional for `--int8`: Python `onnxruntime`
- `pockettts-tools export-voice`
  - `pocket-tts` CLI installed (Python package), available in `PATH` or set via `--tts-cli-path`

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

Exports a `.safetensors` voice embedding from a speaker WAV prompt and prints a suggested `voices/manifest.json` entry.
This is an optional tooling command requiring Python `pocket-tts`.

```bash
./pockettts-tools export-voice --audio speaker.wav --out voices/my_voice.safetensors --id my-voice --license "CC-BY-4.0"
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
- Optional ONNX bundle: `web/dist/models/*.onnx` + `web/dist/models/manifest.json`

Run the workflow:

- GitHub Actions -> `Web WASM App` -> `Run workflow`
- Keep `include-models=true` to bundle exported ONNX models into the artifact.
- GitHub Pages deployment is automated via `.github/workflows/deploy-pages.yml` on pushes to `main`.

The uploaded artifact (`pockettts-web-wasm-<run_id>`) can be served as static files.
It provides:

- `Generate Fallback Tone WAV`: wasm-only fallback tone generation (sanity path, not model speech quality).
- `Verify ONNX Models`: browser-side ONNX smoke inference over bundled graphs via `onnxruntime-web`.
- `Synthesize via ONNX (Exp)`: experimental browser autoregressive graph orchestration, now executed in Go wasm (`PocketTTSKernel.synthesizeModel`) via a thin JS ORT bridge.
  - Uses exported `latent_to_mimi` graph when present for upstream-aligned latent denorm + quantizer projection.

At startup the app now runs capability checks and only enables actions that are currently available (kernel, manifest, required model graphs).

Current gap for full browser PocketTTS inference:

- Native ONNX Runtime (`onnxruntime-purego`) is not available in browser wasm.
- Remaining gap: architecture-accurate state/KV-cache parity and voice-conditioning parity with upstream Python runtime.
- Model download/export still happens in CI/tooling (not in-browser).

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
