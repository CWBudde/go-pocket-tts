# go-pocket-tts

A pure-Go CLI and HTTP server for [PocketTTS](https://github.com/kyutai-labs/pocket-tts) text-to-speech synthesis. The default backend runs inference directly from safetensors weights — no Python, no ONNX Runtime required. An optional ONNX backend is available as a fallback.

**Online demo:** <https://cwbudde.github.io/go-pocket-tts/> runs the same Go inference in
your browser (WebAssembly); it downloads the selected model from Hugging Face.

- Synthesize speech from text with the native Go backend (AVX2/FMA optimized)
- Six embedded upstream model configs: English and German, 6 and 24 layers
- Serve TTS over HTTP with concurrent worker pool and graceful shutdown
- Download PocketTTS model checkpoints from Hugging Face
- Export voice `.safetensors` files from a WAV/PCM prompt
- Optionally export and run ONNX subgraphs via ONNX Runtime
- Run in-browser via experimental WASM kernel ([online demo](https://cwbudde.github.io/go-pocket-tts/))

## Status

All core features are implemented and functional.

- Runtime binary: `pockettts` (`synth`, `export-voice`, `serve`, `doctor`, `health`, `model download`, `model verify`).
- Tooling binary: `pockettts-tools` (`model export`).
- HTTP server endpoints: `GET /health`, `GET /voices`, `POST /tts`.
- Experimental browser app via Go WASM kernel, deployed to GitHub Pages at
  <https://cwbudde.github.io/go-pocket-tts/> with a picker for all six model configs.
- Upstream alignment: the current reintegration pass was checked against
  `kyutai-labs/pocket-tts` commit `2dff8a2d1b3b21bf44ecf0084cc8ce79ab6d6bba`
  and upstream package version `2.1.0`.

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

3. Download the precomputed voices listed in `voices/manifest.json`. This uses
   the Hugging Face CLI (`hf`), installed with `uv tool install huggingface_hub`
   (or `pip install -U huggingface_hub`):

```bash
for voice in alba marius javert jean fantine cosette eponine azelma; do
  hf download kyutai/pocket-tts-without-voice-cloning \
    "embeddings/$voice.safetensors" --local-dir /tmp/pockettts-voices
  cp "/tmp/pockettts-voices/embeddings/$voice.safetensors" voices/
done
```

`just generate` runs steps 1–3 (skipping files that already exist) and then
synthesizes `hello-world.wav`.

4. Run local checks:

```bash
./pockettts doctor
```

5. Generate your first WAV:

```bash
./pockettts synth --text "Hello world" --out hello.wav
```

6. Confirm the file exists:

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
- Python parity fixtures: see [docs/INSTALL.md](docs/INSTALL.md#parity-setup-python-reference)

### Development tools

- `hf` (Hugging Face CLI) for voice downloads and `just generate`: `uv tool install huggingface_hub`
- `prettier` for Markdown/YAML/JSON formatting: `npm i -g prettier` or `brew install prettier`.
  `just fmt` / `just ci` run `treefmt`, which skips Markdown/YAML/JSON checks when it is missing
  (CI does not)
- `gofumpt`, `gci`, `shfmt`, `shellcheck`, `golangci-lint`, `just`, `treefmt`

## Build

```bash
go build -o pockettts ./cmd/pockettts
```

Or run without building a binary:

```bash
go run ./cmd/pockettts --help
```

## Benchmarking

Stage profiling and wasm decode benchmarks are now organized under `bench/`.

- Stage profiler entrypoint: `./bench/stageprof`
- Output directory: `bench/results/` (gitignored)

Recommended commands:

```bash
just bench-stageprof-asm
just bench-stageprof-noavx
just bench-wasm-decode
```

## Quickstart (CLI)

### 1) Download model files

Downloads into `./models` by default.

```bash
./pockettts model download
```

The ungated repo `kyutai/pocket-tts-without-voice-cloning` contains everything
needed to synthesize with the precomputed voices. Only the gated
`kyutai/pocket-tts` repo (weights with the voice-cloning encoder) needs an
`HF_TOKEN` and accepting the model terms on Hugging Face first:

```bash
export HF_TOKEN=...  # or use --hf-token
./pockettts model download --hf-repo kyutai/pocket-tts
```

The ungated checkpoints ship the Mimi encoder zeroed, so `pockettts doctor` reports
`! voice cloning: unavailable` for them: a note, not a failure.

`model download` also fetches the language's default voice next to the voice
manifest the runtime reads (`--paths-voice-manifest`). For English that is
the tracked `voices/manifest.json`, which lists all eight voices, so all of
them are fetched there.
`--voice <id>` (repeatable) picks other predefined voices, `--all-voices` fetches
every one, and `--no-voices` skips them. Other languages download into their own
directories; see [Languages](#languages).

### 2) Sanity-check your setup

Checks, for the selected `--language`:

- native runtime preflight checks
- the voice manifest exists, and the default voice and every listed voice file resolve
- the model exists and has the expected tensors
- the tokenizer loads and its vocab size matches the model config's `n_bins`

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

Without `--voice`, the native backend uses the language's default voice
(`alba` for English, `juergen` for German), like upstream. `native-onnx`
cannot use these voice files and synthesizes without a voice. Override it for a
single request:

```bash
./pockettts synth --text "Hello" --voice alba --out out.wav
```

Write the WAV to stdout:

```bash
./pockettts synth --text "Hello" --out - > out.wav
```

## Languages

`--language` (`tts.language`, `POCKETTTS_TTS_LANGUAGE`) selects one of the
embedded upstream model configs. With the `native-safetensors` backend, one
`serve` process can host several of them; see
[Several languages](#several-languages).

| `--language`             | Layers | Sampler  | Default voice |
| ------------------------ | ------ | -------- | ------------- |
| `english_2026-01`        | 6      | LSD      | `alba`        |
| `english_2026-09`        | 6      | LSD      | `alba`        |
| `english_2026-09_24l`    | 24     | LSD      | `alba`        |
| `english_drifting_26-09` | 6      | drifting | `alba`        |
| `german`                 | 6      | LSD      | `juergen`     |
| `german_24l`             | 24     | LSD      | `juergen`     |

`english_2026-01` is the default and keeps the flat layout from before
per-language models: `models/tts_b6369a24.safetensors`, `models/tokenizer.model`
and `voices/manifest.json`. Every other language comes from `languages/<lang>/`
on Hugging Face, pinned to the revisions in the embedded model config and
checked against pinned SHA256 checksums, and lives in
`models/<lang>/{model.safetensors,tokenizer.json}` and `voices/<lang>/` (voice
files plus `manifest.json`). Each has 27 predefined voices; the English-named
voices in a language's folder are states for that language's model. Unless set
explicitly, `--paths-model-path`, `--paths-tokenizer-model` and
`--paths-voice-manifest` follow the language.

German end to end, with the demo text from upstream `default_parameters.py`:

```bash
./pockettts model download --language german --hf-repo kyutai/pocket-tts-without-voice-cloning
./pockettts doctor --language german
./pockettts synth --language german \
  --text "Hallo Welt. Ich bin Pocket TTS von Kyutai. Ich bin schnell genug, um auch auf kleinen CPUs zu laufen. Ich hoffe, ich gefalle dir." \
  --out hallo.wav
```

`pockettts-tools voice download --language <lang> [--voice <id>…]` fetches
voices on their own (all of the language by default).

## Model export + verify (ONNX)

### Export

```bash
./pockettts-tools model export --models-dir models --out-dir models/onnx
```

The exporter uses the current upstream PocketTTS loader. It exports the model
of the global `--language` (default `english_2026-01`), or a custom upstream
`.yaml` config. `--variant b6369a24` is a hidden, deprecated alias for
`--language english_2026-01`:

```bash
./pockettts-tools model export --language english_2026-01
./pockettts-tools model export --tts-config-path ./pocket_tts_config.yaml
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

Or resolve a pinned bundle from lock file (`bundles/onnx-bundles.lock.json`).
The bundle `variant` defaults to the `--language`; lock entries with the legacy
variant `b6369a24` match `english_2026-01`:

```bash
./pockettts-tools model download-onnx --language english_2026-01 --out-dir models/onnx
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

## Export a voice

By default, exports a legacy `audio_prompt` `.safetensors` file from a speaker
WAV/PCM prompt using the native ONNX `mimi_encoder` + speaker projection path
and prints a suggested `voices/manifest.json` entry.

```bash
./pockettts export-voice --input speaker.wav --out voices/my_voice.safetensors --id my-voice --license "CC-BY-4.0"
```

The WAV prompt may have any sample rate up to 384 kHz and any channel count
(8/16/24/32-bit PCM or 32/64-bit float). Like upstream `export-voice`, it is cut to 30 s, mixed down to
mono, resampled to 24 kHz and ended on a short pause before encoding. Raw PCM
input (any other extension) must be 24 kHz mono 16-bit little-endian.

To produce an upstream-compatible full model-state voice file, use the Python
tooling fallback. Upstream voice `.safetensors` files are serialized prompted
model state: transformer KV-cache tensors plus offsets such as
`<module>/cache` and `<module>/offset`. They are not just raw audio embeddings.

```bash
./pockettts export-voice --format=model-state --input speaker.wav --out voices/my_voice.safetensors --tts-cli-path original/pockettts/.venv/bin/pocket-tts
```

Compatibility summary:

| Format                | Created by                                                   | Native Go backend                          | ONNX backend                |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------ | --------------------------- |
| Upstream model state  | `--format=model-state` or upstream `pocket-tts export-voice` | Accepted directly as prompted FlowLM state | Not supported               |
| Legacy `audio_prompt` | earlier Go tooling or default `--format=legacy-embedding`    | Accepted and re-encoded into native state  | Accepted as voice embedding |

See [voices/README.md](voices/README.md) for format and licensing guidance.

## Server

Start the server (HTTP):

```bash
./pockettts serve
```

Requests without a `voice` use the default voice: the language's built-in voice,
or `--default-voice` (`server.default_voice`, `POCKETTTS_SERVER_DEFAULT_VOICE`).
It takes a voice ID from the manifest, a local `.safetensors` file, an
`https://` URL or a pinned `hf://<org>/<repo>/<path>@<revision>` reference. URL
voices are downloaded once into `<user cache dir>/pockettts/voices/` (for
example `~/Library/Caches/pockettts/voices/` on macOS) without a checksum
check; redirects must stay on `https://`, and `hf://` downloads send `HF_TOKEN`
when it is set. WAV prompts are not accepted yet.

```bash
./pockettts serve --language german \
  --default-voice hf://kyutai/pocket-tts-without-voice-cloning/languages/german/embeddings/anna.safetensors@1e08e6a23401048648a9fdcfde2f89348215c2a7
```

On `native-safetensors`, `serve` loads the default voice at startup and refuses
to start when it cannot (for example before `pockettts model download`) or when
it does not fit the model (for example a `german` voice state with
`german_24l`). The other backends have no default voice and reject
`--default-voice`.

### Several languages

One process can serve several languages on `native-safetensors`. List the extra
ones with `--server-languages` (`server.languages`,
`POCKETTTS_SERVER_LANGUAGES=german,english_2026-01`); requests then pick one
with a `language` field, and `GET /voices?language=<lang>` lists its voices.
Requests without a `language` use the startup language (`--language`):

```bash
./pockettts serve --language german --server-languages english_2026-01
curl -s -X POST http://localhost:8080/tts \
  -d '{"text":"Hello there.","language":"english_2026-01"}' -o hello.wav
```

- The extra languages use their [local layout](#languages) and built-in default
  voice; `--paths-*` and `--default-voice` apply to the startup language only.
  `serve` checks their voice manifest, default voice, model and tokenizer at
  startup and refuses to start when one is missing.
- Each model loads on its first request. At most `--server-max-languages`
  (`server.max_languages`, default 2) stay loaded: the least recently used one
  is unloaded for another language and freed once its running requests finish,
  so memory can briefly exceed the cap. A loaded model keeps only its decoded
  float32 weights (the checkpoint bytes are freed after loading): about
  435–470 MB for the 6-layer models, about 1.34 GB for the `*_24l` models.
- `--workers` limits concurrent synthesis across all languages; a request
  loads its language's model only once it has a worker slot.
- A language that is not served gets 400. With `--backend native-onnx` or `cli`,
  or with `--model-config`, only the startup model is served (`--model-config`:
  requests without a `language`).

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

Try it at <https://cwbudde.github.io/go-pocket-tts/>. This repo includes a GitHub Action that builds a browser app artifact with:

- Go wasm kernel: `web/dist/pockettts-kernel.wasm`
- Go runtime JS shim: `web/dist/wasm_exec.js`
- Static app: `web/dist/index.html`, `web/dist/main.js`
- Language catalog: `web/dist/languages.json`

The page does not bundle models or voices. It fetches the selected config's
model, `tokenizer.json` and voices straight from Hugging Face
(`kyutai/pocket-tts-without-voice-cloning`) at the revisions the CLI downloads,
which `web/languages.json` pins. Regenerate it after a model config or pin
changes:

```bash
go generate ./internal/webmanifest/
```

(`TestCatalogFileUpToDate` fails while it is stale.)

The model picker offers all six embedded configs: `english_2026-01` (the
default), `english_2026-09`, `english_2026-09_24l`, `english_drifting_26-09`,
`german` and `german_24l`. A switch selects the config's default voice,
temperature and, unless you typed your own text, its demo text. The previous
model is dropped from WASM memory first; switching back downloads it again,
which the browser may serve from its HTTP cache. Each 6-layer model is a
~220 MB download, `german_24l` ~670 MB and `english_2026-09_24l` 1.3 GB. The
kernel frees the checkpoint bytes once the weights are decoded, so the 24-layer
models fit in 32-bit WASM memory (4 GB).

Run/deploy workflow:

- GitHub Actions -> `Deploy Web App to GitHub Pages` -> `Run workflow`
- Deployment is handled by `.github/workflows/deploy-pages.yml`.
  - Pushes to `main` build and deploy automatically.

To try it locally, build the same files and serve them with any static file
server:

```bash
mkdir -p web/dist
GOOS=js GOARCH=wasm go build -o web/dist/pockettts-kernel.wasm ./cmd/pockettts-wasm
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" web/dist/
cp web/index.html web/main.js web/languages.json web/dist/
python3 -m http.server -d web/dist 8080
```

The deployed page provides a single synthesis path:

- `Go WASM kernel` orchestration (`PocketTTSKernel.loadModel` + `PocketTTSKernel.synthesize`) for model boot, text preprocessing/chunking, autoregressive generation, and WAV encoding.
  `loadModel(model, tokenizer, progress, {config})` builds the engine from the named embedded model config
  (default `english_2026-01`); `unloadModel()` drops the loaded model.
- Native safetensors inference runs directly in Go/wasm (no `onnxruntime-web` graph bridge).
- Optional voice conditioning by passing `.safetensors` voice files into the Go kernel.

At startup the app runs capability checks and only enables synthesis when kernel + model are ready. If the kernel
stops (for example out of memory), the page says so and asks for a reload.

## Configuration

Configuration is loaded in this order:

1. Flags (see `./pockettts --help`)
2. Environment variables with prefix `POCKETTTS_`
3. Config file passed with `--config`
4. Optional local config file named `pockettts.(yaml|yml|toml|json)` in the working directory

`--config` always points to this Go port's config file. To use an upstream
Python PocketTTS `.yaml` config with `--backend cli` or `export-voice
--format=model-state`, pass it via `--tts-cli-config-path`. For native Go
inference, point `--paths-model-path` at a local model `.safetensors` checkpoint
and `--paths-tokenizer-model` at the matching tokenizer: a `tokenizer.json`
(Hugging Face tokenizers) or a SentencePiece `tokenizer.model`, picked by the
file extension; both give the same token ids for the shipped models on ordinary
text. They differ only on input that literally spells a special token (`<s>`,
`</s>`, `<unk>`, `<pad>`) or a byte piece (`<0x41>`): `tokenizer.json` matches
those as that token, SentencePiece encodes the characters.

`--language` selects one of the embedded upstream model configs and the paths
that follow from it; see [Languages](#languages). `--model-config <file>` loads a
custom upstream model config instead; like upstream's `--config` it cannot be combined with
`--language`, and it needs explicit `--paths-model-path` and
`--paths-tokenizer-model`. `synth`, `bench`, `doctor` and `serve` read voice IDs
from the voice manifest (`--paths-voice-manifest`). Unless `--temperature` is
set, generation uses the model config's `default_temperature` (0.3 for every
shipped config).

`pockettts synth --text -` explicitly reads text from stdin. Omitting `--text`
continues to read stdin as well.

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
  # Voice for requests without one (ID, .safetensors path, https:// or hf:// URL)
  # default_voice: "juergen"
  # Further languages requests may pick, and how many models stay loaded
  # languages: ["english_2026-01"]
  # max_languages: 2

tts:
  # Backend: native-safetensors (default), native-onnx, or cli
  backend: "native"
  # Voice name/ID (or a .safetensors path, depending on your PocketTTS setup)
  voice: "alba"
  # Path to the pocket-tts executable (leave empty to use PATH)
  cli_path: ""
  # Optional PocketTTS config path
  cli_config_path: ""
  concurrency: 1
  quiet: true
```

### Environment variables

Every config key can be set as `POCKETTTS_<SECTION>_<KEY>`, e.g.
`POCKETTTS_TTS_MAX_STEPS` for `tts.max_steps`. The flag-style name
`POCKETTTS_<FLAG>` (e.g. `POCKETTTS_MAX_STEPS` for `--max-steps`) is accepted
too; when both are set, the section-style name wins.

Useful ones:

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
