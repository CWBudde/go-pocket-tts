# PLAN

> **Goal:** Integrate Pocket TTS into Go by spawning the official `pocket-tts generate` CLI as a subprocess, passing text via stdin (`--text -`) and reading WAV audio from stdout (`--output-path -`). This approach maximizes upstream stability and avoids re-implementing the tokenizer, sampling loop, and decoder. Voice states are pre-exported as `.safetensors` via `pocket-tts export-voice`. For repeated/concurrent requests the plan is to optionally front the CLI with `pocket-tts serve` (warm model, streaming HTTP).

## Phase 0 — Scaffolding and repo shape

- [x] Created repository structure:
  - `cmd/pockettts/`
  - `internal/server/`
  - `internal/tts/`
  - `internal/onnx/`
  - `internal/text/`
  - `internal/audio/`
  - `internal/config/`
- [x] Chosen CLI framework: `cobra`
- [x] Added CLI commands:
  - `pockettts synth ...`
  - `pockettts serve ...`
  - `pockettts health ...`
  - `pockettts doctor`
- [x] Added config support with one `Config` struct:
  - flags
  - env vars (`POCKETTTS_*`)
  - optional config file (`yaml`/`toml`/`json` via `--config`)
  - defaults
- [x] Added CI skeleton in `.github/workflows/ci.yml`:
  - `go test ./...`
  - `golangci-lint`
  - build binary for `linux/amd64`
- [x] Added dev smoke command `pockettts doctor`:
  - prints detected ORT library path/version
  - checks model files exist
  - exits non-zero on failure
- [x] Verified project compiles and tests pass locally (`go test ./...`, `go build ./cmd/pockettts`)

---

## Phase 1 — Reuse `../go-call-pocket-tts` instead of reimplementing wrapper

Decision: Phase 1 wrapper work is replaced by adopting `github.com/MeKo-Christian/go-call-pocket-tts` and wiring this repo to it.

- [x] Task 1.1: **Suitability check completed**
  - [x] Python/runtime/install docs already covered upstream (`README.md` in `../go-call-pocket-tts`)
  - [x] Subprocess generate flow already implemented (`pocket-tts generate --text - --output-path -`)
  - [x] Executable path override already supported (`Options.ExecutablePath`)
  - [x] Stdin/stdout piping and stderr capture already implemented
  - [x] Typed failure modes already present (`ErrExecutableNotFound`, `ErrProcessTimeout`, `ErrNonZeroExit`)
  - [x] Integration/golden tests already exist and skip when `pocket-tts` is unavailable

- [x] Task 1.2: **Integrate dependency in this repo**
  - [x] Added module dependency on `github.com/MeKo-Christian/go-call-pocket-tts`
  - [x] Replaced local synth execution path to call the upstream client API
  - [x] Mapped this repo config (`flags/env/file`) into upstream options (`voice`, `config`, `executablePath`, concurrency)
  - [x] Kept `pockettts doctor` as local preflight + model/voice file checks

- [x] Task 1.3: **Acceptance checks after integration**
  - [x] `pockettts synth` produces WAV via upstream library path (`/tmp/task13.wav`, RIFF header validated)
  - [x] Non-zero subprocess exits are surfaced as actionable errors to CLI users
  - [x] CI/unit checks continue to pass without requiring `pocket-tts` binary (`go test ./...`, `go build ./cmd/pockettts`)

---

## Phase 2 — Voice management

- [x] Task 2.1: **Define voice representation**
  - [x] Created `voices/manifest.json` mapping voice IDs to `.safetensors` file paths and license info
  - [x] Defined `Voice` struct in `internal/tts/voice.go` (`ID`, `Path`, `License string`)
  - [x] Documented license restrictions in `voices/README.md` (non-commercial voices must not be used commercially)

- [x] Task 2.2: **Implement `VoiceManager`**
  - [x] Implemented `ListVoices() []Voice` in `internal/tts/voice.go` (manifest-backed)
  - [x] Implemented `ResolvePath(id string) (string, error)` in `internal/tts/voice.go`
  - [x] Added voice file existence validation on resolve (`os.Stat` in `ResolvePath`)

- [x] Task 2.3: **Add `export-voice` helper command**
  - [x] Implemented `pockettts export-voice --audio <file> --out <voice.safetensors>` in `cmd/pockettts/export_voice.go`
  - [x] Delegates to upstream `pocket-tts export-voice` via `go-call-pocket-tts`
  - [x] On success, prints a suggested `manifest.json` entry

- [x] Task 2.4: **Unit tests for VoiceManager**
  - [x] Added fixture-based test for `ListVoices` in `internal/tts/voice_test.go`
  - [x] Added unknown-ID test for `ResolvePath` in `internal/tts/voice_test.go`

---

## Phase 3 — Model acquisition and ONNX bootstrap (`pockettts model` commands)

> Download weights from Hugging Face, convert to ONNX, and wire up the ORT runtime. All steps are exposed as `pockettts model <subcommand>` so the setup is reproducible and scriptable.

- [x] Task 3.1: **`pockettts model download` — fetch weights from Hugging Face**
  - [x] Accept `--hf-repo` (default: `kyutai/pocket-tts`) and `--out-dir` (default: `models/`)
  - [x] Authenticate via `HF_TOKEN` env var or `--hf-token` flag (required for gated repo)
  - [x] Download required checkpoint files from a repo manifest; skip files already present (checksum-based)
  - [x] Print download progress and verify checksums against pinned checksums or a persisted lock manifest (`models/download-manifest.lock.json`)

- [x] Task 3.2: **`pockettts model export` — convert PyTorch weights to ONNX**
  - [x] Delegates to Python helper script `scripts/export_onnx.py` via subprocess (`pockettts model export`)
  - [x] Script exports required sub-graphs: `text_conditioner`, `flow_lm_main`, `flow_lm_flow`, `mimi_encoder`, `mimi_decoder`
  - [x] Supports `--models-dir`, `--out-dir` (default `models/onnx`), optional `--int8` quantization flag
  - [x] Writes `models/onnx/manifest.json` with filenames, input/output names, shapes, and dtypes (inspected from exported ONNX)

- [x] Task 3.3: **ORT bootstrap (`internal/onnx/runtime.go`)**
  - [x] Initialize ORT environment once per process (`sync.Once`) via `onnx.Bootstrap`
  - [x] Load `libonnxruntime` from path configured via `--ort-lib` / `POCKETTTS_ORT_LIB` (with `ORT_LIBRARY_PATH` compatibility)
  - [x] Register graceful shutdown via deferred `onnx.Shutdown()` in `cmd/pockettts/main.go`
  - [x] Document shared-library setup in `docs/INSTALL.md` (system package vs manual download)

- [x] Task 3.4: **`SessionManager` (`internal/onnx/session.go`)**
  - [x] Load all sessions defined in `manifest.json` via `NewSessionManager` and process-wide `LoadSessionsOnce`
  - [x] Thread-safe access via `sync.RWMutex`; reload not supported in MVP (restart required)
  - [x] Log input/output node names per session on load (`slog.Info` per graph)

- [x] Task 3.5: **Tensor utility helpers (`internal/onnx/tensor.go`)**
  - [x] `NewTensor(data []int64 | []float32, shape []int64) (*Tensor, error)`
  - [x] `ExtractFloat32(output) ([]float32, error)` and `ExtractInt64(output) ([]int64, error)`
  - [x] Centralise dtype/shape validation with descriptive errors

- [x] Task 3.6: **`pockettts model verify` — smoke-test loaded sessions**
  - [x] Run one dummy inference pass per session (zero-valued inputs of correct shape)
  - [x] Report pass/fail per session; exit non-zero on any failure
  - [x] Integrate as an additional check in `pockettts doctor`

- [x] Task 3.7: **Unit and integration tests**
  - [x] Unit test tensor helpers with table-driven shape/dtype cases
  - [x] Integration test (build tag `integration`): `model verify` against a committed tiny identity ONNX model

---

## Phase 4 — Text preprocessing and input validation

- [x] Task 4.1: **Implement text normalization (`internal/text/normalize.go`)**
  - [x] Trim leading/trailing whitespace
  - [x] Normalize line endings to `\n`
  - [x] Reject empty string with a typed error

- [x] Task 4.2: **Implement basic chunking (off by default)**
  - [x] `ChunkBySentence(text string, maxChars int) []string` — split on `.`, `!`, `?`
  - [ ] Honour `--chunk` / `--max-chunk-chars` flags (wired in Phase 6)
  - [x] Keep chunk logic simple; each chunk becomes one subprocess call

- [x] Task 4.3: **Unit tests for text package**
  - [x] Table-driven tests for `Normalize`: whitespace, empty, unicode
  - [x] Table-driven tests for `ChunkBySentence`: single sentence, multi-sentence, long text

---

## Phase 5 — WAV handling and audio post-processing

- [x] Task 5.1: **Add `github.com/cwbudde/wav` and wire WAV I/O**
  - [x] `go get github.com/cwbudde/wav`
  - [x] Use library reader to decode WAV bytes received from subprocess stdout
  - [x] Validate decoded format: 24000 Hz, mono, 16-bit PCM; return typed error on mismatch
  - [x] Use library writer to encode PCM and write WAV to file path or stdout (`-`)

- [ ] Task 5.2: **Add `github.com/cwbudde/algo-dsp` and wire DSP chain**
  - [ ] `go get github.com/cwbudde/algo-dsp`
  - [ ] `--normalize`: peak-normalize using library primitive
  - [ ] `--dc-block`: remove DC offset using library primitive
  - [ ] `--fade-in-ms` / `--fade-out-ms`: linear amplitude ramp using library primitive
  - [ ] Apply DSP chain after WAV decode, before WAV encode/write

- [ ] Task 5.3: **Unit tests for audio package**
  - [ ] Test WAV decode → validate format with a fixture WAV
  - [ ] Test WAV encode roundtrip: decode then re-encode, assert header fields match
  - [ ] Test each DSP step with synthetic PCM input

---

## Phase 6 — `synth` CLI command (end-to-end)

- [ ] Task 6.1: **Wire up `pockettts synth`**
  - [ ] Accept `--text` flag or read from stdin when flag is absent
  - [ ] Accept `--voice` flag (resolved via VoiceManager from Phase 2)
  - [ ] Accept `--out` flag (`-` for stdout, default `out.wav`)
  - [ ] Pass extra `pocket-tts` flags through via `--tts-arg key=value` (forwarded verbatim)

- [ ] Task 6.2: **Handle chunked synthesis**
  - [ ] If `--chunk` is set, split text via `ChunkBySentence` from Phase 4
  - [ ] Run one subprocess call per chunk sequentially
  - [ ] Concatenate resulting PCM buffers into a single `AudioBuffer`

- [ ] Task 6.3: **Apply DSP chain and write output**
  - [ ] Run configured DSP steps (Phase 5) on the merged PCM buffer
  - [ ] Write WAV to file or stdout

- [ ] Task 6.4: **Unit tests for `synth` command**
  - [ ] Test stdin fallback: when `--text` is absent, text is read from a mock reader
  - [ ] Test chunk path: multiple PCM buffers are correctly concatenated
  - [ ] Test DSP + write pipeline with a mock audio buffer and no subprocess involved

- [ ] Task 6.5: **Integration test for `synth`** (build tag `integration`)
  - [ ] Skip gracefully if `pocket-tts` binary is not found
  - [ ] `synth --text “Hello.” --voice <fixture-voice> --out /tmp/out.wav`
  - [ ] Assert output file has valid RIFF header and non-zero duration

---

## Phase 7 — `serve` HTTP command

- [ ] Task 7.1: **Implement HTTP server (`internal/server/server.go`)**
  - [ ] `GET /health` — returns `{“status”:”ok”,”version”:”<build-version>”}`
  - [ ] `GET /voices` — returns JSON array from VoiceManager
  - [ ] `POST /tts` — JSON body `{text, voice, chunk?}`; synthesizes and returns `audio/wav`

- [ ] Task 7.2: **Subprocess worker pool**
  - [ ] Configurable concurrency limit via `--workers` (default 2)
  - [ ] Semaphore-based throttling; excess requests wait with context cancellation
  - [ ] Each worker runs one `pocket-tts generate` subprocess independently

- [ ] Task 7.3: **Graceful shutdown**
  - [ ] Handle `SIGINT` / `SIGTERM` via `signal.NotifyContext`
  - [ ] Stop accepting new connections; drain in-flight requests up to `--shutdown-timeout` (default 30s)

- [ ] Task 7.4: **Request validation and limits**
  - [ ] Reject empty or oversized text (`--max-text-bytes`, default 4096)
  - [ ] Per-request context timeout (`--request-timeout`, default 60s)
  - [ ] Return structured JSON errors `{“error”:”...”}` with appropriate HTTP status codes

- [ ] Task 7.5: **Unit tests for server package**
  - [ ] Test `/health` returns 200 with expected body
  - [ ] Test `/tts` with missing or empty body returns 400
  - [ ] Test concurrency throttling with a mock subprocess runner interface

---

## Phase 8 — Observability and operational hardening

- [ ] Task 8.1: **Structured logging**
  - [ ] Use `log/slog` (Go 1.21+) throughout; no third-party logging dependency
  - [ ] Log per-request: voice, text length, synthesis duration, exit code
  - [ ] Log level configurable via `--log-level` / `POCKETTTS_LOG_LEVEL` (default `info`)

- [ ] Task 8.2: **Metrics (optional, off by default)**
  - [ ] Expose Prometheus metrics on `--metrics-addr` if flag is non-empty
  - [ ] Counters: requests total, errors total; histogram: synthesis duration; gauge: worker queue depth

- [ ] Task 8.3: **`doctor` command hardening**
  - [ ] Run `pocket-tts --version` and print result; fail if binary not found or exits non-zero
  - [ ] Check Python version satisfies `>=3.10,<3.15`
  - [ ] Verify each voice file in `manifest.json` exists on disk
  - [x] Run `pockettts model verify` as a sub-check (Phase 3.6)
  - [ ] Print colour-coded pass/fail per check; exit non-zero on any failure

- [ ] Task 8.4: **Tests for observability and `doctor`**
  - [ ] Test `doctor` with a mock environment: all-pass and one-fail scenarios
  - [ ] Test log output contains expected fields (voice, duration) using a captured `slog.Handler`
  - [ ] Test metrics counter increments on success and error paths (using `promtest` or equivalent)

---

## Phase 9 — Benchmarking

- [ ] Task 9.1: **Implement `pockettts bench` command**
  - [ ] Flags: `--text`, `--voice`, `--runs` (default 5), `--format json|table`
  - [ ] Treat first run as cold-start; remaining runs as warm
  - [ ] Compute per-run and aggregate: min, max, mean synthesis duration

- [ ] Task 9.2: **Realtime factor (RTF) calculation**
  - [ ] Parse output WAV duration from header after each run
  - [ ] RTF = synthesis_duration / audio_duration; print alongside latency
  - [ ] Optional: `--rtf-threshold` flag — exit non-zero if mean RTF exceeds threshold (CI gate)

- [ ] Task 9.3: **Unit tests for `bench` command**
  - [ ] Test aggregation logic (min/max/mean) with synthetic timing data
  - [ ] Test RTF calculation with known WAV duration and synthesis duration
  - [ ] Test `--rtf-threshold` gate: assert exit non-zero when threshold exceeded

---

## Phase 10 — Packaging and release

- [ ] Task 10.1: **Document runtime requirements**
  - [ ] Update `docs/INSTALL.md`: Python 3.10–<3.15, `pip install pocket-tts`, HF model access (gated CC-BY-4.0), ORT shared library, voice license summary
  - [ ] Document `pockettts model download` + `model export` as the canonical setup flow
  - [ ] Note supported platforms: Linux/amd64 primary; macOS/arm64 best-effort

- [ ] Task 10.2: **Release artifacts**
  - [ ] Build Go binary for `linux/amd64` and `darwin/arm64` in CI
  - [ ] Include `voices/manifest.json` example, sample `pockettts.yaml` config, and `scripts/export_onnx.py` in release archive

- [ ] Task 10.3: **Operational README**
  - [ ] Quick-start: `pockettts model download` → `model export` → `doctor` → `synth` → `serve`
  - [ ] Example: `pockettts synth --text “Hello world.” --voice en-default --out hello.wav`
  - [ ] Example: `pockettts serve --listen :8080 --workers 4`
  - [ ] Docker example: sidecar pattern with Go binary + Python `pocket-tts` in one image
