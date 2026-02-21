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
  - [x] Implemented `pockettts-tools export-voice --audio <file> --out <voice.safetensors>` in `cmd/pockettts-tools/export_voice.go`
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
  - [x] Delegates to Python helper script `scripts/export_onnx.py` via subprocess (`pockettts-tools model export`)
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

- [x] Task 5.2: **Add `github.com/cwbudde/algo-dsp` and wire DSP chain**
  - [x] `go get github.com/cwbudde/algo-dsp`
  - [x] `--normalize`: peak-normalize using library primitive
  - [x] `--dc-block`: remove DC offset using `design.Highpass` biquad filter (20 Hz cutoff)
  - [x] `--fade-in-ms` / `--fade-out-ms`: linear amplitude ramp
  - [x] Apply DSP chain after WAV decode, before WAV encode/write (wired in Phase 6)

- [x] Task 5.3: **Unit tests for audio package**
  - [x] Test WAV decode → validate format with a fixture WAV
  - [x] Test WAV encode roundtrip: decode then re-encode, assert header fields match
  - [x] Test each DSP step with synthetic PCM input

---

## Phase 6 — `synth` CLI command (end-to-end)

- [x] Task 6.1: **Wire up `pockettts synth`**
  - [x] Accept `--text` flag or read from stdin when flag is absent
  - [x] Accept `--voice` flag (resolved via VoiceManager from Phase 2)
  - [x] Accept `--out` flag (`-` for stdout, default `out.wav`)
  - [x] Pass extra `pocket-tts` flags through via `--tts-arg key=value` (forwarded verbatim)

- [x] Task 6.2: **Handle chunked synthesis**
  - [x] If `--chunk` is set, split text via `ChunkBySentence` from Phase 4
  - [x] Run one subprocess call per chunk sequentially
  - [x] Concatenate resulting PCM buffers into a single `AudioBuffer`

- [x] Task 6.3: **Apply DSP chain and write output**
  - [x] Run configured DSP steps (Phase 5) on the merged PCM buffer
  - [x] Write WAV to file or stdout

- [x] Task 6.4: **Unit tests for `synth` command**
  - [x] Test stdin fallback: when `--text` is absent, text is read from a mock reader
  - [x] Test chunk path: multiple PCM buffers are correctly concatenated
  - [x] Test DSP + write pipeline with a mock audio buffer and no subprocess involved

- [x] Task 6.5: **Integration test for `synth`** (build tag `integration`)
  - [x] Skip gracefully if `pocket-tts` binary is not found
  - [x] `synth --text “Hello.” --voice <fixture-voice> --out /tmp/out.wav`
  - [x] Assert output file has valid RIFF header and non-zero duration

- [x] Task 6.6: **Native synthesis backend follow-up (de-Pythonize runtime path)**
  - [x] Add backend selection (`--backend` / `POCKETTTS_BACKEND`: `native|cli`)
  - [x] Implement `native` synthesis path as default (Go + ONNX Runtime, no `pocket-tts` subprocess)
  - [x] Keep `cli` backend as compatibility mode during migration

---

## Phase 7 — `serve` HTTP command

- [x] Task 7.1: **Implement HTTP server (`internal/server/server.go`)**
  - [x] `GET /health` — returns `{“status”:”ok”,”version”:”<build-version>”}`
  - [x] `GET /voices` — returns JSON array from VoiceManager
  - [x] `POST /tts` — JSON body `{text, voice, chunk?}`; synthesizes and returns `audio/wav`

- [x] Task 7.2: **Subprocess worker pool**
  - [x] Configurable concurrency limit via `--workers` (default 2)
  - [x] Semaphore-based throttling; excess requests wait with context cancellation
  - [x] Each worker runs one `pocket-tts generate` subprocess independently

- [x] Task 7.3: **Graceful shutdown**
  - [x] Handle `SIGINT` / `SIGTERM` via `signal.NotifyContext`
  - [x] Stop accepting new connections; drain in-flight requests up to `--shutdown-timeout` (default 30s)

- [x] Task 7.4: **Request validation and limits**
  - [x] Reject empty or oversized text (`--max-text-bytes`, default 4096)
  - [x] Per-request context timeout (`--request-timeout`, default 60s)
  - [x] Return structured JSON errors `{“error”:”...”}` with appropriate HTTP status codes

- [x] Task 7.5: **Unit tests for server package**
  - [x] Test `/health` returns 200 with expected body
  - [x] Test `/tts` with missing or empty body returns 400
  - [x] Test concurrency throttling with a mock subprocess runner interface

- [ ] Task 7.6: **Backend parity follow-up for `serve`**
  - [x] Reuse same backend selection as `synth` (`native|cli`)
  - [x] Ensure `native` mode serves TTS without Python subprocess workers
  - [x] Keep worker-pool behavior for `cli` mode only

---

## Phase 8 — Observability and operational hardening

- [x] Task 8.1: **Structured logging**
  - [x] Use `log/slog` (Go 1.21+) throughout; no third-party logging dependency
  - [x] Log per-request: voice, text length, synthesis duration, exit code
  - [x] Log level configurable via `--log-level` / `POCKETTTS_LOG_LEVEL` (default `info`)

- [ ] Task 8.2: **Metrics (optional, off by default)**
  - [ ] Expose Prometheus metrics on `--metrics-addr` if flag is non-empty
  - [ ] Counters: requests total, errors total; histogram: synthesis duration; gauge: worker queue depth

- [x] Task 8.3: **`doctor` command hardening**
  - [x] Run `pocket-tts --version` and print result; fail if binary not found or exits non-zero
  - [x] Check Python version satisfies `>=3.10,<3.15`
  - [x] Verify each voice file in `manifest.json` exists on disk
  - [x] Run `pockettts model verify` as a sub-check (Phase 3.6)
  - [x] Print colour-coded pass/fail per check; exit non-zero on any failure

- [x] Task 8.4: **Tests for observability and `doctor`**
  - [x] Test `doctor` with a mock environment: all-pass and one-fail scenarios
  - [x] Test log output contains expected fields (voice, duration) using a captured `slog.Handler`
  - [ ] Test metrics counter increments on success and error paths (skipped — metrics Task 8.2 not implemented)

- [x] Task 8.5: **Doctor backend-awareness follow-up**
  - [x] Make `doctor` checks conditional on selected backend
  - [x] In `native` mode, do not require `pocket-tts` binary or Python
  - [x] In `cli` mode, keep `pocket-tts` and Python checks

---

## Phase 9 — Benchmarking

- [x] Task 9.1: **Implement `pockettts bench` command**
  - [x] Flags: `--text`, `--voice`, `--runs` (default 5), `--format json|table`
  - [x] Treat first run as cold-start; remaining runs as warm
  - [x] Compute per-run and aggregate: min, max, mean synthesis duration

- [x] Task 9.2: **Realtime factor (RTF) calculation**
  - [x] Parse output WAV duration from header after each run
  - [x] RTF = synthesis_duration / audio_duration; print alongside latency
  - [x] Optional: `--rtf-threshold` flag — exit non-zero if mean RTF exceeds threshold (CI gate)

- [x] Task 9.3: **Unit tests for `bench` command**
  - [x] Test aggregation logic (min/max/mean) with synthetic timing data
  - [x] Test RTF calculation with known WAV duration and synthesis duration
  - [x] Test `--rtf-threshold` gate: assert exit non-zero when threshold exceeded

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

---

## Phase 11 — Runtime de-Pythonization and dependency cleanup

- [x] Task 11.1: **Classify Python usage by scope**
  - [x] Runtime path: target Python-free (`synth`, `serve`, `doctor` in `native` mode)
  - [x] Tooling path: allow Python (`model export`, optional `export-voice`)
  - [x] Document boundary clearly in README + INSTALL

- [x] Task 11.2: **Isolate tooling commands**
  - [x] Keep `model export` as Python-required helper command
  - [x] Mark `export-voice` as optional tooling command (not runtime requirement)
  - [x] Add explicit error messages when tooling prerequisites are missing

- [x] Task 11.3: **Remove runtime dependency on `go-call-pocket-tts`**
  - [x] Remove usage from `synth` and `doctor` runtime path
  - [x] Keep only for tooling compatibility (if still needed), or remove entirely
  - [x] Add CI job proving runtime commands pass in environment without Python

- [x] Task 11.4: **Packaging profiles**
  - [x] Add `runtime-native` profile (no Python installed)
  - [x] Add `tooling` profile (Python + pocket-tts + export dependencies)
  - [x] Validate both profiles in CI

---

## Phase 12 — Browser/WebAssembly runtime

- [x] Task 12.1: **Build browser kernel from Go**
  - [x] Added `cmd/pockettts-wasm` (`js/wasm` target) exposing `PocketTTSKernel` to JS
  - [x] Reused Go text/audio code for normalize/tokenize/demo WAV generation in wasm
  - [x] Added static web app shell in `web/` consuming the wasm kernel

- [x] Task 12.2: **Web artifact CI workflow**
  - [x] Added `.github/workflows/web-wasm-app.yml`
  - [x] Workflow compiles wasm kernel (`GOOS=js GOARCH=wasm`)
  - [x] Workflow copies Go runtime shim (`wasm_exec.js`) into web artifact
  - [x] Workflow uploads ready-to-serve static artifact

- [x] Task 12.3: **Bundle exported ONNX models for browser smoke checks**
  - [x] Added optional model pipeline in web workflow (`include-models`)
  - [x] Reused `pockettts model download` + `pockettts-tools model export` in CI
  - [x] Bundles `models/onnx/*.onnx` + `manifest.json` into `web/dist/models/`
  - [x] Web app runs manifest-based browser smoke inference via `onnxruntime-web`

- [ ] Task 12.4: **Implement full browser TTS inference loop**
  - [x] Added experimental browser autoregressive loop scaffold in `web/main.js` using `flow_lm_main` + `flow_lm_flow`
  - [x] Added `latent_to_mimi` export graph (`scripts/export_onnx.py`) to match upstream latent denorm + quantizer projection path
  - [x] Integrated `text_conditioner` + decoder graph orchestration in browser runtime
  - [x] Produces WAV waveform from model outputs in-browser (experimental quality)
  - [ ] Replace remaining heuristic/state assumptions with architecture-accurate KV-cache/state handling parity
  - [ ] Add browser voice-conditioning parity (audio prompt / safetensors state path)

- [ ] Task 12.5: **Browser performance and compatibility hardening**
  - [ ] Add WebGPU/WebAssembly execution provider selection strategy + fallback
  - [ ] Add chunked model loading/progress UI and memory guardrails
  - [ ] Add browser integration tests (Playwright) for smoke inference

---

## Phase 13 — Go-first web runtime (de-JS orchestration)

- [x] Task 13.1: **Move browser autoregressive orchestration into Go wasm**
  - [x] Added `PocketTTSKernel.synthesizeModel(...)` in `cmd/pockettts-wasm/main_wasm.go`
  - [x] Moved graph orchestration loop (`text_conditioner` -> `flow_lm_main` -> `flow_lm_flow` -> `latent_to_mimi` -> `mimi_decoder`) into Go wasm
  - [x] Kept JS as thin host bridge (`PocketTTSBridge.runGraph`) for ORT Web session execution
  - [x] Model WAV output now returned from Go wasm (base64)

- [x] Task 13.2: **Narrow JS to pure host/runtime glue**
  - [x] Moved bridge/runtime responsibilities out of UI script into `web/bridge.js`
  - [x] Kept `web/main.js` focused on UI wiring, capability checks, progress, and audio playback
  - [x] Added stricter bridge contract validation in `web/bridge_contract.js`
  - [x] Added bridge contract tests in `web/bridge_contract.test.mjs`

- [x] Task 13.3: **Add Go-native ONNX bundle acquisition command**
  - [x] Implemented `pockettts-tools model download-onnx` for prebuilt ONNX bundles (zip/tar.gz)
  - [x] Added lock-file support for pinned URLs/checksums (`bundles/onnx-bundles.lock.json`)
  - [x] Added checksum verification + manifest/required-graph validation after extraction
  - [x] Integrated optional prebuilt-bundle path into web workflows (`onnx-bundle-url`, `onnx-bundle-sha256`)

- [ ] Task 13.4: **Runtime parity and quality hardening**
  - [ ] Align Go wasm prompt/token behavior with upstream sentencepiece/tokenizer path
  - [ ] Improve EOS/stopping and long-text chunking parity vs upstream generation
  - [ ] Add deterministic regression test vectors for web synth outputs

---

## Phase 14 — Comprehensive integration tests

> **Goal:** Establish a reproducible, CI-gated integration test suite that validates the full synthesis pipeline end-to-end across both backends, covering the HTTP server, CLI commands, and audio output correctness. Integration tests use the `integration` build tag and skip gracefully when required dependencies (`pocket-tts` binary, ONNX runtime, voice files) are unavailable.

- [x] Task 14.1: **Test infrastructure and fixtures**
  - [x] Add `testdata/` directory under `cmd/pockettts/` with a minimal fixture WAV (silence, ~0.1 s at 24 kHz mono) for use as a voice-conditioning prompt
  - [x] Add a shared `internal/testutil` package with helpers: `RequirePocketTTS(t)`, `RequireONNXRuntime(t)`, `RequireVoiceFile(t, id)` — each calls `t.Skip` with a clear reason when the prerequisite is absent
  - [x] Add an integration test matrix in CI (`.github/workflows/test-integration.yml`) using a self-hosted or large runner that has `pocket-tts` and models available; gate the job on the `integration` tag being passed to `go test`

- [ ] Task 14.2: **CLI `synth` integration tests (both backends)**
  - [ ] `TestSynthCLI_ShortText`: synthesize ≤ 50 chars via `--backend cli`, assert RIFF header, non-zero PCM samples, and 24 kHz sample rate
  - [ ] `TestSynthCLI_Chunked`: synthesize multi-sentence text with `--chunk`, assert concatenated output is longer than any single chunk
  - [ ] `TestSynthCLI_DSPChain`: add `--normalize --dc-block --fade-in-ms 10 --fade-out-ms 10`, assert output is still valid WAV with equal sample count
  - [ ] `TestSynthCLI_Stdout`: use `--out -`, capture stdout, assert RIFF bytes
  - [ ] `TestSynthNative_ShortText`: same assertions as `TestSynthCLI_ShortText` for `--backend native`; skip when ONNX runtime or model is absent
  - [ ] `TestSynthNative_Chunked`: chunked synthesis via native backend; assert PCM sample count grows with chunk count

- [ ] Task 14.3: **HTTP server (`serve`) integration tests**
  - [ ] `TestServe_HealthEndpoint`: start server on a random free port with `httptest.NewServer` or a live `net.Listen`, `GET /healthz`, assert `{"status":"ok"}` and 200
  - [ ] `TestServe_VoicesEndpoint`: `GET /voices`, assert JSON array contains at least the fixture voice ID
  - [ ] `TestServe_TTSEndpoint_CLI`: `POST /tts` `{"text":"Hello.","voice":"<fixture>"}` via `cli` backend; assert response `Content-Type: audio/wav` and valid RIFF body
  - [ ] `TestServe_TTSEndpoint_Native`: same for `native` backend; skip when ONNX runtime absent
  - [ ] `TestServe_TTSEndpoint_EmptyText`: assert 400 status and `{"error":...}` body
  - [ ] `TestServe_TTSEndpoint_OversizedText`: send text exceeding `--max-text-bytes`; assert 400
  - [ ] `TestServe_ConcurrentRequests`: fire N concurrent `POST /tts` requests up to the worker limit; assert all succeed and durations are bounded

- [ ] Task 14.4: **`doctor` integration tests**
  - [ ] `TestDoctorPasses_CLI`: run `pockettts doctor` against a valid environment (pocket-tts binary + voices + model files); assert exit 0 and `doctor checks passed` in stdout
  - [ ] `TestDoctorPasses_Native`: same in native mode (no pocket-tts required); assert exit 0
  - [ ] `TestDoctorFails_MissingVoiceFile`: point manifest at a non-existent voice file; assert exit non-zero and failure message in stderr
  - [ ] `TestDoctorFails_BadPocketTTS`: provide a fake `pocket-tts` that exits 1; assert failure surfaced

- [ ] Task 14.5: **`model verify` integration tests**
  - [ ] `TestModelVerify_PassesWithValidONNX`: run `model verify` against the committed tiny identity ONNX from Phase 3.7; assert exit 0
  - [ ] `TestModelVerify_FailsWithMissingManifest`: point `--manifest` at a non-existent path; assert structured error returned
  - [ ] `TestModelVerify_FailsWithCorruptONNX`: write a truncated `.onnx` file; assert exit non-zero with actionable error message

- [ ] Task 14.6: **Audio output correctness assertions**
  - [ ] Extract a shared `assertValidWAV(t, data []byte)` helper that checks: RIFF header, PCM sub-chunk, 24000 Hz sample rate, 16-bit depth, non-zero sample count
  - [ ] Add `assertWAVDurationApprox(t, data []byte, minSec, maxSec float64)` to sanity-check synthesis output is plausibly speech-length
  - [ ] Apply both helpers consistently across Task 14.2 and 14.3 test assertions

- [x] Task 14.7: **CI integration test job**
  - [x] Add `.github/workflows/test-integration.yml` triggered on `workflow_dispatch` and `schedule` (nightly)
  - [x] Job installs `pocket-tts`, downloads model subset, runs `go test -tags integration ./...`
  - [x] Upload test output and any generated WAV artifacts for post-run inspection
  - [ ] Gate merges: add integration job as an optional status check (required on `main` only after models are stable)
