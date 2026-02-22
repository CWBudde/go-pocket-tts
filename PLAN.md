# PLAN

> **Goal:** Go CLI and HTTP server for PocketTTS. Phases 0–15 complete. Active work: Phases 16–19 (native ONNX inference pipeline). Future: Phases 20–23 (voice encoding, safetensors hardening, streaming, KV-cache inference).

## Phases 0–15 ✅ Complete

- **0** Repo scaffolding, Cobra CLI, config, CI skeleton, `doctor` command
- **1** Adopted `go-call-pocket-tts` for CLI subprocess backend
- **2** Voice manifest (`voices/manifest.json`), `VoiceManager`, `export-voice` helper
- **3** `model download/export/verify`, ORT bootstrap (`sync.Once`), `SessionManager`, tensor helpers
- **4** Text normalization and sentence-level chunking
- **5** WAV I/O (`cwbudde/wav`), DSP chain (normalize, DC-block, fade)
- **6** `synth` end-to-end: text→WAV, chunking, DSP, stdout support, `native|cli` backend flag
- **7** HTTP server (`/health`, `/voices`, `/tts`), worker pool, graceful shutdown, request limits
- **8** Structured logging (`log/slog`), `doctor` backend-awareness (native skips Python checks)
- **9** `bench` command with RTF calculation and CI gate flag
- **10** Packaging/release: INSTALL docs, release artifacts, operational README _(partially done)_
- **11** Runtime de-Pythonization: `native` mode needs no Python; `cli` mode kept for compatibility
- **12** Browser/WASM kernel (`cmd/pockettts-wasm`), static web app, ONNX bundle CI pipeline
- **13** Go-first WASM orchestration, JS narrowed to host bridge, `download-onnx` bundle command
- **14** Integration test suite (`integration` build tag): `synth`, `serve`, `doctor`, `model verify`, CI job
- **15** Fixed `doctor` voice path resolution (was CWD-relative; now resolved absolute via `VoiceManager`)

---

## Reference architecture (from Python/Rust implementations)

The ONNX export (`scripts/export_onnx.py`) produces **6 graphs** (not 5 — includes `mimi_encoder` for voice audio encoding):

| Graph              | Inputs                                                                   | Outputs                                              | Notes                                                         |
| ------------------ | ------------------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------- |
| `text_conditioner` | `tokens [1, T] int64`                                                    | `text_embeddings [1, T, 1024] f32`                   | SentencePiece token IDs (vocab=4000)                          |
| `flow_lm_main`     | `sequence [1, T_seq, 32] f32`, `text_embeddings [1, T_text, 1024] f32`   | `last_hidden [1, 1024] f32`, `eos_logits [1, 1] f32` | Stateless (no KV cache); re-processes full sequence each call |
| `flow_lm_flow`     | `condition [1, 1024] f32`, `s [1,1] f32`, `t [1,1] f32`, `x [1, 32] f32` | `flow_direction [1, 32] f32`                         | SimpleMLPAdaLN with AdaLN modulation                          |
| `latent_to_mimi`   | `latent [1, T, 32] f32`                                                  | `mimi_latent [1, 512, T] f32`                        | Denormalize (×std+mean) + Conv1d quantizer projection         |
| `mimi_decoder`     | `latent [1, 512, T] f32`                                                 | `audio [1, 1, N_samples] f32`                        | Stateless; upsample → transformer → SEANet decoder            |
| `mimi_encoder`     | `audio [1, 1, N_samples] f32`                                            | `latent [1, T, 512] f32`                             | For voice audio encoding (Phase 21)                           |

**Key constants** (variant `b6369a24`):

- `ldim = 32` (latent dim, from `mimi.quantizer.dimension`)
- `d_model = 1024`, `num_heads = 16`, `num_layers = 6` (transformer backbone)
- `flow_dim = 512`, `flow_depth = 6` (flow network)
- `sample_rate = 24000`, `frame_rate = 12.5` (≈1920 samples/frame)
- `temperature = 0.7`, `eos_threshold = -4.0` (on raw logits, not sigmoid)
- `lsd_decode_steps = 1` (single Euler step by default)
- `n_bins = 4000` (SentencePiece vocabulary size)

**Generation loop** (per text chunk):

1. Tokenize text via SentencePiece → `tokens [1, T]`
2. Run `text_conditioner(tokens)` → `text_embeddings [1, T, 1024]`
3. If voice conditioning: `concat([voice_emb [1, T_v, 1024], text_emb], dim=1)`
4. Init BOS: `sequence = NaN [1, 1, 32]` (graph replaces NaN with learned `bos_emb`)
5. Loop until EOS or max_steps:
   a. Run `flow_lm_main(sequence, text_embeddings)` → `last_hidden`, `eos_logits`
   b. Check `eos_logits[0] > -4.0` → set EOS countdown (1–3 extra frames)
   c. Sample noise `x ~ N(0, sqrt(temperature))` shape `[1, 32]`
   d. Run `flow_lm_flow(last_hidden, s=0, t=1, x)` → `flow_dir`; `x += flow_dir / steps`
   e. Reshape result to `[1, 1, 32]`, append to sequence
6. Stack all latent frames → `[1, T_frames, 32]`
7. Run `latent_to_mimi` → `[1, 512, T_frames]`
8. Run `mimi_decoder` → `[1, 1, N_samples]` at 24 kHz

**Text preprocessing** (matching reference):

- Capitalize first letter, add period if missing punctuation
- Pad with 8 leading spaces if < 5 words
- Split long text at sentence boundaries into ≤50-token chunks
- `max_frames = ceil(num_tokens/3 + 2) × 12.5`
- `frames_after_eos`: 3 if ≤4 words, else 1

**Voice conditioning** (.safetensors pre-encoded):

- Load first tensor from `.safetensors` file → `[T_voice, 1024]` or `[1, T_voice, 1024]`
- Reshape to `[1, T_voice, 1024]` and prepend to `text_embeddings`

---

## Phase 16 — ORT Session Execution Foundation

> **Goal:** Wire `onnxruntime-purego` (already in go.mod, used by `model/verify.go`) into a reusable `Runner` type. Replace the sine-wave stub `Engine` with manifest-based runner management. By the end, `Engine` can load a manifest, create ORT sessions, and run any ONNX graph with correct tensor I/O.

- [x] Task 16.1: **Create `Runner` type for ONNX graph execution**
  - [x] `internal/onnx/runner.go`: `Runner` wraps `ort.Runtime` + `ort.Env` + `ort.Session` per graph
  - [x] `Run(ctx, inputs map[string]*Tensor) (map[string]*Tensor, error)` — converts to/from ORT tensors
  - [x] Handle float32/int64 dtype dispatch via `ort.NewTensorValue` and `ort.GetTensorData`
  - [x] `Close()` for session cleanup (idempotent)
  - [x] Tests: round-trip with `identity_float32.onnx` from `model/testdata/`

- [x] Task 16.2: **Add `cfg.Paths.ONNXManifest` config field**
  - [x] Add `ONNXManifest string` to `PathsConfig`, default `"models/onnx/manifest.json"`
  - [x] Register flag `--paths-onnx-manifest`, env `POCKETTTS_ONNX_MANIFEST`
  - [x] Wire defaults, aliases

- [x] Task 16.3: **Refactor `Engine` to use `SessionManager` + `Runner`**
  - [x] `NewEngine(manifestPath string, cfg RunnerConfig)` — loads manifest, creates `Runner` per graph
  - [x] Remove sine-wave stub and `threads` field
  - [x] `Engine.Runner(name) (*Runner, bool)` accessor
  - [x] `Engine.Close()` to tear down all runners
  - [x] Temporary `Infer()` shim returns error (real pipeline in Phase 18)

- [x] Task 16.4: **Update `tts.Service` and CLI callers**
  - [x] `NewService(cfg)` uses `DetectRuntime()` + `NewEngine(cfg.Paths.ONNXManifest, rcfg)`
  - [x] Add `Service.Close()` to clean up engine
  - [x] Verify `synth` and `serve` commands still compile and work

- [x] Task 16.5: **Full test sweep**
  - [x] `go test ./...` — all 13 packages pass
  - [x] `golangci-lint run ./...` — no new warnings (1 pre-existing in doctor_test.go)
  - [x] Integration tests with identity ONNX model via `Runner`

---

## Phase 17 — SentencePiece Tokenizer + Text Conditioning

> **Goal:** Replace the unicode code-point preprocessor with SentencePiece tokenization (pure Go, no CGO), and run the `text_conditioner` ONNX graph to produce text embeddings.

- [x] Task 17.1: **Add SentencePiece binding**
  - [x] Evaluated libraries: model is UNIGRAM; chose `github.com/vikesh-raj/go-sentencepiece-encoder` (pure Go, no CGO, UNIGRAM+BPE, matches Python reference exactly — Phase 20 tokenizer item now N/A)
  - [x] `internal/tokenizer/tokenizer.go` with `Tokenizer` interface: `Encode(text string) ([]int64, error)`
  - [x] `SentencePieceTokenizer` loads `tokenizer.model`; 10 tests, 100% coverage

- [x] Task 17.2: **Add config for tokenizer model path**
  - [x] `cfg.Paths.TokenizerModel string`, default `"models/tokenizer.model"`
  - [x] Flag `--paths-tokenizer-model`, env `POCKETTTS_PATHS_TOKENIZER_MODEL`; alias, Viper default, `registerAliases` wired
  - [x] `tts.Service` holds `tokenizer.Tokenizer`; `NewService` initialises it from `cfg.Paths.TokenizerModel`

- [x] Task 17.3: **Text preprocessing matching reference implementation**
  - [x] Capitalize first letter of input
  - [x] Add trailing period if last character is alphanumeric
  - [x] Pad with 8 leading spaces if < 5 words
  - [x] Normalize whitespace (newlines → spaces, collapse doubles)
  - [x] Calculate `max_frames = ceil(num_tokens/3 + 2) × 12.5` (on `ChunkMetadata`)
  - [x] Calculate `frames_after_eos`: 3 if ≤4 words, else 1 (on `ChunkMetadata`)
  - [x] `text.PrepareChunks(input, tok, maxTokens)` tokenizes first, then splits at sentence boundaries respecting the 50-token budget; returns `[]ChunkMetadata`

- [x] Task 17.4: **Run `text_conditioner` ONNX graph**
  - [x] `Engine.TextConditioner(ctx, tokens []int64) (*Tensor, error)` — builds `[1, T] int64` tensor, runs graph, returns `text_embeddings [1, T, 1024] float32`
  - [x] `Engine.runners` field generalised to `runnerIface` to allow unit testing without ORT
  - [x] `Service.Synthesize` uses `PrepareChunks` + `TextConditioner` per chunk; Phase 18 stub follows

- [x] Task 17.5: **Tests**
  - [x] Unit tests: SentencePiece tokenizer produces correct token IDs (done in 17.1, 10 tests, 100% coverage)
  - [x] Unit tests: `PrepareText` (capitalization, padding, punctuation, whitespace) + `ChunkMetadata` (MaxFrames, FramesAfterEOS) + `PrepareChunks` (splitting, metadata, error paths) — `internal/text/prepare_test.go`
  - [x] Unit tests: `Engine.TextConditioner` error paths (missing graph, empty tokens, runner error, correct output) — `internal/onnx/text_conditioner_test.go`
  - [x] Integration test (`integration` tag): tokenize real text + run `text_conditioner`, verify output shape `[1, T, 1024]` — `internal/onnx/text_conditioner_integration_test.go`

---

## Phase 18 — Autoregressive Generation Loop

> **Goal:** Implement the full generation pipeline: `flow_lm_main` autoregressive loop, `flow_lm_flow` Euler ODE integration, `latent_to_mimi` + `mimi_decoder` audio decoding. `Engine.Infer()` produces real 24 kHz speech audio.

- [x] Task 18.1: **Implement `flow_lm_main` autoregressive loop**
  - [x] Initialize BOS: `NewBOSSequence()` → `NaN [1, 1, 32]` (graph handles NaN→bos_emb internally) — `internal/onnx/flow_lm.go`
  - [x] Each step: `Engine.FlowLMStep(sequence, text_embeddings)` → `last_hidden [1, 1024]`, `eos_logits [1, 1]`
  - [x] EOS detection: `EOSDetected(eosLogits, threshold)` — raw logits, strict `>` comparison
  - [x] Sequence growth: `AppendLatentFrame(sequence, frame)` — `[1, S, 32]` + `[1, 1, 32]` → `[1, S+1, 32]`
  - [x] Unit tests: missing graph, runner error, correct I/O keys/shapes, BOS NaN values, sequence growth, EOS threshold logic — `internal/onnx/flow_lm_test.go`
  - [ ] EOS countdown + max_steps loop orchestration (wired in Task 18.4)

- [x] Task 18.2: **Implement Euler flow integration (LSD decode)**
  - [x] `Engine.FlowLMFlow(ctx, lastHidden, temperature, steps)` → latent frame `[1, 1, 32]` — `internal/onnx/flow_lm.go`
  - [x] Sample noise `x ~ N(0, sqrt(temperature))` shape `[1, 32]`; deterministic when temp=0
  - [x] Euler loop: `s = i/steps`, `t = (i+1)/steps`, run `flow_lm_flow(condition, s, t, x)` → `flow_direction [1, 32]`, `x += flow_dir / steps`
  - [x] Unit tests: missing graph, runner error, single-step arithmetic (temp=0), multi-step arithmetic, missing output — `internal/onnx/flow_lm_test.go`

- [x] Task 18.3: **Implement `latent_to_mimi` + `mimi_decoder` pipeline**
  - [x] `StackLatentFrames(frames)` — concatenate `[1, 1, 32]` frames → `[1, T, 32]` — `internal/onnx/audio_decode.go`
  - [x] `Engine.LatentToMimi(ctx, latent)` — input `latent [1, T, 32]` → output `mimi_latent [1, 512, T]`
  - [x] `Engine.MimiDecode(ctx, mimiLatent)` — input `latent [1, 512, T]` → output `[]float32` PCM samples
  - [x] Unit tests: stacking (single, multi, empty), missing graph, runner error, correct I/O, missing output — `internal/onnx/audio_decode_test.go`

- [x] Task 18.4: **Wire complete pipeline into `Engine.GenerateAudio()` + `Service.Synthesize()`**
  - [x] `GenerateConfig` struct with Temperature, EOSThreshold, MaxSteps, LSDDecodeSteps, FramesAfterEOS — `internal/onnx/generate.go`
  - [x] `Engine.GenerateAudio(ctx, tokens, cfg)` — full pipeline: text_conditioner → AR loop (FlowLMStep + EOS countdown + FlowLMFlow) → StackLatentFrames → LatentToMimi → MimiDecode → `[]float32` PCM
  - [x] `Service.Synthesize` rewired: PrepareChunks → GenerateAudio per chunk → concatenate PCM
  - [x] Removed `Engine.Infer` stub (no longer needed)
  - [x] `audio.ExpectedSampleRate` already set to 24000 — no changes needed
  - [x] Unit tests: non-empty PCM, MaxSteps limit, EOS countdown (fires at step 2 + 3 after = 5 total), missing graph, empty tokens, error propagation — `internal/onnx/generate_test.go`

- [x] Task 18.5: **Add generation config fields to `TTSConfig`**
  - [x] `Temperature float64` (default `0.7`), `EOSThreshold float64` (default `-4.0`), `MaxSteps int` (default `256`), `LSDDecodeSteps int` (default `1`) — `internal/config/config.go`
  - [x] CLI flags: `--temperature`, `--eos-threshold`, `--max-steps`, `--lsd-steps` + viper defaults + aliases
  - [x] `Service` stores `TTSConfig`, `generateConfig()` reads from it instead of hardcoded defaults
  - [x] Unit tests: default values, flag registration, flag override loading — `internal/config/config_test.go`

- [x] Task 18.6: **Tests**
  - [x] Unit test: mock `Runner`, verify `Engine.GenerateAudio` calls graphs in correct order with correct tensor shapes and names — already covered in `internal/onnx/generate_test.go` (6 tests), `flow_lm_test.go` (14 tests), `audio_decode_test.go` (11 tests)
  - [x] Unit test: verify EOS detection logic (countdown, threshold) — `flow_lm_test.go` (5 EOSDetected subtests) + `generate_test.go` (TestGenerateAudio_EOSCountdown)
  - [x] Unit test: verify Euler integration arithmetic (single step, multi-step) — `flow_lm_test.go` (TestFlowLMFlow_SingleStep_ReturnsLatentFrame, TestFlowLMFlow_MultiStep_Arithmetic)
  - [x] Integration test (`integration` tag): `Engine.GenerateAudio("Hello.")` against real ONNX models → non-trivial audio, plausible length (0.5–5 s), not silence — `internal/onnx/generate_integration_test.go`
  - [x] CLI regression: `pockettts synth --backend native` → valid 24 kHz WAV — already covered in `cmd/pockettts/synth_cli_integration_test.go` (TestSynthNative_ShortText, TestSynthNative_Chunked)

---

## Phase 19 — Voice Conditioning (.safetensors)

> **Goal:** Load pre-encoded voice embeddings from `.safetensors` files and inject them as conditioning prefix into the generation loop. Supports the predefined Kyutai voice embeddings.

- [x] Task 19.1: **Implement safetensors reader**
  - [x] Create `internal/safetensors/reader.go` — `LoadFirstTensor(path)` and `LoadVoiceEmbedding(path)`
  - [x] Parse safetensors format: 8-byte LE header length → JSON header → raw tensor data
  - [x] Extract first tensor: name, dtype (F32 only), shape, raw bytes → `[]float32`
  - [x] Handle both `[T, 1024]` and `[1, T, 1024]` shapes (reshape 2D → 3D) via `LoadVoiceEmbedding`
  - [x] Unit tests: 14 tests covering 2D/3D/multi-tensor, error paths (empty, truncated, invalid JSON, unsupported dtype, missing file, 1D/4D) — `internal/safetensors/reader_test.go`

- [x] Task 19.2: **Inject voice embeddings into generation loop**
  - [x] Added `VoiceEmbedding *Tensor` field to `GenerateConfig` — optional voice conditioning tensor `[1, T_voice, D]`
  - [x] `ConcatTensorsDim1(a, b)` — concatenates two 3D float32 tensors along dim=1 — `internal/onnx/tensor.go`
  - [x] `GenerateAudio` prepends voice embedding to text_embeddings before AR loop when `VoiceEmbedding` is set
  - [x] Unit tests: ConcatTensorsDim1 (basic, dim mismatch, batch mismatch, not-3D), GenerateAudio with/without voice embedding — `internal/onnx/voice_inject_test.go`

- [ ] Task 19.3: **Wire voice into `tts.Service` and CLI**
  - [ ] `Service.Synthesize()` resolves voice ID → path via `VoiceManager`, passes to `Engine.Infer`
  - [ ] `synth --voice <id>` and `serve` `/tts?voice=<id>` pass voice to native backend
  - [ ] Graceful fallback: if no voice specified, generate without conditioning prefix

- [ ] Task 19.4: **Tests**
  - [ ] Unit test: safetensors reader with synthetic test data (known shape + values)
  - [ ] Unit test: verify voice embedding is correctly prepended to text_embeddings
  - [ ] Integration test (`integration` tag): generate with and without voice, verify outputs differ
  - [ ] CLI: `pockettts synth --backend native --voice alba --text "Hello" --out /tmp/voice.wav`

---

## Phase 20 — Voice Encoding from Audio

> **Goal:** Support encoding raw audio files into voice embeddings via the `mimi_encoder` ONNX graph + a `speaker_proj_weight` linear projection. Enables creating voice embeddings from arbitrary audio without external tooling.

- [ ] Task 20.1: **Run `mimi_encoder` to get audio latents**
  - [ ] Accept raw WAV/PCM input → `audio [1, 1, N_samples] float32` tensor
  - [ ] Run `mimi_encoder` via `Runner` → `latent [1, T, 512] float32`

- [ ] Task 20.2: **Apply speaker projection**
  - [ ] Load `speaker_proj_weight` from model safetensors (shape `[1024, 512]`)
  - [ ] Matrix-multiply latent `[1, T, 512]` × `weight^T [512, 1024]` → `embedding [1, T, 1024]`
  - [ ] Expose as `Engine.EncodeVoice(audioPath string) ([]float32, error)`

- [ ] Task 20.3: **Wire into CLI**
  - [ ] `export-voice` command uses `Engine.EncodeVoice` instead of Python subprocess
  - [ ] Save result as `.safetensors` file for later use with voice conditioning

- [ ] Task 20.4: **Tests**
  - [ ] Unit test: verify projection arithmetic with known weight + latent values
  - [ ] Integration test (`integration` tag): encode a short WAV, verify output shape `[1, T, 1024]`
  - [ ] CLI: `pockettts export-voice --input audio.wav --out voice.safetensors`

---

## Phase 21 — Safetensors Hardening

> **Goal:** Harden the safetensors reader introduced in Phase 19 for production use: support multiple tensors, all relevant dtypes (float32, float16, bfloat16), and large files via memory mapping.

- [ ] Task 21.1: **Extend safetensors reader**
  - [ ] Support reading named tensors by key (not just the first tensor)
  - [ ] Handle float16 and bfloat16 dtypes → convert to float32
  - [ ] Add `ReadAll() (map[string][]float32, error)` for multi-tensor files

- [ ] Task 21.2: **Memory-map large files**
  - [ ] Use `mmap` for files above a configurable threshold (default 64 MiB)
  - [ ] Ensure safe cleanup on `Close()`

- [ ] Task 21.3: **Tests**
  - [ ] Unit test: round-trip write + read for float32, float16 tensors
  - [ ] Unit test: multi-tensor file, verify correct key lookup
  - [ ] Benchmark: compare mmap vs. `os.ReadFile` for a 100 MiB file

---

## Phase 22 — Streaming Audio Generation

> **Goal:** Generate and decode latent frames concurrently using a producer/consumer pipeline, matching the Rust/Python threading model. Enable real-time audio streaming from the HTTP server.

- [ ] Task 22.1: **Implement streaming generation pipeline**
  - [ ] Run `flow_lm_main` + `flow_lm_flow` in a goroutine, emit latent frames to a channel
  - [ ] Run `latent_to_mimi` + `mimi_decoder` in a consumer goroutine, emit PCM chunks
  - [ ] Configurable channel buffer depth to tune latency vs. throughput

- [ ] Task 22.2: **HTTP streaming endpoint**
  - [ ] `/tts/stream` returns `audio/wav` with chunked transfer encoding
  - [ ] Write WAV header upfront (with unknown length), flush PCM chunks as they arrive
  - [ ] Honour `context.Context` cancellation to abort in-progress generation

- [ ] Task 22.3: **Tests**
  - [ ] Unit test: pipeline produces same total PCM as batch mode for the same input
  - [ ] Integration test (`integration` tag): streaming endpoint delivers first chunk within 500 ms
  - [ ] Load test: concurrent streaming requests do not deadlock or leak goroutines

---

## Phase 23 — KV-Cache Autoregressive Inference

> **Goal:** Re-export `flow_lm_main` with stateful KV-cache inputs/outputs. Each autoregressive step processes only the new frame instead of the full growing sequence, reducing inference from O(n²) to O(n).

- [ ] Task 23.1: **Re-export KV-cache ONNX graph**
  - [ ] Extend `scripts/export_onnx.py` to export `flow_lm_main_kv` with past key/value state tensors
  - [ ] Update ONNX manifest schema to declare stateful graphs with `state_inputs`/`state_outputs`
  - [ ] Verify exported graph with `model verify` command

- [ ] Task 23.2: **Implement stateful runner**
  - [ ] `StatefulRunner` wraps `Runner`, carries KV-cache tensors between steps
  - [ ] `Step(lastHidden, textEmbeddings, state) → (lastHidden, eosLogits, newState)`
  - [ ] Reset state on new inference call

- [ ] Task 23.3: **Replace O(n²) loop in `Engine.Infer`**
  - [ ] Detect presence of `flow_lm_main_kv` in manifest; fall back to stateless if absent
  - [ ] Wire `StatefulRunner` into the autoregressive loop from Phase 18

- [ ] Task 23.4: **Tests**
  - [ ] Unit test: stateful runner produces identical logits to stateless runner for same sequence
  - [ ] Benchmark (`go test -bench`): measure RTF improvement for 5 s, 15 s, 30 s utterances
  - [ ] Integration test (`integration` tag): end-to-end synthesis with KV-cache produces valid audio
