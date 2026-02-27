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
| `mimi_encoder`     | `audio [1, 1, N_samples] f32`                                            | `latent [1, T, 512] f32`                             | For voice audio encoding (Phase 20)                           |

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

- [x] Task 19.3: **Wire voice into `tts.Service` and CLI**
  - [x] `Service.Synthesize(input, voicePath)` loads `.safetensors` and passes `VoiceEmbedding` to `GenerateConfig` — `internal/tts/service.go`
  - [x] `synth --voice <id>` passes resolved voice path to `synthesizeNative()` → `Service.Synthesize` — `cmd/pockettts/synth.go`
  - [x] `serve` `/tts?voice=<id>` passes voice through `nativeSynthesizer.Synthesize` — `internal/server/server.go`
  - [x] Graceful fallback: empty voicePath = no voice conditioning (no safetensors loaded)

- [x] Task 19.4: **Tests**
  - [x] Unit test: safetensors reader with synthetic test data (known shape + values) — `TestLoadVoiceEmbedding_DataValuesPreserved`, `TestLoadFirstTensor_MetadataKeyIgnored` in `internal/safetensors/reader_test.go`
  - [x] Unit test: verify voice embedding is correctly prepended to text_embeddings — `TestGenerateAudio_WithVoiceEmbedding_PrependsToTextEmb` (19.2, `voice_inject_test.go`) + `TestSynthesize_BadSafetensorsPath_ReturnsError`, `TestSynthesize_InvalidSafetensorsFile_ReturnsError`, `TestSynthesize_EmptyVoicePath_SkipsEmbeddingLoad` in `internal/tts/service_test.go`
  - [x] Integration test (`integration` tag): generate with and without voice, verify outputs differ — `TestGenerateAudioIntegration_VoiceConditioningDiffersFromUnvoiced` in `internal/onnx/generate_integration_test.go`
  - [x] CLI: `pockettts synth --backend native --voice alba --text "Hello, this is a voice test." --out /tmp/voice_alba.wav` — produces 20 s WAV ✓
    - Fixed: `resolveVoiceForNative` added to `cmd/pockettts/synth.go` to resolve voice IDs to safetensors paths for native backend
    - Fixed: `scripts/export_onnx.py` gained `--max-seq` flag (default 256, used 512 for voice conditioning) to avoid RoPE Expand failures
    - Added: `voices/alba.safetensors` downloaded from `kyutai/pocket-tts-without-voice-cloning/embeddings/alba.safetensors` (shape `[1, 125, 1024]`)

---

## Phase 20 — Voice Encoding from Audio

> **Goal:** Support encoding raw audio files into voice embeddings via the `mimi_encoder` ONNX graph + a `speaker_proj_weight` linear projection. Enables creating voice embeddings from arbitrary audio without external tooling.

- [x] Task 20.1: **Run `mimi_encoder` to get audio latents**
  - [x] Accept raw WAV/PCM input → `audio [1, 1, N_samples] float32` tensor
  - [x] Run `mimi_encoder` via `Runner` → `latent [1, T, 512] float32`

- [x] Task 20.2: **Apply speaker projection**
  - [x] Load `speaker_proj_weight` from model safetensors (shape `[1024, 512]`)
  - [x] Matrix-multiply latent `[1, T, 512]` × `weight^T [512, 1024]` → `embedding [1, T, 1024]`
  - [x] Expose as `Engine.EncodeVoice(audioPath string) ([]float32, error)`

- [x] Task 20.3: **Wire into CLI**
  - [x] `export-voice` command uses `Engine.EncodeVoice` instead of Python subprocess
  - [x] Save result as `.safetensors` file for later use with voice conditioning

- [x] Task 20.4: **Tests**
  - [x] Unit test: verify projection arithmetic with known weight + latent values
  - [x] Integration test (`integration` tag): encode a short WAV, verify output shape `[1, T, 1024]`
  - [x] CLI: `pockettts export-voice --input audio.wav --out voice.safetensors`

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

---

## Phase 24 — XN-like Runtime Design + Reuse Plan

> **Goal:** Define the non-ONNX runtime architecture and lock interfaces so implementation can proceed incrementally without breaking CLI/server behavior.

- [x] Task 24.1: **Define runtime target and backend contract**
  - [x] Added explicit backend target `native-safetensors` alongside `native` (ONNX) + `cli` (`internal/config/backend.go`, `internal/config/config.go`, `cmd/pockettts/synth.go`)
  - [x] Added `tts.Runtime` interface boundary so ONNX and safetensors-native runtimes can coexist (`internal/tts/runtime.go`, `internal/tts/service.go`)
  - [x] Kept CLI/server flags and request schema stable while backend internals changed (existing `native` path unchanged; `native-safetensors` returns explicit "not implemented yet")

- [x] Task 24.2: **Map reusable components from completed phases**
  - [x] Reused tokenizer + text chunking from Phase 17 in runtime-neutral service pipeline (`internal/tts/service.go`, validated by `TestSynthesize_UsesSentenceChunkingPipeline` in `internal/tts/service_test.go`)
  - [x] Reused generation controls from Phase 18 via `RuntimeGenerateConfig` pass-through (`internal/tts/runtime.go`, `internal/tts/runtime_onnx.go`, validated by `TestSynthesize_ReusesGenerationConfig`)
  - [x] Reused voice embedding ingestion from Phase 19 through runtime-neutral `VoiceEmbedding` (`internal/tts/service.go`, `internal/safetensors/reader.go`, validated by `TestSynthesize_ReusesVoiceEmbeddingIngestion`)

- [x] Task 24.3: **Parity harness**
  - [x] Added parity harness scaffold that runs the same text/voice/seed case across configured backends and reports per-backend status (`ok`/`skipped`/`error`) — `internal/tts/parity.go`
  - [x] Added JSON snapshot persistence for reference metrics (`token_count`, `chunk_count`, `sample_count`, `peak_abs`, `rms`, `pcm_hash_sha256`) — `internal/tts/parity.go`
  - [x] Current safetensors-native backend is explicitly marked `skipped` until runtime implementation exists; ONNX path remains active and test-covered — `internal/tts/parity_test.go`

---

## Phase 25 — Safetensors Model Loader (VarBuilder-style)

> **Goal:** Load full model weights directly from `.safetensors` into named tensors, similar to `xn` var-builder behavior.

- [ ] Task 25.1: **Implement named tensor store**
  - [x] Extended safetensors handling beyond "first tensor" to keyed lookup via `Store` (`OpenStore`, `Tensor`, `ReadAll`) — `internal/safetensors/store.go`
  - [x] Added dtype conversion support for `f32`, `f16`, `bf16` to float32 decode path — `internal/safetensors/store.go`
  - [ ] Lazy decode is implemented; mmap-backed loading is deferred to Phase 21 hardening

- [x] Task 25.2: **Key remapping layer**
  - [x] Added pluggable key mapper (`KeyMapper`) for remap rules from checkpoint keys to runtime names — `internal/safetensors/store.go`
  - [x] Added strict/lenient remap modes with collision/rejection handling — `internal/safetensors/store.go`

- [x] Task 25.3: **Tests**
  - [x] Added unit tests for keyed lookup, remap modes, dtype conversion, shape validation, missing tensor diagnostics — `internal/safetensors/store_test.go`
  - [x] Added corruption tests for invalid offsets and unsupported dtype handling (plus existing truncated-header coverage) — `internal/safetensors/store_test.go`, `internal/safetensors/reader_test.go`

---

## Phase 26 — Tensor/Ops Runtime Foundation

> **Goal:** Implement the minimal tensor operations needed to execute PocketTTS modules directly in Go.

- [x] Task 26.1: **Core tensor ops**
  - [x] Added deterministic CPU tensor primitives (`reshape`, `transpose`, `concat`, `narrow`, `gather`, broadcast `add`/`mul`, `layer_norm`, `softmax`) in `internal/runtime/tensor/tensor.go`
  - [x] Added deterministic CPU `matmul` + `linear` projection path in `internal/runtime/tensor/tensor.go`
  - [x] Added unit coverage for primitives and numeric behavior in `internal/runtime/tensor/tensor_test.go`

- [x] Task 26.2: **Model-critical kernels**
  - [x] Added kernels for FlowLM + Mimi building blocks: `attention`, `MLP`, `Conv1d`, `ConvTranspose1d`, `causal masking`, `RoPE` in `internal/runtime/ops/ops.go`
  - [x] Added per-kernel ONNX parity tolerance targets (`abs`/`rel`) in `internal/runtime/ops/tolerance.go`
  - [x] Added kernel-level tests in `internal/runtime/ops/ops_test.go`

- [x] Task 26.3: **Performance envelope**
  - [x] Added benchmarks for core kernels and decode-like throughput (`BenchmarkMatMulFlowLM`, `BenchmarkLayerNormFlowLM`, `BenchmarkAttentionFlowLM`, `BenchmarkConv1DMimi`, `BenchmarkFrameDecodeThroughput`) in `internal/runtime/ops/ops_bench_test.go`
  - [x] Documented optional acceleration points (SIMD/assembly/fusion/parallelization) while preserving correctness contract in `internal/runtime/ops/PERFORMANCE.md`

---

## Phase 27 — Port PocketTTS Modules from Weights

> **Goal:** Rebuild model modules in Go using safetensors-loaded weights (no ONNX graphs).

- [x] Task 27.1: **FlowLM stack**
  - [x] Implemented safetensors-native conditioner/token embedding path (`flow_lm.conditioner.embed.weight`) via `LUTConditioner` — `internal/native/conditioner.go`
  - [x] Implemented FlowLM transformer + EOS head + BOS handling (`NaN` -> `bos_emb`) in native module path — `internal/native/flow_lm.go`, `internal/native/flow_transformer.go`
  - [x] Implemented flow network (`flow_lm_flow` equivalent) including LSD decode loop and timestep embedding/res-block/final-layer structure — `internal/native/flow_net.go`, `internal/native/flow_lm.go`

- [x] Task 27.2: **Mimi stack**
  - [x] Implemented latent projection (`latent_to_mimi` equivalent): denorm with `emb_std`/`emb_mean` + quantizer projection (`mimi.quantizer.output_proj`) — `internal/native/model.go`, `internal/native/mimi.go`
  - [x] Implemented native Mimi decoder path (upsample + decoder transformer + SEANet decoder blocks) producing PCM-like output at configured sample rate — `internal/native/mimi.go`
  - [x] Added Mimi encoder hook surface (`EncodeVoiceHook` / `EncodeToLatent`) with explicit `not implemented` sentinel for future Phase 20 integration — `internal/native/model.go`, `internal/native/mimi.go`

- [x] Task 27.3: **Module-level parity tests**
  - [x] Added ONNX parity integration tests for representative modules (`flow_lm_flow`, `latent_to_mimi`) using real checkpoint + ONNX sessions — `internal/native/parity_integration_test.go`
  - [x] Added parity reporting helper with max abs/rel error + shape checks and per-kernel tolerance contract integration — `internal/native/parity.go`, `internal/runtime/ops/tolerance.go`
  - [x] Current status note: `latent_to_mimi` meets configured tolerance; `flow_lm_flow` currently reports out-of-tolerance drift but is tracked non-fatally with explicit metrics while the flow-net port is tightened in follow-up phases

---

## Phase 28 — End-to-End AR Inference (No ONNX)

> **Goal:** Run the full generation loop from text tokens to PCM entirely on safetensors-native modules.

- [x] Task 28.1: **AR loop integration**
  - [x] Wired safetensors-native runtime pipeline: text conditioning -> AR step -> flow decode -> latent accumulation -> Mimi decode in `internal/tts/runtime_native_safetensors.go` using `internal/native` modules
  - [x] Reused Phase 18 generation controls and EOS countdown semantics in native runtime (`Temperature`, `EOSThreshold`, `MaxSteps`, `LSDDecodeSteps`, `FramesAfterEOS`)
  - [x] Kept existing ONNX (`native-onnx`) runtime path unchanged and selectable alongside native-safetensors

- [x] Task 28.2: **Voice conditioning integration**
  - [x] Added voice embedding prepend in safetensors-native runtime exactly at text embedding stage (`[1,T_voice,D] + [1,T_text,D]` along dim=1) in `internal/tts/runtime_native_safetensors.go`
  - [x] Added integration coverage verifying no-voice vs with-voice synthesis divergence via PCM hash comparison in `internal/tts/native_safetensors_integration_test.go`

- [x] Task 28.3: **Integration tests**
  - [x] Added `integration` tests for short/medium prompts, chunked text, and voice-conditioned synthesis on `native-safetensors` backend in `internal/tts/native_safetensors_integration_test.go`
  - [x] Added CLI regression test `synth --backend native-safetensors` producing valid WAV in `cmd/pockettts/synth_cli_integration_test.go`
  - [x] Updated backend wiring in CLI/server so `native-safetensors` executes native runtime instead of returning \"not implemented yet\" (`cmd/pockettts/synth.go`, `internal/server/server.go`)

---

## Phase 29 — Stateful Inference, Streaming, and Concurrency

> **Goal:** Bring safetensors-native runtime to feature parity with planned advanced runtime phases.

- [ ] Task 29.1: **KV-cache/state reuse**
  - [x] Implemented stateful AR cache lifecycle aligned with Phase 23 objectives (without ONNX state tensors):
    - native FlowLM now exposes `InitState` + `PromptText` + incremental `SampleNextLatentStateful`
    - flow transformer gained per-layer KV cache state and `prefill`/`step` execution path
    - safetensors runtime switched from stateless full-sequence recompute per step to prompt-once + incremental stepping
  - [x] State reset/reuse correctness:
    - state is initialized per chunk/request in runtime and never shared across synth calls
    - ONNX (`native-onnx`) path remains unchanged and selectable

- [x] Task 29.2: **Streaming synthesis**
  - [x] Chunk-level streaming: each ≤50-token text chunk is generated, decoded, and flushed as a
    `PCMChunk` to the client immediately via `/tts/stream` (POST, `audio/wav`, chunked transfer
    encoding with `0xFFFFFFFF` unknown-length WAV header).
  - [x] `Service.SynthesizeStream(ctx, input, voice, out chan<- PCMChunk)` sends one chunk per text
    segment, closes channel on return, respects context cancellation and backpressure.
  - [x] `StreamingSynthesizer` interface + `WithStreamer` option; `nativeSynthesizer` implements both
    `Synthesizer` and `StreamingSynthesizer`. CLI backend returns 501 Not Implemented.
  - [x] New `internal/audio/wav_stream.go`: `WriteWAVHeaderStreaming`, `WritePCM16Samples`.
  - [x] Note: true frame-level Mimi streaming deferred (requires stateful decoder rewrite).

- [x] Task 29.3: **Concurrency + resource control**
  - [x] Context propagation: new `Service.SynthesizeCtx(ctx, …)` method; `Synthesize` delegates with
    `context.Background()`. `nativeSynthesizer.Synthesize` now passes HTTP handler context through.
  - [x] Native backend semaphore enabled: `runtimeDeps()` returns `workers=2` (default) instead of 0.
    Existing semaphore, timeout, and queueing logic now applies to native mode.
  - [x] Queue awareness logging via `acquireWorker` helper (two-stage select, logs when queued).
  - [ ] Memory budgeting for model weights, cache, and per-request buffers (deferred to future phase)

---

## Phase 30 — CLI/Server Productization + Doctor

> **Goal:** Make safetensors-native runtime first-class in CLI/server UX and operational checks.

- [ ] Task 30.1: **Config + command wiring**
  - [ ] Add config defaults/env/flags for `native-safetensors` backend and runtime-specific tuning knobs
  - [ ] Keep backward compatibility for existing ONNX backend flags and manifests during transition

- [ ] Task 30.2: **Doctor + verify tooling**
  - [ ] Extend `doctor` checks to validate safetensors model presence, key schema, and runtime readiness
  - [ ] Add `model verify --backend native-safetensors` to run smoke inference without ONNX assets

- [ ] Task 30.3: **User docs**
  - [ ] Update README/INSTALL with "ONNX path" and "safetensors-native path" setup matrices
  - [ ] Document checkpoint compatibility, remap policy, and troubleshooting guide

---

## Phase 31 — Rollout, Benchmarks, and Default Switch ✅ Complete

> **Goal:** Safely move from ONNX-default to safetensors-native-default once parity and performance are proven.

- [x] Task 31.1: **Parity gate** — `flow_lm_flow` parity test passes within tolerance (MaxAbsErr < 0.0002)
- [x] Task 31.2: **RMS norm bug fix** — `rmsNormWithAlpha` was using `mean(x²)` (standard RMS) instead of `var(x)` with Bessel correction (N-1) as the Python `_rms_norm` implementation requires. Despite the misleading function name, Python uses variance-based normalization. Fixed; parity error dropped from MaxAbsErr=1.04 to within tolerance.
- [x] Task 31.3: **Default migration** — `BackendNative` now maps to `"native-safetensors"`. The `"native"` alias resolves to safetensors. ONNX backend remains available as `"native-onnx"` (`config.BackendNativeONNX`).

---

## Phase 32 — ONNX Backend: Fix Long-Text Garbled Beginning

> **Goal:** Fix the garbled audio at the beginning of ONNX-generated speech for longer inputs.

**Problem:** The ONNX `flow_lm_main` backend produces garbled audio at the beginning of long text (e.g. "This is a really long sentence. But it works fine." → garbled start, then clear "really long sentence. But it works fine."). Short text ("Hello.") generates correctly.

**Root cause analysis:** The ONNX `FlowLMMainWrapper` (in `scripts/export_onnx.py`) creates a **fresh KV-cache state every call** (`state = clone_model_state(self.base_state)`). The Go AR loop passes the full growing sequence + text embeddings each step, so the ONNX model re-processes the entire `[voice(125) + text(T) + seq(S)]` context from scratch on every iteration — no persistent KV-cache across steps.

This non-incremental approach is architecturally different from the Python reference, which:
1. Initializes KV-cache once (or loads pre-computed v2 voice embeddings)
2. Prompts text into the cache (incremental prefill)
3. Steps autoregressively with single-frame inputs, cached K/V from prior steps

The native-safetensors backend already follows the correct incremental pattern and produces clear audio for all text lengths.

**Implemented approach (Task 32.1):**

- [x] Task 32.1: **Re-export ONNX with incremental interface** — Exported `flow_lm_prefill` and `flow_lm_step` graphs with explicit KV-cache I/O. Go `GenerateAudio` dispatches to the stateful path when these graphs are present, falling back to the legacy stateless `flow_lm_main` path for older bundles.
  - `scripts/export_onnx.py`: added `FlowLMPrefillWrapper`, `FlowLMStepWrapper`, `extract_kv_tensors`, `rebuild_state_from_kv` helpers and two new `ExportSpec` entries
  - `internal/onnx/flow_lm.go`: added `FlowLMKVState`, `Engine.FlowLMPrefill`, `Engine.FlowLMStepStateful`
  - `internal/onnx/generate.go`: `GenerateAudio` dispatcher → `generateAudioStateful` (prefill+step) or `generateAudioStateless` (legacy); shared `decodeLatentsToAudio`
  - Integration test added (`TestGenerateAudioIntegration_StatefulPath_ProducesPlausibleAudio`); skips when re-exported bundle unavailable
  - **Pending:** Re-run `scripts/export_onnx.py` in a Python+PyTorch environment to produce `flow_lm_prefill.onnx` and `flow_lm_step.onnx` and update the manifest
- [ ] Task 32.2: Superseded by 32.1 (deferred)
- [ ] Task 32.3: **Evaluate deprecation** — Deferred; ONNX backend kept for compatibility

---

## Phase 33 — MimiDecode Memory & Parallelism Optimizations

> **Goal:** Eliminate the remaining allocation pressure and sequential bottlenecks in the native MimiDecode path (`internal/runtime/ops`), following the Conv1D/ConvTranspose1D im2col+AVX2 work from Phase 31.
>
> **Context:** After Phase 31 / Phase 33 preparatory work, `BenchmarkMimiDecode` (20 latent frames) runs in ~1.5 s with ~327 MB/op of transient allocations. Three targeted tasks address the remaining cost.

- [x] Task 33.1: **Pool im2col and kernelT scratch buffers** — Added `scratchPools` (17 size-class `sync.Pool`s, powers of two from 2^10 to 2^26 floats) with `getScratch`/`putScratch` helpers. Replaced all `make([]float32, …)` calls in `conv1DFastGroups1` (imcol), `convTranspose1DGroups1` (kernelT, inputT, temp) with pooled scratch buffers. Oversized requests (>256 MB) fall back to plain allocation. Result: `BenchmarkMimiDecode` allocations dropped from 327 MB/op → 259 MB/op (68 MB saved, 21%); timing stable at ~1.4 s/op.
  - Files changed: `internal/runtime/ops/ops.go`
  - No API or correctness changes; pool is internal to the `ops` package.

- [x] Task 33.2: **Precompute `kernelT` at model load time** — Added `ops.RepackConvTransposeKernel` (exported one-time repack function) and `ops.ConvTranspose1DPrePacked` (accepts pre-packed kernelT). `convTr1dLayer` now stores a `kernelT []float32` field, populated in `loadConvTr1D` when `groups == 1`. `convTranspose1DGroups1` accepts `prePackedKernelT` parameter: non-nil skips repack, nil falls back to dynamic repack.
  - Files changed: `internal/runtime/ops/ops.go`, `internal/native/mimi.go`

- [x] Task 33.3: **Parallelize the `oc` loop in `conv1DFastGroups1`** — Added `--conv-workers` CLI flag (default 1 = sequential; `POCKETTTS_CONV_WORKERS` env). When workers ≥ 2, `parallelFor` splits the output-channel loop across goroutines using `sync.WaitGroup`. Applied to both `conv1DFastGroups1` (oc GEMM loop) and `convTranspose1DGroups1` (restructured from kx→ix→oc to oc-outer for parallelism; each oc range writes to disjoint output rows). Race-detector verified. Benchmark results on i7-1255U: workers=2 gives ~1.3× on MimiDecode, workers=4/8 limited by memory bandwidth (large late-stage im2col matrices overflow L3 cache). Best used for small-to-medium convolutions that fit in cache.
  - Files changed: `internal/runtime/ops/ops.go` (SetConvWorkers, parallelFor, restructured loops), `internal/config/config.go` (RuntimeConfig.ConvWorkers, --conv-workers flag), `internal/tts/service.go` (wire ops.SetConvWorkers on native backend init)
  - Tests added: `TestConv1DParallel`, `TestConvTranspose1DParallel`, `BenchmarkConv1DMimiParallel` (all race-clean)
