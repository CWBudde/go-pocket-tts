# PLAN

> **Goal:** Go CLI and HTTP server for PocketTTS. Phases 0–15 complete. Active work: Phases 16–19 (native ONNX inference pipeline). Future: Phases 20–22 (pure-Go replacements for CGO dependencies).

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

- [ ] Task 16.1: **Create `Runner` type for ONNX graph execution**
  - [ ] `internal/onnx/runner.go`: `Runner` wraps `ort.Runtime` + `ort.Env` + `ort.Session` per graph
  - [ ] `Run(ctx, inputs map[string]*Tensor) (map[string]*Tensor, error)` — converts to/from ORT tensors
  - [ ] Handle float32/int64 dtype dispatch via `ort.NewTensorValue` and `ort.GetTensorData`
  - [ ] `Close()` for session cleanup (idempotent)
  - [ ] Tests: round-trip with `identity_float32.onnx` from `model/testdata/`

- [ ] Task 16.2: **Add `cfg.Paths.ONNXManifest` config field**
  - [ ] Add `ONNXManifest string` to `PathsConfig`, default `"models/onnx/manifest.json"`
  - [ ] Register flag `--paths-onnx-manifest`, env `POCKETTTS_ONNX_MANIFEST`
  - [ ] Wire defaults, aliases

- [ ] Task 16.3: **Refactor `Engine` to use `SessionManager` + `Runner`**
  - [ ] `NewEngine(manifestPath string, cfg RunnerConfig)` — loads manifest, creates `Runner` per graph
  - [ ] Remove sine-wave stub and `threads` field
  - [ ] `Engine.Runner(name) (*Runner, bool)` accessor
  - [ ] `Engine.Close()` to tear down all runners
  - [ ] Temporary `Infer()` shim returns error (real pipeline in Phase 18)

- [ ] Task 16.4: **Update `tts.Service` and CLI callers**
  - [ ] `NewService(cfg)` uses `DetectRuntime()` + `NewEngine(cfg.Paths.ONNXManifest, rcfg)`
  - [ ] Add `Service.Close()` to clean up engine
  - [ ] Verify `synth` and `serve` commands still compile and work

- [ ] Task 16.5: **Full test sweep**
  - [ ] `go test ./...` — all existing tests pass
  - [ ] `golangci-lint run ./...` — no new warnings
  - [ ] Integration tests with identity ONNX model via `Runner`

---

## Phase 17 — SentencePiece Tokenizer + Text Conditioning

> **Goal:** Replace the unicode code-point preprocessor with SentencePiece tokenization via CGO, and run the `text_conditioner` ONNX graph to produce text embeddings.

- [ ] Task 17.1: **Add SentencePiece CGO binding**
  - [ ] Evaluate and add Go SentencePiece binding (e.g. `github.com/pquerna/sentencepiece-go` or direct CGO wrapper)
  - [ ] Create `internal/tokenizer/tokenizer.go` with `Tokenizer` interface: `Encode(text string) ([]int64, error)`
  - [ ] Implement `SentencePieceTokenizer` that loads `tokenizer.model` file

- [ ] Task 17.2: **Add config for tokenizer model path**
  - [ ] Add `cfg.Paths.TokenizerModel` field, default `"models/tokenizer.model"`
  - [ ] Bind to `--tokenizer-model` flag and `POCKETTTS_TOKENIZER_MODEL` env var
  - [ ] Wire into `Engine` or `tts.Service` construction

- [ ] Task 17.3: **Text preprocessing matching reference implementation**
  - [ ] Capitalize first letter of input
  - [ ] Add trailing period if last character is alphanumeric
  - [ ] Pad with 8 leading spaces if < 5 words
  - [ ] Normalize whitespace (newlines → spaces, collapse doubles)
  - [ ] Calculate `max_frames = ceil(num_tokens/3 + 2) × 12.5`
  - [ ] Calculate `frames_after_eos`: 3 if ≤4 words, else 1
  - [ ] Update sentence chunking to respect 50-token budget (tokenize first, then split)

- [ ] Task 17.4: **Run `text_conditioner` ONNX graph**
  - [ ] Tokenize input text → `[]int64`
  - [ ] Create input tensor `tokens [1, T] int64`
  - [ ] Run `text_conditioner` via `Runner` → extract `text_embeddings [1, T, 1024] float32`
  - [ ] Return embeddings for use in generation loop

- [ ] Task 17.5: **Tests**
  - [ ] Unit test: verify SentencePiece tokenizer produces correct token IDs for known inputs
  - [ ] Unit test: verify text preprocessing (capitalization, padding, punctuation)
  - [ ] Integration test (`integration` tag): tokenize + run `text_conditioner`, verify output shape `[1, T, 1024]`

---

## Phase 18 — Autoregressive Generation Loop

> **Goal:** Implement the full generation pipeline: `flow_lm_main` autoregressive loop, `flow_lm_flow` Euler ODE integration, `latent_to_mimi` + `mimi_decoder` audio decoding. `Engine.Infer()` produces real 24 kHz speech audio.

- [ ] Task 18.1: **Implement `flow_lm_main` autoregressive loop**
  - [ ] Initialize BOS: `sequence = NaN [1, 1, 32]` (graph handles NaN→bos_emb internally)
  - [ ] Each step: run `flow_lm_main(sequence, text_embeddings)` → `last_hidden [1, 1024]`, `eos_logits [1, 1]`
  - [ ] EOS detection: `eos_logits[0] > eos_threshold` (default `-4.0`, raw logits — **not** sigmoid)
  - [ ] On first EOS: start countdown (`frames_after_eos`), continue generating
  - [ ] After countdown expires or `max_steps` reached: stop
  - [ ] Each step: decode `last_hidden` → latent frame via flow (Task 18.2), append `[1, 1, 32]` to sequence
  - [ ] Note: sequence grows each step; `flow_lm_main` re-processes entire sequence (O(n²), no KV cache)

- [ ] Task 18.2: **Implement Euler flow integration (LSD decode)**
  - [ ] Sample noise: `x ~ N(0, sqrt(temperature))` shape `[1, 32]`; `temperature` default `0.7`
  - [ ] For `i` in `0..lsd_decode_steps` (default 1):
    - `s = i / steps`, `t = (i+1) / steps`
    - Create `s [1,1]`, `t [1,1]` tensors
    - Run `flow_lm_flow(last_hidden, s, t, x)` → `flow_dir [1, 32]`
    - `x += flow_dir / steps`
  - [ ] Return `x` reshaped to `[1, 1, 32]`

- [ ] Task 18.3: **Implement `latent_to_mimi` + `mimi_decoder` pipeline**
  - [ ] Stack all accumulated latent frames → `latent [1, T_frames, 32]`
  - [ ] Run `latent_to_mimi(latent)` → `mimi_latent [1, 512, T_frames]`
  - [ ] Run `mimi_decoder(mimi_latent)` → `audio [1, 1, N_samples]`
  - [ ] Extract and return `[]float32` PCM samples

- [ ] Task 18.4: **Wire complete pipeline into `Engine.Infer()`**
  - [ ] Replace sine-wave stub with: tokenize → text_conditioner → AR loop → latent_to_mimi → mimi_decoder
  - [ ] Return 24 kHz float32 PCM audio
  - [ ] Update `audio.EncodeWAV` calls to use 24000 Hz sample rate (was 22050)

- [ ] Task 18.5: **Add generation config fields to `TTSConfig`**
  - [ ] `Temperature float64` (default `0.7`)
  - [ ] `EOSThreshold float64` (default `-4.0`)
  - [ ] `MaxSteps int` (default `256`)
  - [ ] `LSDDecodeSteps int` (default `1`)
  - [ ] Bind to CLI flags: `--temperature`, `--eos-threshold`, `--max-steps`, `--lsd-steps`

- [ ] Task 18.6: **Tests**
  - [ ] Unit test: mock `Runner`, verify `Engine.Infer` calls graphs in correct order with correct tensor shapes and names
  - [ ] Unit test: verify EOS detection logic (countdown, threshold)
  - [ ] Unit test: verify Euler integration arithmetic (single step, multi-step)
  - [ ] Integration test (`integration` tag): `Engine.Infer("Hello.")` against real ONNX models → non-trivial audio, plausible length (~1–3s), not a sine wave
  - [ ] CLI regression: `pockettts synth --backend native --text "Hello world." --out /tmp/native.wav` → valid 24 kHz WAV

---

## Phase 19 — Voice Conditioning (.safetensors)

> **Goal:** Load pre-encoded voice embeddings from `.safetensors` files and inject them as conditioning prefix into the generation loop. Supports the predefined Kyutai voice embeddings.

- [ ] Task 19.1: **Implement safetensors reader**
  - [ ] Create `internal/safetensors/reader.go`
  - [ ] Parse safetensors format: 8-byte LE header length → JSON header → raw tensor data
  - [ ] Extract first tensor: name, dtype (float32), shape, raw bytes → `[]float32`
  - [ ] Handle both `[T, 1024]` and `[1, T, 1024]` shapes (reshape 2D → 3D)

- [ ] Task 19.2: **Inject voice embeddings into generation loop**
  - [ ] Extend `Engine.Infer` signature: `Infer(text string, opts ...InferOption)` with `WithVoice(path string)`
  - [ ] Load voice embedding via safetensors reader → `[1, T_voice, 1024]`
  - [ ] Prepend to `text_embeddings`: `concat([voice_emb, text_emb], dim=1)` before passing to `flow_lm_main`

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

## Future Phases (planned)

### Phase 20 — Pure Go SentencePiece (drop CGO tokenizer dependency)

> Replace CGO SentencePiece binding with a pure Go implementation. Parse the `tokenizer.model` protobuf, implement BPE merge algorithm natively. Eliminates `libsentencepiece` build dependency.

### Phase 21 — Voice encoding from audio (mimi_encoder + speaker_proj)

> Support encoding raw audio files into voice embeddings via the `mimi_encoder` ONNX graph + `speaker_proj_weight` linear projection. Enables creating voice embeddings from any audio without external tooling.

### Phase 22 — Pure Go safetensors (drop CGO if any, optimize)

> Harden the safetensors reader: support all dtypes, multiple tensors, large files via mmap. If any CGO dependency was introduced, replace with pure Go.

### Phase 23 — Streaming audio generation

> Generate and decode latent frames concurrently (producer/consumer pattern matching Rust/Python threading model). Enable real-time audio streaming from the HTTP server.

### Phase 24 — KV-cache ONNX graphs for O(n) inference

> Re-export `flow_lm_main` with stateful KV cache inputs/outputs. Each autoregressive step processes only the new frame instead of the full sequence, reducing inference from O(n²) to O(n).
