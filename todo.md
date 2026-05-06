# PocketTTS Upstream Reintegration TODO

Source inspected: `original/pockettts` is a clean checkout at `2dff8a2d1b3b21bf44ecf0084cc8ce79ab6d6bba`.

This list is organized for reintegrating upstream Python changes into this Go port. Code-path changes that can affect native inference or voice compatibility are first; docs-only upstream changes are last.

## Progress

- [x] 2026-05-05: Added absolute-position attention support (`AttentionWithPositions`) with invalid-key and context-window masking tests.
- [x] 2026-05-05: Moved FlowLM streaming transformer state from implicit concatenated length to explicit per-layer `offset` tracking.
- [x] 2026-05-05: Added `I64` safetensors decoding and detection/loading scaffolding for upstream full model-state voice files.
- [x] 2026-05-05: Hydrate upstream full voice states into native `FlowLMState` and route them through native synthesis.
- [x] 2026-05-05: Added growable FlowLM K/V cache capacity and invalid-position masking for unused cache slots.
- [x] 2026-05-06: Added a Python reference dump script plus optional Go parity tests for FlowLM prefill/step and Mimi decode fixtures.
- [x] 2026-05-06: Synced `original/pockettts` to latest `main`, installed upstream Python dependencies in `.venv`, generated `/tmp/native_runtime_parity.json`, and passed the P0 native Python parity tests.
- [x] 2026-05-06: Added upstream model-state export mode to `export-voice`, added browser/WASM full-state voice ingestion, refreshed local default voices to upstream model-state format, and passed a native synth smoke with `--voice alba`.
- [x] 2026-05-06: Completed P1 generation sizing and CLI parity: dynamic per-chunk max generation estimates, config-derived Mimi step metadata, `--text -` stdin alias, and docs for Go vs upstream Python config/model paths.
- [x] Remaining P0 runtime work: run Python parity fixtures.

## P0: Native Runtime Compatibility

- [x] Port the upstream fused transformer/cache implementation.
  - Upstream source: `2dff8a2`, especially `pocket_tts/modules/transformer.py`, `pocket_tts/modules/mimi_transformer.py`, and `pocket_tts/models/tts_model.py`.
  - Go targets: `internal/native/flow_transformer.go`, `internal/runtime/ops/attention.go`, `scripts/export_onnx.py`.
  - [x] Replace shape-derived `current_end` semantics with explicit per-layer `offset`.
  - [x] Add legacy voice-state import scaffolding that translates `current_end.shape[0]` to `offset`.
  - [x] Match upstream logical KV cache semantics for current Go state: append at `offset`, then expose valid keys/values through `offset + input_len`.
  - [x] Avoid repeatedly concatenating K/V tensors in the hot path if practical; upstream preallocates and expands cache capacity.
  - [x] Add parity tests for prompt prefill plus one autoregressive step against Python reference tensors.
  - [x] Generate and run the Python reference fixture after installing upstream Python dependencies.

- [x] Replace causal-offset-only attention masking with upstream position-based masking.
  - Upstream source: `pocket_tts/modules/transformer.py::_build_attention_mask`.
  - [x] Required behavior: `delta = pos_q - pos_k`, allow keys only when `pos_k >= 0`, `delta >= 0`, and, when a context is configured, `delta < context`.
  - [x] Go target: `internal/runtime/ops/attention.go`.
  - [x] Keep the existing fast 4D attention path, but add tests for absolute positions, invalid key positions, and context-limited attention.

- [x] Align RoPE offsets with upstream cache offsets.
  - Upstream applies RoPE using the cache backend's current `offset` before appending new K/V.
  - Go target: `internal/native/flow_transformer.go`.
  - [x] Use the explicit cache `offset` as the RoPE position before appending new K/V.
  - [x] Verify full prompt mode and step mode both use absolute positions identical to upstream.

- [x] Recheck Mimi transformer behavior after upstream fusion.
  - Upstream source: `2dff8a2` uses the generic `StreamingMultiheadAttention` with a context window.
  - Go target: `internal/native/mimi.go`.
  - [x] Use upstream-style position-based attention with Mimi's configured context window.
  - [x] Add fixture generator and optional Go parity tests for `LatentToMimi` plus `MimiDecode` on 1, 2, and several latent frames.
  - [x] Generate and run the Mimi Python reference fixture after installing upstream Python dependencies.

## P0: Voice State Safetensors

- [x] Support upstream full model-state voice files.
  - Upstream source: `8443de8`, `pocket_tts/models/tts_model.py::export_model_state` and `_import_model_state`.
  - Current Go behavior reads/writes a single `audio_prompt` tensor. Upstream now stores flattened model state tensors as `<module>/<tensor>`, for example cache and `offset` per transformer module.
  - Go targets: `internal/safetensors/reader.go`, `internal/tts/service.go`, `cmd/pockettts/export_voice.go`, `cmd/pockettts-wasm/main_wasm.go`, native FlowLM state types.
  - [x] Teach safetensors voice-file inspection to detect both formats:
    - legacy/simple `audio_prompt` embedding files,
    - upstream full state files with keys containing `/cache`, `/offset`, or legacy `/current_end`.
  - [x] Add typed loading scaffolding for upstream full state files and translate legacy `/current_end` tensors to `/offset`.
  - [x] For full state files, hydrate `FlowLMState` directly and skip re-encoding the audio prompt.
  - [x] Preserve support for existing local voice manifests until regenerated.

- [x] Update `export-voice` to produce upstream-compatible model-state safetensors.
  - Upstream source: `8443de8`, `pocket_tts/main.py::export_voice`.
  - Upstream CLI now exports the full prompted model state, not just `audio_prompt`.
  - Go target: `cmd/pockettts/export_voice.go`.
  - [x] Implemented Python tooling fallback via `pocket-tts export-voice` behind `--format=model-state`.
  - [x] Kept `--format=legacy-embedding` as the default so existing ONNX/native embedding workflows remain available.
  - [x] Added a regression test for the model-state export path and ran a real smoke export to `/tmp/go_exported_voice_state.safetensors`.

- [x] Regenerate or redownload default voices in the new format.
  - Upstream source: `original/pockettts/scripts/generate_default_voices.py`.
  - The script exports current default language voices to `.safetensors`.
  - [x] Redownloaded the existing manifest voice IDs locally as upstream `english_2026-01` full model-state files.
  - [x] Kept `voices/manifest.json` stable.
  - [x] Verified `pockettts synth --voice alba` with the refreshed full-state voice.

## P1: Generation Cache Sizing and Streaming

- [x] Mirror upstream dynamic generation length estimation.
  - Upstream source: `a47c7e4` and `33a9371`, `TTSModel._estimate_max_gen_len`.
  - Formula: `ceil((token_count / 3.0 + 2.0) * mimi.frame_rate)`.
  - [x] Use the estimate as the default per-chunk generation step limit and pass Mimi sequence sizing metadata into runtime configs.

- [x] Replace hardcoded Mimi decoder step assumptions with config-derived values.
  - Upstream source: `33a9371`, `_generate_audio_stream_short_text`.
  - Compute `mimi_steps_per_latent = int(mimi.encoder_frame_rate / mimi.frame_rate)`.
  - Compute `mimi_sequence_length = max_gen_len * mimi_steps_per_latent`.
  - [x] Go targets: `internal/native/mimi.go`, service streaming path when frame-level streaming is added.

- [x] Revalidate chunked long-text generation against upstream.
  - Upstream keeps simple sentence splitting and adds `frames_after_eos_guess + 2` per chunk.
  - Go targets: `internal/text`, `internal/tts`, `cmd/pockettts/synth.go`.
  - [x] Add a parity test for short text, multi-sentence text, and very short prompts with eight-space padding.

## P1: CLI and API Parity

- [x] Add `--text -` as an explicit stdin alias for `pockettts synth`.
  - Upstream source: `431878c`, `pocket_tts/main.py::generate`.
  - Current Go behavior reads stdin only when `--text` is empty. Keep that, but also treat `--text -` as stdin for upstream CLI compatibility.
  - [x] Go target: `cmd/pockettts/synth.go`.

- [x] Check local config/model-file behavior.
  - Upstream source: `2b51649` and `28b8244`.
  - Upstream accepts either a model variant signature or a path to a local config `.yaml`.
  - [x] Confirm Go tooling and docs clearly map this to `--config`, `--tts-cli-config-path`, model safetensors paths, and ONNX manifest paths.

- [x] Decide whether to mirror upstream `export-voice` CLI simplification.
  - Upstream removed most generation flags from `export-voice` and always calls `get_state_for_audio_prompt(..., truncate=True)`.
  - [x] Kept the richer native/ONNX-oriented command surface and documented `--format=model-state` for upstream-compatible export.

- [x] Expose an equivalent public API for voice-state import/export if this port grows a Go library API.
  - Upstream source: `8443de8`, `pocket_tts/__init__.py`.
  - [x] No public Go library API exists yet; internal voice-state load/export seams are in place and documented for future promotion.

## P2: Device, ONNX, and Browser Details

- [x] Fold in CUDA/device-state fixes where relevant.
  - Upstream source: `a5da339`, `aca7dc8`.
  - Python fixed state tensor device placement for streaming convs and Mimi attention, then removed the old CUDA graph workaround.
  - Pure Go native does not need CUDA handling, but ONNX/GPU paths should ensure state buffers live on the execution provider's device when applicable.
  - [x] 2026-05-06: Confirmed no native-Go CUDA/device action is needed. The Go ONNX path passes host tensors through `onnxruntime-purego`; full upstream model-state voice files remain native/Python-only and are rejected before ONNX state handling.

- [x] Update ONNX export/runtime contract after transformer/state changes.
  - Upstream transformer cache schema changed to `offset`.
  - Go target: `scripts/export_onnx.py`, `bundles/onnx-bundles.lock.json`, ONNX backend tests.
  - Confirm exported `flow_lm_prefill` and `flow_lm_step` use the same KV order, cache shape, and offset updates expected by Go.
  - [x] 2026-05-06: Updated `scripts/export_onnx.py` for the current upstream `TTSModel.load_model(language=...|config=...)` API while preserving legacy `--variant=b6369a24` as an alias for `english_2026-01`.
  - [x] 2026-05-06: Pinned Go-side ONNX state behavior with a unit test that `FlowLMStepStateful` passes the current `offset` input and stores the returned offset.

- [ ] Publish the rebuilt ONNX graph bundle.
  - [x] 2026-05-06: Rebuilt local ignored `models/onnx` for `english_2026-01` with `flow_lm_prefill`, `flow_lm_step`, and the corrected `kv_out_N`/`offset_out` step outputs.
  - [x] 2026-05-06: Fixed `mimi_decoder` export sizing so its state length is `max_latents * mimi_steps_per_latent`; this passes the 256-frame stateful integration path.
  - [x] 2026-05-06: Created upload-ready archive `/tmp/pockettts-onnx-english_2026-01-stateful.tar.gz` (468.1M), SHA256 `8d5124e35cc609a35c4ad038c532498189f3d40fdfef1f6a0f931a7ce3f070f6`.
  - [ ] `bundles/onnx-bundles.lock.json` still needs the final published artifact URL plus this checksum after upload.

- [x] Consider the upstream web `AudioContext` latency hint.
  - Upstream source: `9878cd0`, `pocket_tts/static/index.html`.
  - Go target: `web/main.js` if the browser demo creates an `AudioContext`.
  - [x] 2026-05-06: No code change needed in `web/main.js`; this demo attaches a completed WAV `Blob` to an `HTMLAudioElement` and does not create an `AudioContext`. The upstream `{ latencyHint: "playback" }` applies to its streaming `AudioContext` player.

## P2: Docs and Metadata

- [ ] Update docs for full-state voice safetensors.
  - Upstream source: `edec7f8`, `8443de8`, `e43b485`.
  - Explain that upstream `.safetensors` voice files are model-state/KV-cache files, not just raw audio embeddings.
  - Include compatibility notes for legacy `audio_prompt` files created by earlier Go tooling.

- [ ] Update docs links and project references only if useful.
  - Upstream sources: `2f6ac56`, `4ae9fb2`, `5d8b15c`, `ef56ab4`, `d773bf9`, `a1341e8`, `cb555a5`, `6ee89b1`, `09d81f7`.
  - These are mostly README/docs-site changes and related-project listings; low runtime value for this Go port.

- [ ] Note version alignment.
  - Inspected upstream checkout is `2dff8a2`; upstream `pyproject.toml` currently reports `2.1.0`.
  - Record the upstream commit hash in release notes once compatibility work is complete.

## Verification Checklist

- [x] Add Python reference dump scripts under `scripts/` or `tests/parity/` that run against `original/pockettts`.
- [x] Generate fixtures for FlowLM prefill, one-step FlowLM, latent-to-Mimi, and Mimi decode.
- [ ] Generate fixtures for tokenizer output, text embeddings, and voice model state.
- [x] Add Go tests for loading both legacy `audio_prompt` voice files and upstream full-state voice files.
- [ ] Add Go/Python parity tests for attention masks using offset/context edge cases.
- [ ] Run `go test ./...` after each reintegration slice.
- [x] Run an end-to-end synth smoke test with an upstream-generated `.safetensors` voice.
