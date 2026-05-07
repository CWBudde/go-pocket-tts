# PLAN

> Open work items for go-pocket-tts. The core TTS pipeline (native-safetensors backend) is complete and default. Items below are improvements and hardening.

## Safetensors Hardening

- [ ] Memory-map large files (mmap for files > 64 MiB) with safe cleanup on `Close()`

## Streaming Audio Generation

True frame-level streaming — decode and flush latent frames as they are generated, rather than waiting for the full AR loop to finish.

- [ ] Run AR loop (FlowLM) in a producer goroutine, emit latent frames to a channel
- [ ] Run Mimi decoder in a consumer goroutine, emit PCM chunks
- [ ] Requires stateful Mimi decoder rewrite (currently stateless, processes all frames at once)

Note: chunk-level streaming (`/tts/stream`) is already implemented — each text chunk is flushed as it completes. This is about sub-chunk, frame-level streaming for lower latency.

## ONNX Backend

The ONNX backend (`native-onnx`) is functional but has a known issue: garbled audio at the beginning of longer text inputs. The Go-side stateful ONNX path (prefill+step) is implemented, and a rebuilt local bundle for `english_2026-01` exists.

- [ ] Publish rebuilt ONNX graph bundle
  - Local archive: `/tmp/pockettts-onnx-english_2026-01-stateful.tar.gz`
  - SHA256: `8d5124e35cc609a35c4ad038c532498189f3d40fdfef1f6a0f931a7ce3f070f6`
  - After upload, update `bundles/onnx-bundles.lock.json` with the final artifact URL and checksum
- [ ] Evaluate whether to keep or deprecate the ONNX backend long-term

## Upstream Parity

Follow-up items from the PocketTTS upstream reintegration pass checked against upstream commit `2dff8a2d1b3b21bf44ecf0084cc8ce79ab6d6bba`.

- [ ] Generate fixtures for tokenizer output, text embeddings, and voice model state
- [ ] Add Go/Python parity tests for attention masks using offset/context edge cases
- [ ] Keep running `go test ./...` after each reintegration slice
  - Current caveat: local ONNX-backed native parity tests can panic inside `onnxruntime-purego` with `runtime.AddCleanup`; `go test ./... -skip 'TestParity_.*_VsONNX'` passes.

## Performance

- [ ] Memory budgeting for model weights, KV-cache, and per-request buffers
- [ ] Im2col tiling for cache-friendliness on large convolutions (res3: 38400x192 imcol = 30 MB, overflows L3)

## Reference Architecture

Key constants (variant `b6369a24`):

- `ldim = 32`, `d_model = 1024`, `num_heads = 16`, `num_layers = 6`
- `flow_dim = 512`, `flow_depth = 6`
- `sample_rate = 24000`, `frame_rate = 12.5` (1920 samples/frame)
- `temperature = 0.7`, `eos_threshold = -4.0`, `lsd_decode_steps = 1`
- `n_bins = 4000` (SentencePiece vocabulary)

ONNX graphs (6 total): `text_conditioner`, `flow_lm_main`, `flow_lm_flow`, `latent_to_mimi`, `mimi_decoder`, `mimi_encoder`
