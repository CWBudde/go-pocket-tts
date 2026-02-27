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

The ONNX backend (`native-onnx`) is functional but has a known issue: garbled audio at the beginning of longer text inputs. The Go-side stateful ONNX path (prefill+step) is implemented but requires re-exporting the ONNX graphs from Python.

- [ ] Re-run `scripts/export_onnx.py` to produce `flow_lm_prefill.onnx` and `flow_lm_step.onnx`
- [ ] Evaluate whether to keep or deprecate the ONNX backend long-term

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
