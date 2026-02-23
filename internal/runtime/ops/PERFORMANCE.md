# Runtime Kernel Performance Notes

This package intentionally starts with deterministic, single-threaded CPU kernels
as the correctness baseline for safetensors-native inference.

## Current benchmark entry points

Run:

```bash
go test -bench . -run ^$ ./internal/runtime/ops
```

Benchmarks include:

- `BenchmarkMatMulFlowLM`
- `BenchmarkLayerNormFlowLM`
- `BenchmarkAttentionFlowLM`
- `BenchmarkConv1DMimi`
- `BenchmarkFrameDecodeThroughput`

## Optional acceleration points (correctness contract unchanged)

1. `MatMul` and `Linear`: SIMD/assembly kernels for inner-product loops.
2. `Softmax` + masking: fuse max/exp/sum/normalize pass and mask application.
3. `LayerNorm`: vectorized mean/variance reduction and affine fuse.
4. `Attention`: fuse `QK^T` scale + mask + softmax where memory reuse is possible.
5. `Conv1D`/`ConvTranspose1D`: im2col + GEMM or direct SIMD kernels for common kernel sizes.
6. Parallelization: shard batch/head/channel loops with worker pools while keeping deterministic reduction order where parity requires it.

Any acceleration must preserve the tolerance targets in `tolerance.go`.
