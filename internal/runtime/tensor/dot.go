package tensor

// dotF32Generic computes the dot product of two equal-length float32 slices.
// Used as the fallback on platforms without assembly (e.g. WASM).
//
// The loop is manually unrolled with 4 independent accumulators to break
// the FP dependency chain — on WASM this lets V8/SpiderMonkey pipeline
// the F32Mul+F32Add ops instead of stalling on each addition.
// The BCE hints (bounds-check elimination) avoid per-element index checks.
func dotF32Generic(a, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	// BCE: prove all indices in [0, n) are valid once.
	_ = a[n-1]
	_ = b[n-1]

	var s0, s1, s2, s3 float32

	i := 0
	for ; i+7 < n; i += 8 {
		s0 += a[i+0] * b[i+0]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
		s0 += a[i+4] * b[i+4]
		s1 += a[i+5] * b[i+5]
		s2 += a[i+6] * b[i+6]
		s3 += a[i+7] * b[i+7]
	}

	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}

	return s0 + s1 + s2 + s3
}

// DotProduct returns the dot product of a and b.
// len(a) must equal len(b); the caller is responsible for this.
func DotProduct(a, b []float32) float32 {
	return dotF32(a, b)
}
