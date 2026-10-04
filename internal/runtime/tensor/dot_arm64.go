//go:build arm64

package tensor

// dotNEONMinLen is the length from which the NEON kernel beats the unrolled
// Go loop: below it the call and the horizontal reduction cost more than the
// vector loop saves (M5 Pro: Go 2.4 ns vs NEON 6.3 ns at n=8, 7.8 vs 9.6 ns
// at n=44, NEON ahead from n=48).
const dotNEONMinLen = 48

// dotF32 dispatches to NEON assembly on ARM64. All AArch64 CPUs have NEON,
// so no runtime feature detection is needed.
func dotF32(a, b []float32) float32 {
	if len(a) >= dotNEONMinLen {
		return dotF32NEON(&a[0], &b[0], len(a))
	}

	return dotF32Generic(a, b)
}

// dotF32NEON computes the dot product of the n float32 values starting at a
// and b using NEON instructions. n must be >= 1.
//
//go:noescape
func dotF32NEON(a, b *float32, n int) float32
