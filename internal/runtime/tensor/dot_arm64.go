//go:build arm64

package tensor

// dotF32 dispatches to NEON assembly on ARM64. All AArch64 CPUs have NEON,
// so no runtime feature detection is needed.
func dotF32(a, b []float32) float32 {
	if len(a) >= 4 {
		return dotF32NEON(&a[0], &b[0], len(a))
	}

	return dotF32Generic(a, b)
}

// dotF32NEON computes the dot product of the n float32 values starting at a
// and b using NEON instructions. n must be >= 1.
//
//go:noescape
func dotF32NEON(a, b *float32, n int) float32
