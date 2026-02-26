//go:build amd64

package tensor

import "golang.org/x/sys/cpu"

// useAVX2FMA is initialised once at program start. True when the CPU has
// both AVX2 and FMA (needed for VFMADD231PS / VFMADD231SS).
var useAVX2FMA = cpu.X86.HasAVX2 && cpu.X86.HasFMA

// dotF32 returns the dot product of a and b.
// len(a) must equal len(b); the caller is responsible for this.
func dotF32(a, b []float32) float32 {
	if useAVX2FMA && len(a) >= 8 {
		return dotF32AVX2(&a[0], &b[0], len(a))
	}
	return dotF32Generic(a, b)
}

// dotF32AVX2 computes the dot product of the n float32 values starting at a
// and b using AVX2 + FMA instructions. n must be >= 1.
//
//go:noescape
func dotF32AVX2(a, b *float32, n int) float32
