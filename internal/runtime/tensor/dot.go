package tensor

// dotF32Generic computes the dot product of two equal-length float32 slices
// using a scalar loop. Used as the fallback on non-AVX2 platforms.
func dotF32Generic(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}

// DotProduct returns the dot product of a and b.
// len(a) must equal len(b); the caller is responsible for this.
func DotProduct(a, b []float32) float32 {
	return dotF32(a, b)
}
