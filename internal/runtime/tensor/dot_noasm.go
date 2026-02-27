//go:build !amd64 && !arm64

package tensor

func dotF32(a, b []float32) float32 {
	return dotF32Generic(a, b)
}
