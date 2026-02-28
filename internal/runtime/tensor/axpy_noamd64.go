//go:build !amd64 && !arm64

package tensor

func axpyF32(dst []float32, alpha float32, src []float32) {
	axpyF32Generic(dst, alpha, src)
}
