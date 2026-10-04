//go:build !arm64

package tensor

func matMulTransB(c []float32, ldc int, a, b []float32, m, n, k int) {
	matMulTransBGeneric(c, ldc, a, b, m, n, k)
}
