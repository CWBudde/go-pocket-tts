package tensor

import "fmt"

// MatMulTransB computes the m×n product of a [m×k] and the transpose of
// b [n×k], both row-major with row stride k:
//
//	c[i*ldc+j] = dot(a[i*k:(i+1)*k], b[j*k:(j+1)*k])   for i < m, j < n
//
// Entries of c outside those positions are left untouched. Callers add bias
// or accumulate themselves.
//
// On arm64 a register-blocked NEON kernel computes 4×4 blocks of c, so each
// loaded vector feeds four FMAs instead of one. Elsewhere it is one
// DotProduct per output, the same arithmetic as before.
func MatMulTransB(c []float32, ldc int, a, b []float32, m, n, k int) {
	if m < 0 || n < 0 || k < 0 || ldc < n {
		panic(fmt.Sprintf("tensor: MatMulTransB bad dims m=%d n=%d k=%d ldc=%d", m, n, k, ldc))
	}

	if m == 0 || n == 0 {
		return
	}

	if len(a) < m*k || len(b) < n*k || len(c) < (m-1)*ldc+n {
		panic(fmt.Sprintf("tensor: MatMulTransB short slices: len(a)=%d len(b)=%d len(c)=%d for m=%d n=%d k=%d ldc=%d",
			len(a), len(b), len(c), m, n, k, ldc))
	}

	matMulTransB(c, ldc, a, b, m, n, k)
}

func matMulTransBGeneric(c []float32, ldc int, a, b []float32, m, n, k int) {
	for i := range m {
		aRow := a[i*k : (i+1)*k]
		cRow := c[i*ldc : i*ldc+n]

		for j := range cRow {
			cRow[j] = dotF32(aRow, b[j*k:(j+1)*k])
		}
	}
}
