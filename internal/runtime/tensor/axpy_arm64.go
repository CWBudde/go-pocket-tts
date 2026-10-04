//go:build arm64

package tensor

// axpyNEONMinLen is the length from which the NEON kernel beats the Go loop
// (M5 Pro: Go 1.8 ns vs NEON 2.9 ns at n=6, NEON ahead from n=8).
const axpyNEONMinLen = 8

func axpyF32(dst []float32, alpha float32, src []float32) {
	n := len(dst)
	if n >= axpyNEONMinLen {
		n4 := n &^ 3
		axpyF32NEON(&dst[0], &src[0], alpha, n4)

		if n4 == n {
			return
		}

		axpyF32Generic(dst[n4:], alpha, src[n4:])

		return
	}

	axpyF32Generic(dst, alpha, src)
}

//go:noescape
func axpyF32NEON(dst, src *float32, alpha float32, n int)
