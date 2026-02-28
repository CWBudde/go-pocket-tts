//go:build arm64

package tensor

func axpyF32(dst []float32, alpha float32, src []float32) {
	n := len(dst)
	if n >= 4 {
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

