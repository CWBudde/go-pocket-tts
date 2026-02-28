package tensor

// Axpy computes dst += alpha * src element-wise.
// If src and dst lengths differ, the shorter length is used.
func Axpy(dst []float32, alpha float32, src []float32) {
	n := min(len(src), len(dst))

	if n == 0 || alpha == 0 {
		return
	}

	axpyF32(dst[:n], alpha, src[:n])
}

func axpyF32Generic(dst []float32, alpha float32, src []float32) {
	for i := range dst {
		dst[i] += alpha * src[i]
	}
}
