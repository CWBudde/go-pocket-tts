package tensor

import "math"

func equalI64(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

func equalF32(a, b []float32, tol float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if tol == 0 {
			if a[i] != b[i] {
				return false
			}

			continue
		}

		if math.Abs(float64(a[i]-b[i])) > tol {
			return false
		}
	}

	return true
}

func make16(val float32) []float32 {
	s := make([]float32, 16)
	for i := range s {
		s[i] = val
	}

	return s
}
