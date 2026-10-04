//go:build arm64

package tensor

// gemmPanelFloats bounds the b rows one pass over a reuses (128 KiB), so a
// large b (Linear weights) stays in cache while every block of a rows runs
// over it.
const gemmPanelFloats = 1 << 15

// matMulTransB runs the NEON kernels over the multiple-of-4 part of k and
// adds the k tail in Go: 4×4 blocks for groups of four a rows, 1×4 blocks
// for the leftover rows (a batch-1 Linear is all 1×4), and DotProduct for
// the leftover b rows.
func matMulTransB(c []float32, ldc int, a, b []float32, m, n, k int) {
	k4 := k &^ 3
	if k4 == 0 {
		matMulTransBGeneric(c, ldc, a, b, m, n, k)
		return
	}

	panel := max(4, (gemmPanelFloats/k)&^3)

	for j0 := 0; j0 < n; j0 += panel {
		j1 := min(n, j0+panel)
		j4 := j0 + (j1-j0)&^3

		i := 0
		for ; i+4 <= m; i += 4 {
			for j := j0; j < j4; j += 4 {
				var blk [16]float32

				gemm4x4NEON(&a[i*k], &b[j*k], k4, k, &blk)

				if k4 == k {
					for r := range 4 {
						copy(c[(i+r)*ldc+j:(i+r)*ldc+j+4], blk[r*4:r*4+4])
					}

					continue
				}

				for r := range 4 {
					aRow := a[(i+r)*k : (i+r+1)*k]
					cRow := c[(i+r)*ldc+j : (i+r)*ldc+j+4]

					for s := range cRow {
						cRow[s] = blk[r*4+s] + dotTail(aRow, b[(j+s)*k:(j+s+1)*k], k4)
					}
				}
			}
		}

		for ; i < m; i++ {
			aRow := a[i*k : (i+1)*k]

			for j := j0; j < j4; j += 4 {
				var blk [4]float32

				gemm1x4NEON(&aRow[0], &b[j*k], k4, k, &blk)

				cRow := c[i*ldc+j : i*ldc+j+4]
				if k4 == k {
					copy(cRow, blk[:])
					continue
				}

				for s := range cRow {
					cRow[s] = blk[s] + dotTail(aRow, b[(j+s)*k:(j+s+1)*k], k4)
				}
			}
		}

		for i := range m {
			aRow := a[i*k : (i+1)*k]
			for j := j4; j < j1; j++ {
				c[i*ldc+j] = dotF32(aRow, b[j*k:(j+1)*k])
			}
		}
	}
}

// dotTail sums a[p]*b[p] for p >= from: the k%4 elements the kernels skip.
func dotTail(a, b []float32, from int) float32 {
	var s float32
	for p := from; p < len(a); p++ {
		s += a[p] * b[p]
	}

	return s
}

// gemm4x4NEON sets out[r*4+s] to the dot product over the first k4 floats of
// a row r and b row s (r, s < 4); rows are stride floats apart. k4 must be a
// positive multiple of 4.
//
//go:noescape
func gemm4x4NEON(a, b *float32, k4, stride int, out *[16]float32)

// gemm1x4NEON sets out[s] to the dot product over the first k4 floats of the
// single a row and b row s (s < 4); b rows are stride floats apart. k4 must
// be a positive multiple of 4.
//
//go:noescape
func gemm1x4NEON(a, b *float32, k4, stride int, out *[4]float32)
