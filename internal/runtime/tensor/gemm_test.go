package tensor

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"
)

const gemmSentinel = float32(-12345)

func randGemmData(rng *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = rng.Float32()*2 - 1
	}

	return out
}

// gemmDotLoop is the arithmetic every MatMulTransB caller used before: one
// DotProduct per output.
func gemmDotLoop(c []float32, ldc int, a, b []float32, m, n, k int) {
	for i := range m {
		for j := range n {
			c[i*ldc+j] = DotProduct(a[i*k:(i+1)*k], b[j*k:(j+1)*k])
		}
	}
}

// TestMatMulTransBMatchesReference covers the 4×4 and 1×4 blocks, leftover
// rows and columns, every k tail and an ldc wider than n whose gap must stay
// untouched.
func TestMatMulTransBMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))

	for _, k := range []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 64, 70, 576} {
		for m := 1; m <= 9; m++ {
			for n := 1; n <= 9; n++ {
				a := randGemmData(rng, m*k)
				b := randGemmData(rng, n*k)
				ldc := n + 3

				c := make([]float32, (m-1)*ldc+n+2)
				for i := range c {
					c[i] = gemmSentinel
				}

				MatMulTransB(c, ldc, a, b, m, n, k)

				for i := range m {
					for j := range n {
						var want, mag float64

						for p := range k {
							prod := float64(a[i*k+p]) * float64(b[j*k+p])
							want += prod
							mag += math.Abs(prod)
						}

						got := float64(c[i*ldc+j])
						if math.Abs(got-want) > 1e-6*mag+1e-6 {
							t.Fatalf("m=%d n=%d k=%d: c[%d][%d] = %v; want %v", m, n, k, i, j, got, want)
						}
					}

					// The gap after each row and the slack after the last
					// one are not part of the output.
					end := min((i+1)*ldc, len(c))
					for g := i*ldc + n; g < end; g++ {
						if c[g] != gemmSentinel {
							t.Fatalf("m=%d n=%d k=%d: c[%d] = %v outside the output was overwritten", m, n, k, g, c[g])
						}
					}
				}
			}
		}
	}
}

// TestMatMulTransBPanels covers b split into several cache panels (k=1024
// gives 32-row panels), the leftover b rows of the last panel, ldc == n and
// operands longer than m*k and n*k.
func TestMatMulTransBPanels(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	k := 1024

	for _, m := range []int{1, 4, 5, 9} {
		for _, n := range []int{32, 33, 35, 36, 70, 99} {
			a := randGemmData(rng, (m+1)*k)
			b := randGemmData(rng, (n+1)*k)
			c := make([]float32, m*n)

			MatMulTransB(c, n, a, b, m, n, k)

			for i := range m {
				for j := range n {
					var want, mag float64

					for p := range k {
						prod := float64(a[i*k+p]) * float64(b[j*k+p])
						want += prod
						mag += math.Abs(prod)
					}

					if got := float64(c[i*n+j]); math.Abs(got-want) > 1e-6*mag+1e-6 {
						t.Fatalf("m=%d n=%d: c[%d][%d] = %v; want %v", m, n, i, j, got, want)
					}
				}
			}
		}
	}
}

// TestMatMulTransBGenericIsDotLoop: without a SIMD kernel MatMulTransB is
// bit-identical to the per-output dot products it replaces, so amd64 and
// wasm results do not move.
func TestMatMulTransBGenericIsDotLoop(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	m, n, k := 7, 9, 70
	a := randGemmData(rng, m*k)
	b := randGemmData(rng, n*k)

	want := make([]float32, m*n)
	gemmDotLoop(want, n, a, b, m, n, k)

	got := make([]float32, m*n)
	matMulTransBGeneric(got, n, a, b, m, n, k)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("generic c[%d] = %v; dot loop %v", i, got[i], want[i])
		}
	}

	if runtime.GOARCH == "arm64" {
		return
	}

	MatMulTransB(got, n, a, b, m, n, k)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MatMulTransB c[%d] = %v; dot loop %v", i, got[i], want[i])
		}
	}
}

func TestMatMulTransBEmpty(t *testing.T) {
	c := []float32{gemmSentinel}

	MatMulTransB(c, 1, nil, nil, 0, 1, 4)
	MatMulTransB(c, 1, nil, nil, 1, 0, 4)

	if c[0] != gemmSentinel {
		t.Fatalf("empty product wrote c[0] = %v", c[0])
	}

	MatMulTransB(c, 1, nil, nil, 1, 1, 0)

	if c[0] != 0 {
		t.Fatalf("k=0: c[0] = %v; want 0", c[0])
	}
}

func TestMatMulTransBPanicsOnShortSlices(t *testing.T) {
	m, n, k := 4, 4, 8
	a := make([]float32, m*k)
	b := make([]float32, n*k)
	c := make([]float32, m*n)

	cases := map[string]func(){
		"short a":    func() { MatMulTransB(c, n, a[:m*k-1], b, m, n, k) },
		"short b":    func() { MatMulTransB(c, n, a, b[:n*k-1], m, n, k) },
		"short c":    func() { MatMulTransB(c[:m*n-1], n, a, b, m, n, k) },
		"ldc < n":    func() { MatMulTransB(c, n-1, a, b, m, n, k) },
		"negative k": func() { MatMulTransB(c, n, a, b, m, n, -1) },
	}

	for name, call := range cases {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("no panic")
				}
			}()

			call()
		})
	}
}

// BenchmarkMatMulTransB compares the kernel with the per-output dot loop on
// the shapes of the hot callers: an im2col conv tile, a batch-1 Linear
// (GEMV) and a batched Linear.
func BenchmarkMatMulTransB(b *testing.B) {
	shapes := []struct {
		name    string
		m, n, k int
	}{
		{"conv_tile_64x56x576", 64, 56, 576},
		{"gemv_1x1024x1024", 1, 1024, 1024},
		{"linear_16x1024x1024", 16, 1024, 1024},
	}

	rng := rand.New(rand.NewPCG(5, 6))

	for _, s := range shapes {
		a := randGemmData(rng, s.m*s.k)
		w := randGemmData(rng, s.n*s.k)
		c := make([]float32, s.m*s.n)

		b.Run(s.name+"/kernel", func(b *testing.B) {
			for range b.N {
				MatMulTransB(c, s.n, a, w, s.m, s.n, s.k)
			}
		})

		b.Run(s.name+"/dotloop", func(b *testing.B) {
			for range b.N {
				gemmDotLoop(c, s.n, a, w, s.m, s.n, s.k)
			}
		})
	}
}
