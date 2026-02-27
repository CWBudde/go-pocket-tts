package tensor

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

func TestSoftmax(t *testing.T) {
	x, _ := New([]float32{1, 2, 3}, []int64{3})

	out, err := Softmax(x, 0)
	if err != nil {
		t.Fatalf("softmax: %v", err)
	}

	want := []float32{0.09003057, 0.24472848, 0.66524094}
	if got := out.Data(); !equalF32(got, want, 1e-5) {
		t.Fatalf("softmax = %v, want ~%v", got, want)
	}
}

func TestLayerNorm(t *testing.T) {
	x, _ := New([]float32{1, 2, 3, 4}, []int64{1, 4})
	w, _ := New([]float32{1, 1, 1, 1}, []int64{4})
	b, _ := New([]float32{0, 0, 0, 0}, []int64{4})

	out, err := LayerNorm(x, w, b, 1e-5)
	if err != nil {
		t.Fatalf("layernorm: %v", err)
	}

	got := out.Data()

	want := []float32{-1.3416355, -0.44721183, 0.44721183, 1.3416355}
	if !equalF32(got, want, 1e-4) {
		t.Fatalf("layernorm = %v, want ~%v", got, want)
	}
}

func TestMatMul2D(t *testing.T) {
	a, _ := New([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	b, _ := New([]float32{7, 8, 9, 10, 11, 12}, []int64{3, 2})

	out, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("matmul: %v", err)
	}

	if got := out.Shape(); !equalI64(got, []int64{2, 2}) {
		t.Fatalf("shape = %v, want [2 2]", got)
	}

	want := []float32{58, 64, 139, 154}
	if got := out.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func TestMatMulBatchBroadcast(t *testing.T) {
	a, _ := New([]float32{
		1, 2, 3, 4, // batch 0, 2x2
		5, 6, 7, 8, // batch 1, 2x2
	}, []int64{2, 2, 2})
	b, _ := New([]float32{1, 0, 0, 1}, []int64{1, 2, 2})

	out, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("matmul: %v", err)
	}

	if got := out.Shape(); !equalI64(got, []int64{2, 2, 2}) {
		t.Fatalf("shape = %v, want [2 2 2]", got)
	}

	if got := out.Data(); !equalF32(got, a.Data(), 0) {
		t.Fatalf("data = %v, want %v", got, a.Data())
	}
}

func TestLinear(t *testing.T) {
	x, _ := New([]float32{1, 2, 3, 4}, []int64{2, 2})
	w, _ := New([]float32{1, 0, 0, 1, 1, 1}, []int64{3, 2})
	b, _ := New([]float32{0.5, -0.5, 1}, []int64{3})

	out, err := Linear(x, w, b)
	if err != nil {
		t.Fatalf("linear: %v", err)
	}

	if got := out.Shape(); !equalI64(got, []int64{2, 3}) {
		t.Fatalf("shape = %v, want [2 3]", got)
	}

	want := []float32{1.5, 1.5, 4, 3.5, 3.5, 8}
	if got := out.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

func randTensor(rng *rand.Rand, shape ...int64) *Tensor {
	t, _ := Zeros(shape)
	for i := range t.data {
		t.data[i] = rng.Float32()*2 - 1
	}

	return t
}

func BenchmarkLinear(b *testing.B) {
	for _, tc := range []struct {
		batch, in, out int64
	}{
		{1, 512, 1024},
		{4, 512, 1024},
		{1, 1024, 4096},
	} {
		rng := rand.New(rand.NewPCG(42, 0))
		x := randTensor(rng, tc.batch, tc.in)
		w := randTensor(rng, tc.out, tc.in)
		bias := randTensor(rng, tc.out)

		name := fmt.Sprintf("b=%d_in=%d_out=%d", tc.batch, tc.in, tc.out)
		b.Run(name, func(b *testing.B) {
			for range b.N {
				_, _ = Linear(x, w, bias)
			}
		})
	}
}

func BenchmarkMatMul(b *testing.B) {
	for _, tc := range []struct {
		m, k, n int64
	}{
		{64, 512, 64},
		{128, 512, 128},
	} {
		rng := rand.New(rand.NewPCG(42, 0))
		a := randTensor(rng, tc.m, tc.k)
		bm := randTensor(rng, tc.k, tc.n)

		name := fmt.Sprintf("m=%d_k=%d_n=%d", tc.m, tc.k, tc.n)
		b.Run(name, func(b *testing.B) {
			for range b.N {
				_, _ = MatMul(a, bm)
			}
		})
	}
}

func BenchmarkSoftmax(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	x := randTensor(rng, 4, 512)

	b.ResetTimer()

	for range b.N {
		_, _ = Softmax(x, -1)
	}
}

func BenchmarkLayerNorm(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	x := randTensor(rng, 4, 512)
	w := randTensor(rng, 512)
	bias := randTensor(rng, 512)

	b.ResetTimer()

	for range b.N {
		_, _ = LayerNorm(x, w, bias, 1e-5)
	}
}
