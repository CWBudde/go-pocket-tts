package native

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
)

func TestLinearForwardIntoMatchesTensorLinear(t *testing.T) {
	w, err := tensor.New([]float32{
		1.0, -2.0,
		0.5, 0.25,
		-1.5, 3.0,
	}, []int64{3, 2})
	if err != nil {
		t.Fatalf("weight: %v", err)
	}

	bias, err := tensor.New([]float32{0.1, -0.2, 0.3}, []int64{3})
	if err != nil {
		t.Fatalf("bias: %v", err)
	}

	x, err := tensor.New([]float32{
		1, 2,
		3, 4,
		5, 6,
		-1, -2,
	}, []int64{2, 2, 2})
	if err != nil {
		t.Fatalf("x: %v", err)
	}

	l := &Linear{Weight: w, Bias: bias, inDim: 2, outDim: 3}

	out, err := tensor.Zeros([]int64{2, 2, 3})
	if err != nil {
		t.Fatalf("out zeros: %v", err)
	}

	err = l.ForwardInto(x, out)
	if err != nil {
		t.Fatalf("ForwardInto: %v", err)
	}

	want, err := tensor.Linear(x, w, bias)
	if err != nil {
		t.Fatalf("tensor.Linear: %v", err)
	}

	assertCloseSlice(t, out.RawData(), want.RawData(), 1e-5)
}

func TestLayerNormForwardIntoMatchesTensorLayerNorm(t *testing.T) {
	x, err := tensor.New([]float32{
		1.2, -0.4, 0.7, 2.1,
		0.9, 0.3, -1.0, 1.5,
	}, []int64{2, 4})
	if err != nil {
		t.Fatalf("x: %v", err)
	}

	w, err := tensor.New([]float32{1.1, 0.9, 1.2, 0.8}, []int64{4})
	if err != nil {
		t.Fatalf("weight: %v", err)
	}

	b, err := tensor.New([]float32{0.05, -0.03, 0.02, 0.01}, []int64{4})
	if err != nil {
		t.Fatalf("bias: %v", err)
	}

	ln := &LayerNorm{Weight: w, Bias: b, Eps: 1e-5, dim: 4}

	out, err := tensor.Zeros([]int64{2, 4})
	if err != nil {
		t.Fatalf("out zeros: %v", err)
	}

	err = ln.ForwardInto(x, out)
	if err != nil {
		t.Fatalf("ForwardInto: %v", err)
	}

	want, err := tensor.LayerNorm(x, w, b, 1e-5)
	if err != nil {
		t.Fatalf("tensor.LayerNorm: %v", err)
	}

	assertCloseSlice(t, out.RawData(), want.RawData(), 1e-5)
}

func TestLinearForwardIntoRejectsWrongOutShape(t *testing.T) {
	w, err := tensor.New([]float32{
		1, 2,
		3, 4,
	}, []int64{2, 2})
	if err != nil {
		t.Fatalf("weight: %v", err)
	}

	x, err := tensor.New([]float32{1, 2, 3, 4}, []int64{2, 2})
	if err != nil {
		t.Fatalf("x: %v", err)
	}

	l := &Linear{Weight: w, inDim: 2, outDim: 2}

	out, err := tensor.Zeros([]int64{2, 3})
	if err != nil {
		t.Fatalf("out zeros: %v", err)
	}

	err = l.ForwardInto(x, out)
	if err == nil || !strings.Contains(err.Error(), "out last dim mismatch") {
		t.Fatalf("expected out shape mismatch error, got: %v", err)
	}
}

func assertCloseSlice(t *testing.T, got, want []float32, tol float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("length mismatch got=%d want=%d", len(got), len(want))
	}

	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("value mismatch at %d: got=%f want=%f (tol=%g)", i, got[i], want[i], tol)
		}
	}
}

// TestLinearForwardMatchesFloat64Reference covers every forwardIntoTrusted
// partition: sequential, split by batch rows and split by output columns
// (batch 1 included), with dims that leave kernel tails (not multiples
// of 4) and worker counts whose chunks do not align to the 4-wide blocks.
func TestLinearForwardMatchesFloat64Reference(t *testing.T) {
	prevWorkers := tensor.Workers()
	defer tensor.SetWorkers(prevWorkers)

	rng := rand.New(rand.NewPCG(7, 11))

	dims := []struct{ in, out int }{
		{13, 7},    // below the parallel threshold: always sequential
		{515, 519}, // above it for every batch size
	}

	for _, d := range dims {
		w := randTensor(t, rng, d.out, d.in)
		b := randTensor(t, rng, d.out)

		// 33 rows give 4 workers ≥ 8 rows each, the batch-split path.
		for _, batch := range []int{1, 3, 5, 9, 33} {
			x := randTensor(t, rng, batch, 1, d.in)

			for _, withBias := range []bool{false, true} {
				l := &Linear{Weight: w, inDim: int64(d.in), outDim: int64(d.out)}
				if withBias {
					l.Bias = b
				}

				want, tol := linearFloat64Reference(x.RawData(), w.RawData(), l.Bias, batch, d.out, d.in)

				for _, workers := range []int{1, 4, 7, 16} {
					name := fmt.Sprintf("in=%d/out=%d/batch=%d/bias=%t/workers=%d", d.in, d.out, batch, withBias, workers)

					t.Run(name, func(t *testing.T) {
						tensor.SetWorkers(workers)

						got, err := l.Forward(x)
						if err != nil {
							t.Fatalf("Forward: %v", err)
						}

						assertCloseTol(t, "Forward", got.RawData(), want, tol)

						// ForwardInto must write every output, whatever was there.
						out, err := tensor.Full([]int64{int64(batch), 1, int64(d.out)}, float32(math.NaN()))
						if err != nil {
							t.Fatalf("out: %v", err)
						}

						err = l.ForwardInto(x, out)
						if err != nil {
							t.Fatalf("ForwardInto: %v", err)
						}

						assertCloseTol(t, "ForwardInto", out.RawData(), want, tol)
					})
				}
			}
		}
	}
}

// linearFloat64Reference returns x·wᵀ+bias in float64 and a per-output
// tolerance scaled by sum|x·w|, the bound on float32 rounding error growth.
func linearFloat64Reference(x, w []float32, bias *tensor.Tensor, batch, outDim, inDim int) ([]float64, []float64) {
	want := make([]float64, batch*outDim)
	tol := make([]float64, batch*outDim)

	for bi := range batch {
		for o := range outDim {
			var sum, mag float64

			for i := range inDim {
				p := float64(x[bi*inDim+i]) * float64(w[o*inDim+i])
				sum += p
				mag += math.Abs(p)
			}

			if bias != nil {
				bv := float64(bias.RawData()[o])
				sum += bv
				mag += math.Abs(bv)
			}

			want[bi*outDim+o] = sum
			tol[bi*outDim+o] = 1e-5*mag + 1e-6
		}
	}

	return want, tol
}

func randTensor(tb testing.TB, rng *rand.Rand, shape ...int) *tensor.Tensor {
	tb.Helper()

	n := 1
	shape64 := make([]int64, len(shape))

	for i, s := range shape {
		n *= s
		shape64[i] = int64(s)
	}

	data := make([]float32, n)
	for i := range data {
		data[i] = rng.Float32()*2 - 1
	}

	out, err := tensor.New(data, shape64)
	if err != nil {
		tb.Fatalf("tensor.New: %v", err)
	}

	return out
}

func assertCloseTol(t *testing.T, label string, got []float32, want, tol []float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch got=%d want=%d", label, len(got), len(want))
	}

	for i := range got {
		if diff := math.Abs(float64(got[i]) - want[i]); !(diff <= tol[i]) {
			t.Fatalf("%s: value mismatch at %d: got=%g want=%g (tol=%g)", label, i, got[i], want[i], tol[i])
		}
	}
}

// BenchmarkNativeLinear times forwardIntoTrusted at model-like widths across
// batch sizes and worker counts; it guides the parallel partitioning choice.
func BenchmarkNativeLinear(b *testing.B) {
	prevWorkers := tensor.Workers()
	defer tensor.SetWorkers(prevWorkers)

	rng := rand.New(rand.NewPCG(1, 2))

	for _, d := range []struct{ in, out int }{{1024, 1024}, {512, 2048}} {
		l := &Linear{Weight: randTensor(b, rng, d.out, d.in), Bias: randTensor(b, rng, d.out), inDim: int64(d.in), outDim: int64(d.out)}

		for _, batch := range []int{1, 2, 4, 8, 16, 64} {
			x := randTensor(b, rng, batch, d.in)

			out, err := tensor.Zeros([]int64{int64(batch), int64(d.out)})
			if err != nil {
				b.Fatalf("out: %v", err)
			}

			for _, workers := range []int{1, 8} {
				b.Run(fmt.Sprintf("in=%d/out=%d/b=%d/w=%d", d.in, d.out, batch, workers), func(b *testing.B) {
					tensor.SetWorkers(workers)

					for range b.N {
						err := l.forwardIntoTrusted(x, out)
						if err != nil {
							b.Fatalf("forward: %v", err)
						}
					}
				})
			}
		}
	}
}
