package native

import (
	"math"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
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
