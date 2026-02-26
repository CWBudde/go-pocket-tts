package ops

import (
	"math"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

func TestCausalMask(t *testing.T) {
	s, _ := tensor.New([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, []int64{1, 3, 3})
	out, err := CausalMask(s, 0)
	if err != nil {
		t.Fatalf("causal mask: %v", err)
	}
	got := out.Data()
	if !math.IsInf(float64(got[1]), -1) || !math.IsInf(float64(got[2]), -1) || !math.IsInf(float64(got[5]), -1) {
		t.Fatalf("causal mask did not set upper triangle to -Inf: %v", got)
	}
	if got[0] != 1 || got[3] != 4 || got[4] != 5 || got[6] != 7 || got[7] != 8 || got[8] != 9 {
		t.Fatalf("causal mask changed non-masked values: %v", got)
	}
}

func TestRoPE(t *testing.T) {
	x, _ := tensor.New([]float32{1, 0, 0, 1}, []int64{1, 2, 2})
	cos, _ := tensor.New([]float32{0, 0}, []int64{2, 1})
	sin, _ := tensor.New([]float32{1, 1}, []int64{2, 1})
	out, err := RoPE(x, cos, sin, 0)
	if err != nil {
		t.Fatalf("rope: %v", err)
	}
	want := []float32{0, 1, -1, 0}
	if got := out.Data(); !equalApprox(got, want, 1e-6) {
		t.Fatalf("rope = %v, want %v", got, want)
	}
}

func TestAttentionCausal(t *testing.T) {
	q, _ := tensor.New([]float32{1, 1}, []int64{1, 1, 2, 1})
	k, _ := tensor.New([]float32{0, 10}, []int64{1, 1, 2, 1})
	v, _ := tensor.New([]float32{1, 5}, []int64{1, 1, 2, 1})

	out, err := Attention(q, k, v, true, 0)
	if err != nil {
		t.Fatalf("attention: %v", err)
	}
	got := out.Data()
	if math.Abs(float64(got[0]-1)) > 1e-4 {
		t.Fatalf("query0 output = %f, want near 1.0 (future token masked)", got[0])
	}
	if got[1] < 4.0 {
		t.Fatalf("query1 output = %f, want > 4.0", got[1])
	}
}

func TestMLP(t *testing.T) {
	x, _ := tensor.New([]float32{1, -1}, []int64{1, 2})
	w1, _ := tensor.New([]float32{1, 0, 0, 1}, []int64{2, 2})
	w2, _ := tensor.New([]float32{1, 1}, []int64{1, 2})
	out, err := MLP(x, w1, nil, w2, nil)
	if err != nil {
		t.Fatalf("mlp: %v", err)
	}
	if got := out.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 1 {
		t.Fatalf("shape = %v, want [1 1]", got)
	}
	if got := out.Data()[0]; math.Abs(float64(got-0.4621172)) > 1e-4 {
		t.Fatalf("mlp output = %f, want ~0.4621172", got)
	}
}

func TestConv1D(t *testing.T) {
	input, _ := tensor.New([]float32{1, 2, 3, 4}, []int64{1, 1, 4})
	kernel, _ := tensor.New([]float32{1, 1}, []int64{1, 1, 2})
	out, err := Conv1D(input, kernel, nil, 1, 0, 1, 1)
	if err != nil {
		t.Fatalf("conv1d: %v", err)
	}
	want := []float32{3, 5, 7}
	if got := out.Data(); !equalApprox(got, want, 0) {
		t.Fatalf("conv1d = %v, want %v", got, want)
	}
}

func TestConvTranspose1D(t *testing.T) {
	input, _ := tensor.New([]float32{1, 2, 3}, []int64{1, 1, 3})
	kernel, _ := tensor.New([]float32{1, 1}, []int64{1, 1, 2})
	out, err := ConvTranspose1D(input, kernel, nil, 1, 0, 0, 1, 1)
	if err != nil {
		t.Fatalf("convtranspose1d: %v", err)
	}
	want := []float32{1, 3, 5, 3}
	if got := out.Data(); !equalApprox(got, want, 0) {
		t.Fatalf("convtranspose1d = %v, want %v", got, want)
	}
}

func TestConv1DParallel(t *testing.T) {
	SetConvWorkers(4)
	defer SetConvWorkers(1)

	// Larger tensor so there is real work to split across goroutines.
	input, _ := tensor.New(seqDataT(1*16*64), []int64{1, 16, 64})
	kernel, _ := tensor.New(seqDataT(32*16*3), []int64{32, 16, 3})
	bias, _ := tensor.New(seqDataT(32), []int64{32})

	// Compute with workers=4.
	got, err := Conv1D(input, kernel, bias, 1, 1, 1, 1)
	if err != nil {
		t.Fatalf("conv1d parallel: %v", err)
	}

	// Compute sequentially for reference.
	SetConvWorkers(1)
	want, err := Conv1D(input, kernel, bias, 1, 1, 1, 1)
	if err != nil {
		t.Fatalf("conv1d sequential: %v", err)
	}
	if !equalApprox(got.Data(), want.Data(), 1e-4) {
		t.Fatalf("parallel conv1d differs from sequential")
	}
}

func TestConvTranspose1DParallel(t *testing.T) {
	SetConvWorkers(4)
	defer SetConvWorkers(1)

	input, _ := tensor.New(seqDataT(1*16*32), []int64{1, 16, 32})
	kernel, _ := tensor.New(seqDataT(16*8*5), []int64{16, 8, 5})
	bias, _ := tensor.New(seqDataT(8), []int64{8})

	got, err := ConvTranspose1D(input, kernel, bias, 2, 0, 0, 1, 1)
	if err != nil {
		t.Fatalf("convtranspose1d parallel: %v", err)
	}

	SetConvWorkers(1)
	want, err := ConvTranspose1D(input, kernel, bias, 2, 0, 0, 1, 1)
	if err != nil {
		t.Fatalf("convtranspose1d sequential: %v", err)
	}
	if !equalApprox(got.Data(), want.Data(), 1e-4) {
		t.Fatalf("parallel convtranspose1d differs from sequential")
	}
}

func seqDataT(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i%17)-8) / 17
	}
	return out
}

func TestKernelTolerance(t *testing.T) {
	keys := []string{"matmul", "linear", "softmax", "layer_norm", "causal_mask", "rope", "attention", "mlp", "conv1d", "convtranspose1d"}
	for _, key := range keys {
		tol, err := KernelTolerance(key)
		if err != nil {
			t.Fatalf("KernelTolerance(%q): %v", key, err)
		}
		if key == "causal_mask" {
			if tol.Abs != 0 || tol.Rel != 0 {
				t.Fatalf("causal_mask tolerance = %+v, want {0,0}", tol)
			}
			continue
		}
		if tol.Abs <= 0 || tol.Rel <= 0 {
			t.Fatalf("KernelTolerance(%q) = %+v, want positive abs/rel", key, tol)
		}
	}
	if _, err := KernelTolerance("missing-kernel"); err == nil || !strings.Contains(err.Error(), "no tolerance configured") {
		t.Fatalf("KernelTolerance(missing-kernel) err = %v, want missing tolerance error", err)
	}
}

func equalApprox(got, want []float32, tol float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		delta := math.Abs(float64(got[i] - want[i]))
		if delta > tol {
			return false
		}
	}
	return true
}
