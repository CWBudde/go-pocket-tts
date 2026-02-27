package ops

import (
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

func TestRoPE(t *testing.T) {
	x := mustTensorT(t, []float32{1, 0, 0, 1}, []int64{1, 2, 2})
	cos := mustTensorT(t, []float32{0, 0}, []int64{2, 1})
	sin := mustTensorT(t, []float32{1, 1}, []int64{2, 1})

	out, err := RoPE(x, cos, sin, 0)
	if err != nil {
		t.Fatalf("rope: %v", err)
	}

	want := []float32{0, 1, -1, 0}
	if got := out.Data(); !equalApprox(got, want, 1e-6) {
		t.Fatalf("rope = %v, want %v", got, want)
	}
}

func TestRoPEErrors(t *testing.T) {
	x := mustTensorT(t, make([]float32, 4), []int64{1, 2, 2})
	cos := mustTensorT(t, make([]float32, 2), []int64{2, 1})
	sin := mustTensorT(t, make([]float32, 2), []int64{2, 1})

	cases := []struct {
		name    string
		x       *tensor.Tensor
		cos     *tensor.Tensor
		sin     *tensor.Tensor
		pos     int64
		wantErr string
	}{
		{
			name:    "nil x",
			x:       nil,
			cos:     cos,
			sin:     sin,
			wantErr: "non-nil",
		},
		{
			name:    "negative pos",
			x:       x,
			cos:     cos,
			sin:     sin,
			pos:     -1,
			wantErr: "must be >= 0",
		},
		{
			name:    "rank too small",
			x:       mustTensorT(t, []float32{1, 2}, []int64{2}),
			cos:     cos,
			sin:     sin,
			wantErr: "rank >= 2",
		},
		{
			name:    "odd dim",
			x:       mustTensorT(t, make([]float32, 3), []int64{1, 1, 3}),
			cos:     mustTensorT(t, []float32{1}, []int64{1, 1}),
			sin:     mustTensorT(t, []float32{1}, []int64{1, 1}),
			wantErr: "must be even",
		},
		{
			name:    "cos sin rank mismatch",
			x:       x,
			cos:     mustTensorT(t, []float32{1, 2}, []int64{2}),
			sin:     sin,
			wantErr: "rank 2",
		},
		{
			name:    "sequence too small",
			x:       x,
			cos:     mustTensorT(t, []float32{1}, []int64{1, 1}),
			sin:     mustTensorT(t, []float32{1}, []int64{1, 1}),
			wantErr: "sequence length too small",
		},
		{
			name:    "width mismatch",
			x:       x,
			cos:     mustTensorT(t, []float32{1, 2, 3, 4}, []int64{2, 2}),
			sin:     mustTensorT(t, []float32{1, 2, 3, 4}, []int64{2, 2}),
			wantErr: "width mismatch",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := RoPE(tc.x, tc.cos, tc.sin, tc.pos)
			assertErrContains(t, err, tc.wantErr)
		})
	}
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

	_, err := KernelTolerance("missing-kernel")
	if err == nil || !strings.Contains(err.Error(), "no tolerance configured") {
		t.Fatalf("KernelTolerance(missing-kernel) err = %v, want missing tolerance error", err)
	}
}
