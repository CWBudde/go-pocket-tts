package ops

import (
	"testing"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

func TestConv1D(t *testing.T) {
	input := mustTensorT(t, []float32{1, 2, 3, 4}, []int64{1, 1, 4})
	kernel := mustTensorT(t, []float32{1, 1}, []int64{1, 1, 2})

	out, err := Conv1D(input, kernel, nil, 1, 0, 1, 1)
	if err != nil {
		t.Fatalf("conv1d: %v", err)
	}

	want := []float32{3, 5, 7}
	if got := out.Data(); !equalApprox(got, want, 0) {
		t.Fatalf("conv1d = %v, want %v", got, want)
	}
}

func TestConv1DParallel(t *testing.T) {
	SetConvWorkers(4)
	defer SetConvWorkers(1)

	// Larger tensor so there is real work to split across goroutines.
	input := mustTensorT(t, seqDataT(1*16*64), []int64{1, 16, 64})
	kernel := mustTensorT(t, seqDataT(32*16*3), []int64{32, 16, 3})
	bias := mustTensorT(t, seqDataT(32), []int64{32})

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

func TestConv1DGroupedPath(t *testing.T) {
	input := mustTensorT(t, []float32{
		1, 2, 3, 4,
		10, 20, 30, 40,
	}, []int64{1, 2, 4})
	kernel := mustTensorT(t, []float32{
		1, 1, // oc0
		1, 1, // oc1
	}, []int64{2, 1, 2})

	out, err := Conv1D(input, kernel, nil, 1, 0, 1, 2)
	if err != nil {
		t.Fatalf("Conv1D(groups=2): %v", err)
	}

	want := []float32{
		3, 5, 7,
		30, 50, 70,
	}
	if !equalApprox(out.Data(), want, 0) {
		t.Fatalf("Conv1D(groups=2) = %v, want %v", out.Data(), want)
	}
}

func TestConv1DLeftPadMatchesExplicitPrepend(t *testing.T) {
	input := mustTensorT(t, []float32{
		1, 2, 3, 4,
		10, 20, 30, 40,
	}, []int64{1, 2, 4})
	kernel := mustTensorT(t, []float32{
		1, 1, 1, // oc0, ic0
		1, 1, 1, // oc0, ic1
		2, 2, 2, // oc1, ic0
		2, 2, 2, // oc1, ic1
	}, []int64{2, 2, 3})
	bias := mustTensorT(t, []float32{0.25, -0.5}, []int64{2})

	const leftPad = int64(2)
	const stride = int64(2)
	const dilation = int64(1)

	got, err := Conv1DLeftPad(input, kernel, bias, stride, leftPad, dilation, 1)
	if err != nil {
		t.Fatalf("Conv1DLeftPad: %v", err)
	}

	shape := input.Shape()

	pad, err := tensor.Zeros([]int64{shape[0], shape[1], leftPad})
	if err != nil {
		t.Fatalf("Zeros: %v", err)
	}

	padded, err := tensor.Concat([]*tensor.Tensor{pad, input}, 2)
	if err != nil {
		t.Fatalf("Concat: %v", err)
	}

	want, err := Conv1D(padded, kernel, bias, stride, 0, dilation, 1)
	if err != nil {
		t.Fatalf("Conv1D explicit prepend: %v", err)
	}

	if !equalApprox(got.Data(), want.Data(), 1e-5) {
		t.Fatalf("Conv1DLeftPad = %v, want %v", got.Data(), want.Data())
	}
}

func TestConv1DErrors(t *testing.T) {
	validInput := mustTensorT(t, []float32{1, 2, 3, 4}, []int64{1, 1, 4})
	validKernel := mustTensorT(t, []float32{1, 1}, []int64{1, 1, 2})

	tests := []struct {
		name    string
		input   *tensor.Tensor
		kernel  *tensor.Tensor
		bias    *tensor.Tensor
		stride  int64
		padding int64
		dil     int64
		groups  int64
		wantErr string
	}{
		{
			name:    "nil input",
			input:   nil,
			kernel:  validKernel,
			stride:  1,
			dil:     1,
			groups:  1,
			wantErr: "requires non-nil",
		},
		{
			name:    "invalid stride",
			input:   validInput,
			kernel:  validKernel,
			stride:  0,
			dil:     1,
			groups:  1,
			wantErr: "must be > 0",
		},
		{
			name:    "rank mismatch",
			input:   mustTensorT(t, []float32{1, 2}, []int64{1, 2}),
			kernel:  validKernel,
			stride:  1,
			dil:     1,
			groups:  1,
			wantErr: "rank 3",
		},
		{
			name:    "channels not divisible by groups",
			input:   mustTensorT(t, make([]float32, 6), []int64{1, 3, 2}),
			kernel:  mustTensorT(t, make([]float32, 6), []int64{2, 3, 1}),
			stride:  1,
			dil:     1,
			groups:  2,
			wantErr: "not divisible by groups",
		},
		{
			name:    "kernel in channels mismatch",
			input:   mustTensorT(t, make([]float32, 4), []int64{1, 2, 2}),
			kernel:  mustTensorT(t, make([]float32, 6), []int64{2, 3, 1}),
			stride:  1,
			dil:     1,
			groups:  2,
			wantErr: "kernel in_channels/groups mismatch",
		},
		{
			name:    "bias mismatch",
			input:   validInput,
			kernel:  validKernel,
			bias:    mustTensorT(t, []float32{1, 2}, []int64{2}),
			stride:  1,
			dil:     1,
			groups:  1,
			wantErr: "bias shape",
		},
		{
			name:    "non positive output length",
			input:   validInput,
			kernel:  validKernel,
			stride:  1,
			padding: -10,
			dil:     1,
			groups:  1,
			wantErr: "non-positive output length",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Conv1D(tc.input, tc.kernel, tc.bias, tc.stride, tc.padding, tc.dil, tc.groups)
			assertErrContains(t, err, tc.wantErr)
		})
	}
}
