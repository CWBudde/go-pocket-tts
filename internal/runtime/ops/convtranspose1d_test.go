package ops

import (
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

func TestConvTranspose1D(t *testing.T) {
	input := mustTensorT(t, []float32{1, 2, 3}, []int64{1, 1, 3})
	kernel := mustTensorT(t, []float32{1, 1}, []int64{1, 1, 2})

	out, err := ConvTranspose1D(input, kernel, nil, 1, 0, 0, 1, 1)
	if err != nil {
		t.Fatalf("convtranspose1d: %v", err)
	}

	want := []float32{1, 3, 5, 3}
	if got := out.Data(); !equalApprox(got, want, 0) {
		t.Fatalf("convtranspose1d = %v, want %v", got, want)
	}
}

func TestConvTranspose1DParallel(t *testing.T) {
	SetConvWorkers(4)
	defer SetConvWorkers(1)

	input := mustTensorT(t, seqDataT(1*16*32), []int64{1, 16, 32})
	kernel := mustTensorT(t, seqDataT(16*8*5), []int64{16, 8, 5})
	bias := mustTensorT(t, seqDataT(8), []int64{8})

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

func TestRepackConvTransposeKernel(t *testing.T) {
	kernel := mustTensorT(t, []float32{
		1, 2, // ic0, oc0
		3, 4, // ic0, oc1
		5, 6, // ic0, oc2
		7, 8, // ic1, oc0
		9, 10, // ic1, oc1
		11, 12, // ic1, oc2
	}, []int64{2, 3, 2})

	got := RepackConvTransposeKernel(kernel)
	want := []float32{
		1, 7, // kx0, oc0, ic0..1
		3, 9, // kx0, oc1, ic0..1
		5, 11, // kx0, oc2, ic0..1
		2, 8, // kx1, oc0, ic0..1
		4, 10, // kx1, oc1, ic0..1
		6, 12, // kx1, oc2, ic0..1
	}
	if !equalApprox(got, want, 0) {
		t.Fatalf("RepackConvTransposeKernel() = %v, want %v", got, want)
	}
}

func TestConvTranspose1DPrePacked(t *testing.T) {
	input := mustTensorT(t, []float32{
		1, 2, 3,
		4, 5, 6,
	}, []int64{1, 2, 3})
	kernel := mustTensorT(t, []float32{
		1, 2, // ic0, oc0
		3, 4, // ic0, oc1
		5, 6, // ic1, oc0
		7, 8, // ic1, oc1
	}, []int64{2, 2, 2})
	bias := mustTensorT(t, []float32{0.5, -0.5}, []int64{2})

	if _, err := ConvTranspose1DPrePacked(input, kernel, bias, nil, 1, 0, 0, 1, 2); err == nil || !strings.Contains(err.Error(), "requires groups=1") {
		t.Fatalf("ConvTranspose1DPrePacked(groups=2) err = %v, want groups error", err)
	}

	want, err := ConvTranspose1D(input, kernel, bias, 1, 0, 0, 1, 1)
	if err != nil {
		t.Fatalf("ConvTranspose1D: %v", err)
	}

	gotNilPacked, err := ConvTranspose1DPrePacked(input, kernel, bias, nil, 1, 0, 0, 1, 1)
	if err != nil {
		t.Fatalf("ConvTranspose1DPrePacked(nil packed): %v", err)
	}
	if !equalApprox(gotNilPacked.Data(), want.Data(), 1e-5) {
		t.Fatalf("ConvTranspose1DPrePacked(nil packed) = %v, want %v", gotNilPacked.Data(), want.Data())
	}

	if _, err := ConvTranspose1DPrePacked(input, kernel, bias, make([]float32, 3), 1, 0, 0, 1, 1); err == nil || !strings.Contains(err.Error(), "length mismatch") {
		t.Fatalf("ConvTranspose1DPrePacked(length mismatch) err = %v, want mismatch error", err)
	}

	packed := RepackConvTransposeKernel(kernel)
	gotPacked, err := ConvTranspose1DPrePacked(input, kernel, bias, packed, 1, 0, 0, 1, 1)
	if err != nil {
		t.Fatalf("ConvTranspose1DPrePacked(packed): %v", err)
	}
	if !equalApprox(gotPacked.Data(), want.Data(), 1e-5) {
		t.Fatalf("ConvTranspose1DPrePacked(packed) = %v, want %v", gotPacked.Data(), want.Data())
	}
}

func TestConvTranspose1DGroupedPathWithBias(t *testing.T) {
	input := mustTensorT(t, []float32{
		1, 2, // ic0
		3, 4, // ic1
		5, 6, // ic2
		7, 8, // ic3
	}, []int64{1, 4, 2})
	kernel := mustTensorT(t, []float32{
		1,    // ic0 -> group0
		10,   // ic1 -> group0
		100,  // ic2 -> group1
		1000, // ic3 -> group1
	}, []int64{4, 1, 1})
	bias := mustTensorT(t, []float32{1, 2}, []int64{2})

	out, err := ConvTranspose1D(input, kernel, bias, 1, 0, 0, 1, 2)
	if err != nil {
		t.Fatalf("ConvTranspose1D(groups=2): %v", err)
	}

	want := []float32{
		32, 43, // oc0
		7502, 8602, // oc1
	}
	if !equalApprox(out.Data(), want, 0) {
		t.Fatalf("ConvTranspose1D(groups=2) = %v, want %v", out.Data(), want)
	}
}

func TestConvTranspose1DDepthwisePath(t *testing.T) {
	input := mustTensorT(t, []float32{
		1, 2, 3, // ic0
		4, 0, 6, // ic1
	}, []int64{1, 2, 3})
	kernel := mustTensorT(t, []float32{
		1, 1, // ic0
		2, 0, // ic1
	}, []int64{2, 1, 2})
	bias := mustTensorT(t, []float32{0.5, -0.5}, []int64{2})

	out, err := ConvTranspose1D(input, kernel, bias, 1, 0, 0, 1, 2)
	if err != nil {
		t.Fatalf("ConvTranspose1D(depthwise): %v", err)
	}

	want := []float32{
		1.5, 3.5, 5.5, 3.5, // oc0
		7.5, -0.5, 11.5, -0.5, // oc1
	}
	if !equalApprox(out.Data(), want, 0) {
		t.Fatalf("ConvTranspose1D(depthwise) = %v, want %v", out.Data(), want)
	}
}

func TestConvTranspose1DErrors(t *testing.T) {
	validInput := mustTensorT(t, []float32{1, 2, 3}, []int64{1, 1, 3})
	validKernel := mustTensorT(t, []float32{1, 1}, []int64{1, 1, 2})

	tests := []struct {
		name          string
		input         *tensor.Tensor
		kernel        *tensor.Tensor
		bias          *tensor.Tensor
		stride        int64
		padding       int64
		outputPadding int64
		dil           int64
		groups        int64
		wantErr       string
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
			name:          "invalid output padding",
			input:         validInput,
			kernel:        validKernel,
			stride:        1,
			outputPadding: 1,
			dil:           1,
			groups:        1,
			wantErr:       "output_padding",
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
			name:    "kernel in channels mismatch",
			input:   mustTensorT(t, make([]float32, 6), []int64{1, 2, 3}),
			kernel:  mustTensorT(t, make([]float32, 3), []int64{1, 1, 3}),
			stride:  1,
			dil:     1,
			groups:  1,
			wantErr: "kernel in_channels mismatch",
		},
		{
			name:    "in channels not divisible by groups",
			input:   mustTensorT(t, make([]float32, 6), []int64{1, 2, 3}),
			kernel:  mustTensorT(t, make([]float32, 6), []int64{2, 1, 3}),
			stride:  1,
			dil:     1,
			groups:  3,
			wantErr: "must be divisible by groups",
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
			padding: 100,
			dil:     1,
			groups:  1,
			wantErr: "non-positive output length",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ConvTranspose1D(tc.input, tc.kernel, tc.bias, tc.stride, tc.padding, tc.outputPadding, tc.dil, tc.groups)
			assertErrContains(t, err, tc.wantErr)
		})
	}
}
