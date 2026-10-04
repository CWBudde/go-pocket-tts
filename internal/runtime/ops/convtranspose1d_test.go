package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
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

	_, err := ConvTranspose1DPrePacked(input, kernel, bias, nil, 1, 0, 0, 1, 2)
	if err == nil || !strings.Contains(err.Error(), "requires groups=1") {
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

	_, err = ConvTranspose1DPrePacked(input, kernel, bias, make([]float32, 3), 1, 0, 0, 1, 1)
	if err == nil || !strings.Contains(err.Error(), "length mismatch") {
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

func TestConvTranspose1DRightTrimMatchesNarrow(t *testing.T) {
	input := mustTensorT(t, seqDataT(1*3*5), []int64{1, 3, 5})
	kernel := mustTensorT(t, seqDataT(3*4*4), []int64{3, 4, 4})
	bias := mustTensorT(t, seqDataT(4), []int64{4})

	const rightTrim = int64(2)

	got, err := ConvTranspose1DRightTrim(input, kernel, bias, 2, 0, 0, 1, 1, rightTrim)
	if err != nil {
		t.Fatalf("ConvTranspose1DRightTrim: %v", err)
	}

	full, err := ConvTranspose1D(input, kernel, bias, 2, 0, 0, 1, 1)
	if err != nil {
		t.Fatalf("ConvTranspose1D: %v", err)
	}

	shape := full.Shape()

	want, err := full.Narrow(2, 0, shape[2]-rightTrim)
	if err != nil {
		t.Fatalf("Narrow: %v", err)
	}

	if !equalApprox(got.Data(), want.Data(), 1e-5) {
		t.Fatalf("ConvTranspose1DRightTrim = %v, want %v", got.Data(), want.Data())
	}
}

func TestConvTranspose1DPrePackedRightTrimMatchesNarrow(t *testing.T) {
	input := mustTensorT(t, seqDataT(1*2*6), []int64{1, 2, 6})
	kernel := mustTensorT(t, seqDataT(2*3*5), []int64{2, 3, 5})
	bias := mustTensorT(t, seqDataT(3), []int64{3})
	packed := RepackConvTransposeKernel(kernel)

	const rightTrim = int64(3)

	got, err := ConvTranspose1DPrePackedRightTrim(input, kernel, bias, packed, 2, 0, 0, 1, 1, rightTrim)
	if err != nil {
		t.Fatalf("ConvTranspose1DPrePackedRightTrim: %v", err)
	}

	full, err := ConvTranspose1DPrePacked(input, kernel, bias, packed, 2, 0, 0, 1, 1)
	if err != nil {
		t.Fatalf("ConvTranspose1DPrePacked: %v", err)
	}

	shape := full.Shape()

	want, err := full.Narrow(2, 0, shape[2]-rightTrim)
	if err != nil {
		t.Fatalf("Narrow: %v", err)
	}

	if !equalApprox(got.Data(), want.Data(), 1e-5) {
		t.Fatalf("ConvTranspose1DPrePackedRightTrim = %v, want %v", got.Data(), want.Data())
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

// naiveConvTranspose1D is the textbook ConvTranspose1d (groups 1) with
// float64 accumulation; outLen already has outputPadding and the right trim
// applied.
func naiveConvTranspose1D(in, kernel, bias []float32, batch, inCh, inLen, outCh, kSize, outLen,
	stride, padding, dilation int64,
) []float32 {
	acc := make([]float64, batch*outCh*outLen)

	for b := range batch {
		for ic := range inCh {
			for ix := range inLen {
				v := float64(in[(b*inCh+ic)*inLen+ix])

				for oc := range outCh {
					for kx := range kSize {
						pos := ix*stride - padding + kx*dilation
						if pos >= 0 && pos < outLen {
							acc[(b*outCh+oc)*outLen+pos] += v * float64(kernel[(ic*outCh+oc)*kSize+kx])
						}
					}
				}
			}
		}
	}

	out := make([]float32, len(acc))
	for i, v := range acc {
		if bias != nil {
			v += float64(bias[(int64(i)/outLen)%outCh])
		}

		out[i] = float32(v)
	}

	return out
}

// TestConvTranspose1DMatchesNaive covers the GEMM-and-scatter path: strided,
// padded and dilated taps that fall off both ends, output channels that do
// not fill a 4-row block, inputs long enough to need several ix tiles or too
// short for a 4-column block, and the prepacked and right-trimmed entry
// points.
func TestConvTranspose1DMatchesNaive(t *testing.T) {
	cases := []struct {
		name                         string
		batch, inCh, inLen           int64
		outCh, kSize                 int64
		stride, padding, outPad, dil int64
		rightTrim                    int64
		bias                         bool
	}{
		{"tiny", 1, 1, 3, 1, 2, 1, 0, 0, 1, 0, false},
		{"stride 3 pad dil 2 outpad", 2, 6, 37, 7, 5, 3, 2, 1, 2, 0, true},
		{"pad wider than kernel reach", 1, 9, 20, 5, 3, 2, 7, 1, 1, 0, true},
		{"many ix tiles", 2, 6, 20000, 7, 5, 3, 2, 1, 2, 0, true},
		{"mimi upsample right trim", 1, 16, 50, 9, 16, 8, 0, 0, 1, 8, true},
		{"streaming one frame", 2, 12, 1, 6, 4, 2, 1, 1, 1, 0, true},
		{"streaming three frames", 1, 16, 3, 9, 16, 8, 3, 0, 2, 5, true},
	}

	for _, c := range cases {
		for _, workers := range []int{1, 3} {
			t.Run(fmt.Sprintf("%s/workers=%d", c.name, workers), func(t *testing.T) {
				SetConvWorkers(workers)
				defer SetConvWorkers(0)

				rng := rand.New(rand.NewPCG(uint64(c.inLen), uint64(c.outCh)))
				in := randDataT(rng, c.batch*c.inCh*c.inLen)
				kernel := randDataT(rng, c.inCh*c.outCh*c.kSize)

				var (
					bias  []float32
					biasT *tensor.Tensor
				)

				if c.bias {
					bias = randDataT(rng, c.outCh)
					biasT = mustTensorT(t, bias, []int64{c.outCh})
				}

				inT := mustTensorT(t, in, []int64{c.batch, c.inCh, c.inLen})
				kernelT := mustTensorT(t, kernel, []int64{c.inCh, c.outCh, c.kSize})

				got, err := ConvTranspose1DRightTrim(inT, kernelT, biasT, c.stride, c.padding, c.outPad, c.dil, 1, c.rightTrim)
				if err != nil {
					t.Fatalf("convtranspose1d: %v", err)
				}

				packed, err := ConvTranspose1DPrePackedRightTrim(inT, kernelT, biasT, RepackConvTransposeKernel(kernelT),
					c.stride, c.padding, c.outPad, c.dil, 1, c.rightTrim)
				if err != nil {
					t.Fatalf("convtranspose1d prepacked: %v", err)
				}

				outLen := got.Shape()[2]
				if wantLen := (c.inLen-1)*c.stride - 2*c.padding + c.dil*(c.kSize-1) + c.outPad + 1 - c.rightTrim; outLen != wantLen {
					t.Fatalf("outLen = %d, want %d", outLen, wantLen)
				}

				want := naiveConvTranspose1D(in, kernel, bias, c.batch, c.inCh, c.inLen, c.outCh, c.kSize, outLen,
					c.stride, c.padding, c.dil)

				for name, gotData := range map[string][]float32{"plain": got.RawData(), "prepacked": packed.RawData()} {
					if len(gotData) != len(want) {
						t.Fatalf("%s: len = %d, want %d", name, len(gotData), len(want))
					}

					for i := range want {
						if d := math.Abs(float64(gotData[i] - want[i])); d > 1e-4 {
							t.Fatalf("%s: out[%d] = %g, want %g (|diff| %.3g)", name, i, gotData[i], want[i], d)
						}
					}
				}
			})
		}
	}
}
