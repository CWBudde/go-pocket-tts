package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
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

// naiveConv1D is the textbook Conv1d (groups 1) with float64 accumulation.
func naiveConv1D(in, kernel, bias []float32, batch, inCh, length, outCh, kSize, stride, leftPad, rightPad,
	dilation int64,
) []float32 {
	outLen := (length+leftPad+rightPad-dilation*(kSize-1)-1)/stride + 1
	out := make([]float32, batch*outCh*outLen)

	for b := range batch {
		for oc := range outCh {
			for ox := range outLen {
				var sum float64
				if bias != nil {
					sum = float64(bias[oc])
				}

				for ic := range inCh {
					for kx := range kSize {
						pos := ox*stride - leftPad + kx*dilation
						if pos >= 0 && pos < length {
							sum += float64(in[(b*inCh+ic)*length+pos]) * float64(kernel[(oc*inCh+ic)*kSize+kx])
						}
					}
				}

				out[(b*outCh+oc)*outLen+ox] = float32(sum)
			}
		}
	}

	return out
}

func randDataT(rng *rand.Rand, n int64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = rng.Float32()*2 - 1
	}

	return out
}

// TestConv1DLongInputMatchesNaive covers outputs much longer than one im2col
// tile (the SEANet convs run at 24 kHz): every tile, the padded edges and
// strided or dilated taps must match the naive conv.
func TestConv1DLongInputMatchesNaive(t *testing.T) {
	cases := []struct {
		name                                string
		batch, inCh, length, outCh, kSize   int64
		stride, leftPad, rightPad, dilation int64
		bias                                bool
	}{
		{"causal k7", 1, 16, 5000, 8, 7, 1, 6, 0, 1, true},
		{"stride 4 k8 both pads", 1, 8, 9001, 16, 8, 4, 4, 3, 1, true},
		{"dilation 3 batch 2", 2, 12, 4000, 6, 3, 1, 6, 6, 3, false},
		{"pad wider than one tile row", 1, 4, 3000, 3, 33, 2, 40, 40, 1, true},
	}

	for _, c := range cases {
		for _, workers := range []int{1, 4} {
			t.Run(fmt.Sprintf("%s/workers=%d", c.name, workers), func(t *testing.T) {
				SetConvWorkers(workers)
				defer SetConvWorkers(0)

				rng := rand.New(rand.NewPCG(uint64(c.length), uint64(c.kSize)))
				in := randDataT(rng, c.batch*c.inCh*c.length)
				kernel := randDataT(rng, c.outCh*c.inCh*c.kSize)

				var (
					bias  []float32
					biasT *tensor.Tensor
				)

				if c.bias {
					bias = randDataT(rng, c.outCh)
					biasT = mustTensorT(t, bias, []int64{c.outCh})
				}

				got, err := conv1DWithAsymmetricPadding(
					mustTensorT(t, in, []int64{c.batch, c.inCh, c.length}),
					mustTensorT(t, kernel, []int64{c.outCh, c.inCh, c.kSize}),
					biasT, c.stride, c.leftPad, c.rightPad, c.dilation, 1,
				)
				if err != nil {
					t.Fatalf("conv1d: %v", err)
				}

				want := naiveConv1D(in, kernel, bias, c.batch, c.inCh, c.length, c.outCh, c.kSize,
					c.stride, c.leftPad, c.rightPad, c.dilation)

				gotData := got.RawData()
				if len(gotData) != len(want) {
					t.Fatalf("len = %d, want %d", len(gotData), len(want))
				}

				for i := range want {
					if d := math.Abs(float64(gotData[i] - want[i])); d > 1e-4 {
						t.Fatalf("out[%d] = %g, want %g (|diff| %.3g)", i, gotData[i], want[i], d)
					}
				}
			})
		}
	}
}

// TestConv1DTiledMatchesFullIm2col: the tiled path computes every output from
// the same patch and kernel row as the full im2col. On arm64 an output's sum
// order depends on whether it lands in a 4×4 NEON block or the dot-product
// tail, which tile boundaries shift, so they agree to rounding, not bit for
// bit.
func TestConv1DTiledMatchesFullIm2col(t *testing.T) {
	SetConvWorkers(3)
	defer SetConvWorkers(0)

	const batch, inCh, length, outCh, kSize, stride, leftPad, dilation = 2, 6, 2500, 5, 5, 2, 4, 1

	outLen := int64((length+leftPad-dilation*(kSize-1)-1)/stride + 1)
	rng := rand.New(rand.NewPCG(7, 11))
	in := randDataT(rng, batch*inCh*length)
	kernel := randDataT(rng, outCh*inCh*kSize)
	bias := randDataT(rng, outCh)

	full := make([]float32, batch*outCh*outLen)
	conv1DIm2colFull(in, kernel, bias, batch, inCh, length, outCh, kSize, outLen, stride, leftPad, dilation, full)

	for _, tileRows := range []int{1, 7, 64, int(outLen) - 1} {
		tiled := make([]float32, len(full))
		conv1DIm2colTiled(in, kernel, bias, batch, inCh, length, outCh, kSize, outLen, stride, leftPad, dilation,
			tiled, tileRows)

		for i := range full {
			if d := math.Abs(float64(tiled[i] - full[i])); d > 1e-5 {
				t.Fatalf("tileRows %d: out[%d] = %g, full im2col %g", tileRows, i, tiled[i], full[i])
			}
		}
	}
}

// TestConv1DZeroInputChannels: an empty patch (no input channels) gives the
// bias at every position instead of dividing by the zero patch length.
func TestConv1DZeroInputChannels(t *testing.T) {
	for _, length := range []int64{4, 40000} {
		input := mustTensorT(t, []float32{}, []int64{1, 0, length})
		kernel := mustTensorT(t, []float32{}, []int64{2, 0, 3})
		bias := mustTensorT(t, []float32{0.5, -1}, []int64{2})

		out, err := Conv1D(input, kernel, bias, 1, 1, 1, 1)
		if err != nil {
			t.Fatalf("length %d: %v", length, err)
		}

		data := out.RawData()
		if int64(len(data)) != 2*length || data[0] != 0.5 || data[len(data)-1] != -1 {
			t.Fatalf("length %d: %d outputs, first %g, last %g; want %d outputs of the bias",
				length, len(data), data[0], data[len(data)-1], 2*length)
		}
	}
}

// TestConv1DShortInputMatchesNaive covers the full-im2col path (one patch
// matrix, output channels split across workers) with output channel and
// position counts that do not fill whole 4×4 blocks, down to the 1–3 output
// positions of a streaming decode step.
func TestConv1DShortInputMatchesNaive(t *testing.T) {
	cases := []struct {
		name                                string
		batch, inCh, length, outCh, kSize   int64
		stride, leftPad, rightPad, dilation int64
	}{
		{"61 positions", 2, 7, 61, 13, 3, 1, 2, 1, 2},
		{"1 position", 1, 9, 7, 10, 7, 1, 0, 0, 1},
		{"2 positions", 2, 5, 9, 10, 7, 2, 0, 1, 1},
		{"3 positions", 1, 8, 4, 11, 3, 1, 1, 0, 1},
	}

	for _, c := range cases {
		rng := rand.New(rand.NewPCG(uint64(c.length), uint64(c.outCh)))
		in := randDataT(rng, c.batch*c.inCh*c.length)
		kernel := randDataT(rng, c.outCh*c.inCh*c.kSize)
		bias := randDataT(rng, c.outCh)
		want := naiveConv1D(in, kernel, bias, c.batch, c.inCh, c.length, c.outCh, c.kSize,
			c.stride, c.leftPad, c.rightPad, c.dilation)

		for _, workers := range []int{1, 3, 4, 5} {
			t.Run(fmt.Sprintf("%s/workers=%d", c.name, workers), func(t *testing.T) {
				SetConvWorkers(workers)
				defer SetConvWorkers(0)

				got, err := conv1DWithAsymmetricPadding(
					mustTensorT(t, in, []int64{c.batch, c.inCh, c.length}),
					mustTensorT(t, kernel, []int64{c.outCh, c.inCh, c.kSize}),
					mustTensorT(t, bias, []int64{c.outCh}), c.stride, c.leftPad, c.rightPad, c.dilation, 1,
				)
				if err != nil {
					t.Fatalf("conv1d: %v", err)
				}

				gotData := got.RawData()
				if len(gotData) != len(want) {
					t.Fatalf("len = %d, want %d", len(gotData), len(want))
				}

				for i := range want {
					if d := math.Abs(float64(gotData[i] - want[i])); d > 1e-4 {
						t.Fatalf("out[%d] = %g, want %g (|diff| %.3g)", i, gotData[i], want[i], d)
					}
				}
			})
		}
	}
}
