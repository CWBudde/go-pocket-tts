package native

import (
	"math"
	"testing"

	"github.com/example/go-pocket-tts/internal/runtime/ops"
	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

func TestDenormLatentToBCTMatchesBroadcastReference(t *testing.T) {
	t.Parallel()

	latent, err := tensor.New(
		[]float32{
			0.1, -0.2, 0.3, 0.4,
			-0.5, 0.6, -0.7, 0.8,
			0.9, -1.0, 1.1, -1.2,
		},
		[]int64{1, 3, 4},
	)
	if err != nil {
		t.Fatalf("latent: %v", err)
	}

	embStd, err := tensor.New([]float32{1.5, 0.5, -2.0, 3.0}, []int64{4})
	if err != nil {
		t.Fatalf("embStd: %v", err)
	}

	embMean, err := tensor.New([]float32{0.2, -0.3, 0.4, -0.5}, []int64{4})
	if err != nil {
		t.Fatalf("embMean: %v", err)
	}

	got, err := denormLatentToBCT(latent, embStd, embMean, 4)
	if err != nil {
		t.Fatalf("denormLatentToBCT: %v", err)
	}

	ref, err := tensor.BroadcastMul(latent, embStd)
	if err != nil {
		t.Fatalf("broadcast mul: %v", err)
	}

	ref, err = tensor.BroadcastAdd(ref, embMean)
	if err != nil {
		t.Fatalf("broadcast add: %v", err)
	}

	ref, err = ref.Transpose(1, 2)
	if err != nil {
		t.Fatalf("transpose: %v", err)
	}

	if !shapeEqual(got.Shape(), ref.Shape()) {
		t.Fatalf("shape mismatch: got %v want %v", got.Shape(), ref.Shape())
	}

	gotData := got.RawData()

	refData := ref.RawData()
	if len(gotData) != len(refData) {
		t.Fatalf("data length mismatch: got %d want %d", len(gotData), len(refData))
	}

	const eps = 1e-6
	for i := range gotData {
		if math.Abs(float64(gotData[i]-refData[i])) > eps {
			t.Fatalf("value mismatch at %d: got %.8f want %.8f", i, gotData[i], refData[i])
		}
	}
}

func TestDenormLatentToBCTRejectsInvalidShapes(t *testing.T) {
	t.Parallel()

	latent, err := tensor.New(make([]float32, 1*2*3), []int64{1, 2, 3})
	if err != nil {
		t.Fatalf("latent: %v", err)
	}

	embStd, err := tensor.New([]float32{1, 1}, []int64{2})
	if err != nil {
		t.Fatalf("embStd: %v", err)
	}

	embMean, err := tensor.New([]float32{0, 0}, []int64{2})
	if err != nil {
		t.Fatalf("embMean: %v", err)
	}

	_, err = denormLatentToBCT(latent, embStd, embMean, 3)
	if err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

func TestLatentToMimiProjectorMatchesReference(t *testing.T) {
	t.Parallel()

	flow := &FlowLM{
		cfg: FlowLMConfig{LDim: 4},
	}
	var err error

	flow.embStd, err = tensor.New([]float32{1.5, 0.5, -2.0, 3.0}, []int64{4})
	if err != nil {
		t.Fatalf("embStd: %v", err)
	}

	flow.embMean, err = tensor.New([]float32{0.2, -0.3, 0.4, -0.5}, []int64{4})
	if err != nil {
		t.Fatalf("embMean: %v", err)
	}

	weight, err := tensor.New(
		[]float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			-1, 0.5, 2, -3,
		},
		[]int64{3, 4, 1},
	)
	if err != nil {
		t.Fatalf("weight: %v", err)
	}

	bias, err := tensor.New([]float32{0.1, -0.2, 0.3}, []int64{3})
	if err != nil {
		t.Fatalf("bias: %v", err)
	}

	mimi := &MimiModel{
		quantizerOutProj: &conv1dLayer{
			weight:   weight,
			bias:     bias,
			stride:   1,
			dilation: 1,
			groups:   1,
		},
	}

	projector, err := newLatentToMimiProjector(flow, mimi)
	if err != nil {
		t.Fatalf("newLatentToMimiProjector: %v", err)
	}

	if projector == nil {
		t.Fatal("expected non-nil projector")
	}

	latent, err := tensor.New(
		[]float32{
			0.1, -0.2, 0.3, 0.4,
			-0.5, 0.6, -0.7, 0.8,
			0.9, -1.0, 1.1, -1.2,
		},
		[]int64{1, 3, 4},
	)
	if err != nil {
		t.Fatalf("latent: %v", err)
	}

	got, err := projector.Project(latent)
	if err != nil {
		t.Fatalf("Project: %v", err)
	}

	denorm, err := denormLatentToBCT(latent, flow.embStd, flow.embMean, flow.cfg.LDim)
	if err != nil {
		t.Fatalf("denormLatentToBCT: %v", err)
	}

	want, err := ops.Conv1D(denorm, weight, bias, 1, 0, 1, 1)
	if err != nil {
		t.Fatalf("Conv1D reference: %v", err)
	}

	if !shapeEqual(got.Shape(), want.Shape()) {
		t.Fatalf("shape mismatch: got %v want %v", got.Shape(), want.Shape())
	}

	gotData := got.RawData()

	wantData := want.RawData()
	if len(gotData) != len(wantData) {
		t.Fatalf("data length mismatch: got %d want %d", len(gotData), len(wantData))
	}

	const eps = 1e-5
	for i := range gotData {
		if math.Abs(float64(gotData[i]-wantData[i])) > eps {
			t.Fatalf("value mismatch at %d: got %.8f want %.8f", i, gotData[i], wantData[i])
		}
	}
}

func TestNewLatentToMimiProjectorRejectsUnsupportedKernel(t *testing.T) {
	t.Parallel()

	flow := &FlowLM{
		cfg: FlowLMConfig{LDim: 4},
	}
	var err error

	flow.embStd, err = tensor.New([]float32{1, 1, 1, 1}, []int64{4})
	if err != nil {
		t.Fatalf("embStd: %v", err)
	}

	flow.embMean, err = tensor.New([]float32{0, 0, 0, 0}, []int64{4})
	if err != nil {
		t.Fatalf("embMean: %v", err)
	}

	weight, err := tensor.New(make([]float32, 3*4*3), []int64{3, 4, 3})
	if err != nil {
		t.Fatalf("weight: %v", err)
	}

	mimi := &MimiModel{
		quantizerOutProj: &conv1dLayer{
			weight:   weight,
			stride:   1,
			dilation: 1,
			groups:   1,
		},
	}

	projector, err := newLatentToMimiProjector(flow, mimi)
	if err != nil {
		t.Fatalf("newLatentToMimiProjector: %v", err)
	}

	if projector != nil {
		t.Fatal("expected nil projector for unsupported kernel size")
	}
}

func shapeEqual(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}
