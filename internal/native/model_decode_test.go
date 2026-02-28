package native

import (
	"math"
	"testing"

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
