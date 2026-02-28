package native

import (
	"math"
	"testing"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

func TestMulLastDimInPlaceMatchesBroadcastMul(t *testing.T) {
	t.Parallel()

	x, err := tensor.New([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}, []int64{1, 2, 4})
	if err != nil {
		t.Fatalf("x: %v", err)
	}

	scale, err := tensor.New([]float32{0.5, -1, 2, 0.25}, []int64{4})
	if err != nil {
		t.Fatalf("scale: %v", err)
	}

	got := x.Clone()
	got, err = mulLastDimInPlace(got, scale)
	if err != nil {
		t.Fatalf("mulLastDimInPlace: %v", err)
	}

	want, err := tensor.BroadcastMul(x, scale)
	if err != nil {
		t.Fatalf("BroadcastMul: %v", err)
	}

	const eps = 1e-6
	gd := got.RawData()
	wd := want.RawData()
	if len(gd) != len(wd) {
		t.Fatalf("len mismatch: got %d want %d", len(gd), len(wd))
	}

	for i := range gd {
		if math.Abs(float64(gd[i]-wd[i])) > eps {
			t.Fatalf("value mismatch at %d: got %.8f want %.8f", i, gd[i], wd[i])
		}
	}
}

func TestGeluErfTensorInPlaceMatchesCopyVersion(t *testing.T) {
	t.Parallel()

	x, err := tensor.New([]float32{-2, -1, 0, 1, 2}, []int64{5})
	if err != nil {
		t.Fatalf("x: %v", err)
	}

	got := x.Clone()
	got = geluErfTensorInPlace(got)
	want := geluErfTensor(x)

	const eps = 1e-6
	gd := got.RawData()
	wd := want.RawData()
	for i := range gd {
		if math.Abs(float64(gd[i]-wd[i])) > eps {
			t.Fatalf("value mismatch at %d: got %.8f want %.8f", i, gd[i], wd[i])
		}
	}
}

func TestAddSameShapeInPlaceRejectsShapeMismatch(t *testing.T) {
	t.Parallel()

	a, err := tensor.New([]float32{1, 2, 3, 4}, []int64{2, 2})
	if err != nil {
		t.Fatalf("a: %v", err)
	}

	b, err := tensor.New([]float32{1, 2, 3}, []int64{3})
	if err != nil {
		t.Fatalf("b: %v", err)
	}

	_, err = addSameShapeInPlace(a, b)
	if err == nil {
		t.Fatal("expected shape mismatch error")
	}
}
