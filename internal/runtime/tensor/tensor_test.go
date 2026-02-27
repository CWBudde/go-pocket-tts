package tensor

import (
	"strings"
	"testing"
)

func TestFull(t *testing.T) {
	got, err := Full([]int64{2, 3}, 1.25)
	if err != nil {
		t.Fatalf("Full returned error: %v", err)
	}

	if shape := got.Shape(); !equalI64(shape, []int64{2, 3}) {
		t.Fatalf("shape = %v, want [2 3]", shape)
	}

	want := []float32{1.25, 1.25, 1.25, 1.25, 1.25, 1.25}
	if data := got.Data(); !equalF32(data, want, 0) {
		t.Fatalf("data = %v, want %v", data, want)
	}
}

func TestRawDataAndElemCount(t *testing.T) {
	var nilTensor *Tensor
	if nilTensor.RawData() != nil {
		t.Fatal("nil tensor RawData should be nil")
	}

	if nilTensor.ElemCount() != 0 {
		t.Fatalf("nil tensor ElemCount = %d, want 0", nilTensor.ElemCount())
	}

	tt, err := New([]float32{1, 2, 3}, []int64{3})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if got := tt.ElemCount(); got != 3 {
		t.Fatalf("ElemCount = %d, want 3", got)
	}

	raw := tt.RawData()
	raw[1] = 9

	// RawData exposes the backing slice by design.
	if got := tt.Data()[1]; got != 9 {
		t.Fatalf("RawData mutation did not affect tensor backing data, got %v", got)
	}
}

func TestNilTensor_Methods(t *testing.T) {
	var nilT *Tensor

	if nilT.Shape() != nil {
		t.Error("nil Shape() should be nil")
	}

	if nilT.Data() != nil {
		t.Error("nil Data() should be nil")
	}

	if nilT.Rank() != 0 {
		t.Errorf("nil Rank() = %d; want 0", nilT.Rank())
	}

	if nilT.Clone() != nil {
		t.Error("nil Clone() should be nil")
	}

	if nilT.ElemCount() != 0 {
		t.Errorf("nil ElemCount() = %d; want 0", nilT.ElemCount())
	}
}

func TestReshapePreservesValues(t *testing.T) {
	x, err := New([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	y, err := x.Reshape([]int64{3, 2})
	if err != nil {
		t.Fatalf("reshape: %v", err)
	}

	if got := y.Shape(); !equalI64(got, []int64{3, 2}) {
		t.Fatalf("shape = %v, want [3 2]", got)
	}

	if got := y.Data(); !equalF32(got, []float32{1, 2, 3, 4, 5, 6}, 0) {
		t.Fatalf("data = %v", got)
	}
}

func TestReshape_NilTensor(t *testing.T) {
	var nilT *Tensor

	_, err := nilT.Reshape([]int64{1})
	if err == nil || !strings.Contains(err.Error(), "nil tensor") {
		t.Fatalf("Reshape on nil tensor: got %v", err)
	}
}

func TestReshape_SizeMismatch(t *testing.T) {
	x, _ := New([]float32{1, 2, 3}, []int64{3})

	_, err := x.Reshape([]int64{2, 2})
	if err == nil || !strings.Contains(err.Error(), "cannot reshape") {
		t.Fatalf("Reshape size mismatch: got %v", err)
	}
}

func TestReshape_NegativeDim(t *testing.T) {
	x, _ := New([]float32{1, 2}, []int64{2})

	_, err := x.Reshape([]int64{-1})
	if err == nil || !strings.Contains(err.Error(), "negative") {
		t.Fatalf("Reshape negative dim: got %v", err)
	}
}

func TestClone_Independence(t *testing.T) {
	x, _ := New([]float32{1, 2, 3}, []int64{3})
	y := x.Clone()

	y.RawData()[0] = 99

	if x.Data()[0] != 1 {
		t.Fatal("Clone shares backing data with original")
	}
}
