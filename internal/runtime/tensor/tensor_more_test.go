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

func TestLeftPadShape(t *testing.T) {
	shape := []int64{2, 3}

	// Equal rank should return a copy.
	gotEqual := leftPadShape(shape, 2)
	if !equalI64(gotEqual, []int64{2, 3}) {
		t.Fatalf("leftPadShape equal rank = %v, want [2 3]", gotEqual)
	}

	gotEqual[0] = 99
	if shape[0] != 2 {
		t.Fatalf("leftPadShape should return a copy when rank matches, source mutated: %v", shape)
	}

	gotPadded := leftPadShape(shape, 4)
	if !equalI64(gotPadded, []int64{1, 1, 2, 3}) {
		t.Fatalf("leftPadShape padded = %v, want [1 1 2 3]", gotPadded)
	}
}

func TestNormalizeDim(t *testing.T) {
	got, err := normalizeDim(-1, 3)
	if err != nil {
		t.Fatalf("normalizeDim(-1,3) error: %v", err)
	}

	if got != 2 {
		t.Fatalf("normalizeDim(-1,3) = %d, want 2", got)
	}

	got, err = normalizeDim(1, 3)
	if err != nil {
		t.Fatalf("normalizeDim(1,3) error: %v", err)
	}

	if got != 1 {
		t.Fatalf("normalizeDim(1,3) = %d, want 1", got)
	}

	_, err = normalizeDim(3, 3)
	if err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("expected out-of-range error, got: %v", err)
	}

	_, err = normalizeDim(0, -1)
	if err == nil || !strings.Contains(err.Error(), "invalid rank") {
		t.Fatalf("expected invalid rank error, got: %v", err)
	}
}
