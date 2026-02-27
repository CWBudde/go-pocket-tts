package tensor

import (
	"strings"
	"testing"
)

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

func TestShapeElemCount_Empty(t *testing.T) {
	got, err := shapeElemCount([]int64{})
	if err != nil {
		t.Fatalf("shapeElemCount([]) error: %v", err)
	}

	if got != 1 {
		t.Fatalf("shapeElemCount([]) = %d; want 1 (scalar)", got)
	}
}

func TestShapeElemCount_NegativeDim(t *testing.T) {
	_, err := shapeElemCount([]int64{2, -3})
	if err == nil || !strings.Contains(err.Error(), "negative") {
		t.Fatalf("shapeElemCount with negative dim: got %v", err)
	}
}

func TestShapeElemCount_ZeroDim(t *testing.T) {
	got, err := shapeElemCount([]int64{3, 0, 5})
	if err != nil {
		t.Fatalf("shapeElemCount with zero dim error: %v", err)
	}

	if got != 0 {
		t.Fatalf("shapeElemCount([3,0,5]) = %d; want 0", got)
	}
}

func TestLinearToCoord(t *testing.T) {
	shape := []int64{2, 3, 4}
	strides := computeStrides(shape)

	// Expected strides: [12, 4, 1]
	wantStrides := []int64{12, 4, 1}
	if !equalI64(strides, wantStrides) {
		t.Fatalf("computeStrides(%v) = %v; want %v", shape, strides, wantStrides)
	}

	tests := []struct {
		linear int64
		want   []int64
	}{
		{0, []int64{0, 0, 0}},
		{1, []int64{0, 0, 1}},
		{4, []int64{0, 1, 0}},
		{13, []int64{1, 0, 1}},
		{23, []int64{1, 2, 3}},
	}

	for _, tt := range tests {
		out := make([]int64, 3)
		linearToCoord(tt.linear, shape, strides, out)

		if !equalI64(out, tt.want) {
			t.Errorf("linearToCoord(%d) = %v; want %v", tt.linear, out, tt.want)
		}
	}
}

func TestComputeStrides_Empty(t *testing.T) {
	got := computeStrides(nil)
	if got != nil {
		t.Fatalf("computeStrides(nil) = %v; want nil", got)
	}
}
