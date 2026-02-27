package onnx

import (
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// extractTypedSlice — pointer-to-slice, Tensor, *Tensor, nil, wrong type
// ---------------------------------------------------------------------------

func TestExtractFloat32_PointerToSlice(t *testing.T) {
	data := []float32{1.5, 2.5}
	got, err := ExtractFloat32(&data)
	if err != nil {
		t.Fatalf("ExtractFloat32(*[]float32) error: %v", err)
	}

	if len(got) != 2 || got[0] != 1.5 || got[1] != 2.5 {
		t.Fatalf("got %v; want [1.5 2.5]", got)
	}

	// Returned slice must be a copy.
	got[0] = 99
	if data[0] != 1.5 {
		t.Fatal("ExtractFloat32 returned non-copy")
	}
}

func TestExtractFloat32_NilPointerToSlice(t *testing.T) {
	var p *[]float32

	_, err := ExtractFloat32(p)
	if err == nil || !strings.Contains(err.Error(), "nil") {
		t.Fatalf("expected nil error, got: %v", err)
	}
}

func TestExtractFloat32_FromTensorValue(t *testing.T) {
	tt, err := NewTensor([]float32{10, 20}, []int64{2})
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}

	// Pass Tensor by value (not pointer).
	got, err := ExtractFloat32(*tt)
	if err != nil {
		t.Fatalf("ExtractFloat32(Tensor) error: %v", err)
	}

	if len(got) != 2 || got[0] != 10 {
		t.Fatalf("got %v; want [10 20]", got)
	}
}

func TestExtractFloat32_NilTensorPointer(t *testing.T) {
	var tp *Tensor

	_, err := ExtractFloat32(tp)
	if err == nil || !strings.Contains(err.Error(), "nil") {
		t.Fatalf("expected nil tensor error, got: %v", err)
	}
}

func TestExtractFloat32_WrongDTypeTensor(t *testing.T) {
	tt, err := NewTensor([]int64{1, 2}, []int64{2})
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}

	_, err = ExtractFloat32(tt)
	if err == nil || !strings.Contains(err.Error(), "expected") {
		t.Fatalf("expected type mismatch error, got: %v", err)
	}
}

func TestExtractInt64_PointerToSlice(t *testing.T) {
	data := []int64{3, 4}
	got, err := ExtractInt64(&data)
	if err != nil {
		t.Fatalf("ExtractInt64(*[]int64) error: %v", err)
	}

	if len(got) != 2 || got[0] != 3 {
		t.Fatalf("got %v; want [3 4]", got)
	}
}

func TestExtractFloat32_NilOutput(t *testing.T) {
	_, err := ExtractFloat32(nil)
	if err == nil || !strings.Contains(err.Error(), "nil") {
		t.Fatalf("expected nil error, got: %v", err)
	}
}

func TestExtractFloat32_UnsupportedType(t *testing.T) {
	_, err := ExtractFloat32("hello")
	if err == nil || !strings.Contains(err.Error(), "expected") {
		t.Fatalf("expected type error, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// resolveShape — int, int64, string, edge cases
// ---------------------------------------------------------------------------

func TestResolveShape_IntType(t *testing.T) {
	got, err := resolveShape([]any{int(3), int(4)})
	if err != nil {
		t.Fatalf("resolveShape int: %v", err)
	}

	if len(got) != 2 || got[0] != 3 || got[1] != 4 {
		t.Fatalf("got %v; want [3 4]", got)
	}
}

func TestResolveShape_Int64Type(t *testing.T) {
	got, err := resolveShape([]any{int64(5)})
	if err != nil {
		t.Fatalf("resolveShape int64: %v", err)
	}

	if len(got) != 1 || got[0] != 5 {
		t.Fatalf("got %v; want [5]", got)
	}
}

func TestResolveShape_SymbolicString(t *testing.T) {
	got, err := resolveShape([]any{float64(2), "batch_size"})
	if err != nil {
		t.Fatalf("resolveShape string: %v", err)
	}

	if len(got) != 2 || got[0] != 2 || got[1] != 1 {
		t.Fatalf("got %v; want [2 1]", got)
	}
}

func TestResolveShape_EmptyString(t *testing.T) {
	_, err := resolveShape([]any{" "})
	if err == nil || !strings.Contains(err.Error(), "empty symbolic") {
		t.Fatalf("expected empty-symbolic error, got: %v", err)
	}
}

func TestResolveShape_NegativeFloat(t *testing.T) {
	_, err := resolveShape([]any{float64(-1)})
	if err == nil || !strings.Contains(err.Error(), "not a positive") {
		t.Fatalf("expected positive error, got: %v", err)
	}
}

func TestResolveShape_FractionalFloat(t *testing.T) {
	_, err := resolveShape([]any{float64(2.5)})
	if err == nil || !strings.Contains(err.Error(), "not a positive integer") {
		t.Fatalf("expected integer error, got: %v", err)
	}
}

func TestResolveShape_NegativeInt(t *testing.T) {
	_, err := resolveShape([]any{int(-1)})
	if err == nil || !strings.Contains(err.Error(), "not positive") {
		t.Fatalf("expected positive error, got: %v", err)
	}
}

func TestResolveShape_NegativeInt64(t *testing.T) {
	_, err := resolveShape([]any{int64(-1)})
	if err == nil || !strings.Contains(err.Error(), "not positive") {
		t.Fatalf("expected positive error, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// elementCount
// ---------------------------------------------------------------------------

func TestElementCount_Empty(t *testing.T) {
	got, err := elementCount(nil)
	if err != nil {
		t.Fatalf("elementCount(nil) error: %v", err)
	}

	if got != 1 {
		t.Fatalf("elementCount(nil) = %d; want 1", got)
	}
}

func TestElementCount_NonPositiveDim(t *testing.T) {
	_, err := elementCount([]int64{2, 0, 3})
	if err == nil || !strings.Contains(err.Error(), "not positive") {
		t.Fatalf("expected positive error, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// canonicalDType
// ---------------------------------------------------------------------------

func TestCanonicalDType(t *testing.T) {
	tests := []struct {
		raw  string
		want TensorDType
	}{
		{"float", DTypeFloat32},
		{"float32", DTypeFloat32},
		{"FLOAT32", DTypeFloat32},
		{"tensor(float)", DTypeFloat32},
		{"int64", DTypeInt64},
		{"long", DTypeInt64},
		{"tensor(long)", DTypeInt64},
	}

	for _, tt := range tests {
		t.Run(tt.raw, func(t *testing.T) {
			got, err := canonicalDType(tt.raw)
			if err != nil {
				t.Fatalf("canonicalDType(%q) error: %v", tt.raw, err)
			}

			if got != tt.want {
				t.Fatalf("canonicalDType(%q) = %s; want %s", tt.raw, got, tt.want)
			}
		})
	}
}

func TestCanonicalDType_Unsupported(t *testing.T) {
	_, err := canonicalDType("bfloat16")
	if err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("expected unsupported error, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// unwrapData — nested wrappers
// ---------------------------------------------------------------------------

type wrapper struct{ inner any }

func (w wrapper) Data() any { return w.inner }

func TestUnwrapData_Nested(t *testing.T) {
	inner := []float32{1, 2}
	wrapped := wrapper{inner: wrapper{inner: inner}}

	got, err := unwrapData(wrapped)
	if err != nil {
		t.Fatalf("unwrapData error: %v", err)
	}

	data, ok := got.([]float32)
	if !ok || len(data) != 2 {
		t.Fatalf("unwrapData = %v; want []float32{1,2}", got)
	}
}

func TestUnwrapData_Nil(t *testing.T) {
	_, err := unwrapData(nil)
	if err == nil || !strings.Contains(err.Error(), "nil") {
		t.Fatalf("expected nil error, got: %v", err)
	}
}
