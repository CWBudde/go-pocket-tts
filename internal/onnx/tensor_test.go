package onnx

import (
	"reflect"
	"strings"
	"testing"
)

func TestNewTensor(t *testing.T) {
	t.Run("float32 ok", func(t *testing.T) {
		tt, err := NewTensor([]float32{1, 2, 3, 4}, []int64{2, 2})
		if err != nil {
			t.Fatalf("NewTensor failed: %v", err)
		}

		if tt.DType() != DTypeFloat32 {
			t.Fatalf("expected dtype float32, got %s", tt.DType())
		}

		if !reflect.DeepEqual(tt.Shape(), []int64{2, 2}) {
			t.Fatalf("unexpected shape: %v", tt.Shape())
		}

		got, err := ExtractFloat32(tt)
		if err != nil {
			t.Fatalf("ExtractFloat32 failed: %v", err)
		}

		if !reflect.DeepEqual(got, []float32{1, 2, 3, 4}) {
			t.Fatalf("unexpected data: %v", got)
		}
	})

	t.Run("shape mismatch", func(t *testing.T) {
		_, err := NewTensor([]int64{1, 2, 3}, []int64{2, 2})
		if err == nil {
			t.Fatal("expected shape mismatch error")
		}

		if !strings.Contains(err.Error(), "expects 4 elements, got 3") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

func TestNewZeroTensor(t *testing.T) {
	tests := []struct {
		name      string
		dtype     string
		shape     []any
		wantDType TensorDType
		wantShape []int64
	}{
		{
			name:      "float with symbolic dim",
			dtype:     "float",
			shape:     []any{1.0, "sequence"},
			wantDType: DTypeFloat32,
			wantShape: []int64{1, 1},
		},
		{
			name:      "int64 fixed shape",
			dtype:     "int64",
			shape:     []any{2.0, 3.0},
			wantDType: DTypeInt64,
			wantShape: []int64{2, 3},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewZeroTensor(tt.dtype, tt.shape)
			if err != nil {
				t.Fatalf("NewZeroTensor failed: %v", err)
			}

			if got.DType() != tt.wantDType {
				t.Fatalf("expected dtype %s, got %s", tt.wantDType, got.DType())
			}

			if !reflect.DeepEqual(got.Shape(), tt.wantShape) {
				t.Fatalf("expected shape %v, got %v", tt.wantShape, got.Shape())
			}
		})
	}
}

func TestNewZeroTensorErrors(t *testing.T) {
	tests := []struct {
		name      string
		dtype     string
		shape     []any
		wantError string
	}{
		{
			name:      "unsupported dtype",
			dtype:     "bool",
			shape:     []any{1.0},
			wantError: "unsupported tensor dtype",
		},
		{
			name:      "invalid shape value",
			dtype:     "float32",
			shape:     []any{0.0, 2.0},
			wantError: "not a positive integer",
		},
		{
			name:      "unsupported shape type",
			dtype:     "int64",
			shape:     []any{true},
			wantError: "unsupported type",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewZeroTensor(tt.dtype, tt.shape)
			if err == nil {
				t.Fatal("expected error")
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("expected error containing %q, got %v", tt.wantError, err)
			}
		})
	}
}

func TestExtractors(t *testing.T) {
	floats, err := ExtractFloat32([]float32{1, 2})
	if err != nil {
		t.Fatalf("ExtractFloat32 failed: %v", err)
	}

	if !reflect.DeepEqual(floats, []float32{1, 2}) {
		t.Fatalf("unexpected float extract: %v", floats)
	}

	ints, err := ExtractInt64([]int64{3, 4})
	if err != nil {
		t.Fatalf("ExtractInt64 failed: %v", err)
	}

	if !reflect.DeepEqual(ints, []int64{3, 4}) {
		t.Fatalf("unexpected int extract: %v", ints)
	}

	if _, err := ExtractFloat32([]int64{1}); err == nil {
		t.Fatal("expected float extractor type error")
	}

	if _, err := ExtractInt64([]float32{1}); err == nil {
		t.Fatal("expected int extractor type error")
	}
}
