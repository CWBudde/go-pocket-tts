package onnx

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

type TensorDType string

const (
	DTypeFloat32 TensorDType = "float32"
	DTypeInt64   TensorDType = "int64"
)

type Tensor struct {
	dtype TensorDType
	shape []int64
	data  any
}

func NewTensor[T ~int64 | ~float32](data []T, shape []int64) (*Tensor, error) {
	dtype, err := dtypeFromSlice(data)
	if err != nil {
		return nil, err
	}

	err = validateShapeAgainstData(shape, len(data))
	if err != nil {
		return nil, err
	}

	t := &Tensor{
		dtype: dtype,
		shape: append([]int64(nil), shape...),
	}
	switch dtype {
	case DTypeFloat32:
		converted := make([]float32, len(data))
		for i, v := range data {
			converted[i] = float32(v)
		}

		t.data = converted
	case DTypeInt64:
		converted := make([]int64, len(data))
		for i, v := range data {
			converted[i] = int64(v)
		}

		t.data = converted
	default:
		return nil, fmt.Errorf("unsupported tensor dtype %q", dtype)
	}

	return t, nil
}

func NewZeroTensor(dtype string, shape []any) (*Tensor, error) {
	canonical, err := canonicalDType(dtype)
	if err != nil {
		return nil, err
	}

	resolvedShape, err := resolveShape(shape)
	if err != nil {
		return nil, err
	}

	count, err := elementCount(resolvedShape)
	if err != nil {
		return nil, err
	}

	switch canonical {
	case DTypeFloat32:
		return NewTensor(make([]float32, count), resolvedShape)
	case DTypeInt64:
		return NewTensor(make([]int64, count), resolvedShape)
	default:
		return nil, fmt.Errorf("unsupported tensor dtype %q", canonical)
	}
}

func (t *Tensor) DType() TensorDType {
	return t.dtype
}

func (t *Tensor) Shape() []int64 {
	return append([]int64(nil), t.shape...)
}

func (t *Tensor) Data() any {
	switch v := t.data.(type) {
	case []float32:
		return append([]float32(nil), v...)
	case []int64:
		return append([]int64(nil), v...)
	default:
		return nil
	}
}

func ExtractFloat32(output any) ([]float32, error) {
	return extractTypedSlice[float32](output, DTypeFloat32, "float32")
}

func ExtractInt64(output any) ([]int64, error) {
	return extractTypedSlice[int64](output, DTypeInt64, "int64")
}

func extractTypedSlice[T ~float32 | ~int64](output any, expected TensorDType, typeName string) ([]T, error) {
	v, err := unwrapData(output)
	if err != nil {
		return nil, err
	}

	switch out := v.(type) {
	case []T:
		return append([]T(nil), out...), nil
	case *[]T:
		if out == nil {
			return nil, fmt.Errorf("expected []%s output, got nil *[]%s", typeName, typeName)
		}

		return append([]T(nil), (*out)...), nil
	case Tensor:
		return extractTypedSliceFromTensor[T](&out, expected, typeName)
	case *Tensor:
		if out == nil {
			return nil, errors.New("expected *Tensor output, got nil")
		}

		return extractTypedSliceFromTensor[T](out, expected, typeName)
	default:
		return nil, fmt.Errorf("expected []%s output, got %T", typeName, v)
	}
}

func extractTypedSliceFromTensor[T ~float32 | ~int64](out *Tensor, expected TensorDType, typeName string) ([]T, error) {
	if out.dtype != expected {
		return nil, fmt.Errorf("expected %s tensor, got %s", typeName, out.dtype)
	}

	data, ok := out.data.([]T)
	if !ok {
		return nil, fmt.Errorf("%s tensor has unexpected backing type %T", typeName, out.data)
	}

	return append([]T(nil), data...), nil
}

func unwrapData(output any) (any, error) {
	type dataGetter interface {
		Data() any
	}

	const maxDepth = 16

	v := output
	for range maxDepth {
		if v == nil {
			return nil, errors.New("output is nil")
		}

		getter, ok := v.(dataGetter)
		if !ok {
			return v, nil
		}

		v = getter.Data()
	}

	return nil, fmt.Errorf("nested Data() wrappers exceed max depth %d", maxDepth)
}

func dtypeFromSlice[T ~int64 | ~float32](_ []T) (TensorDType, error) {
	var zero T
	switch any(zero).(type) {
	case int64:
		return DTypeInt64, nil
	case float32:
		return DTypeFloat32, nil
	default:
		return "", fmt.Errorf("unsupported tensor data type %T", zero)
	}
}

func canonicalDType(raw string) (TensorDType, error) {
	normalized := strings.ToLower(strings.TrimSpace(raw))
	normalized = strings.TrimPrefix(normalized, "tensor(")

	normalized = strings.TrimSuffix(normalized, ")")
	switch normalized {
	case "float", "float32":
		return DTypeFloat32, nil
	case "int64", "long":
		return DTypeInt64, nil
	default:
		return "", fmt.Errorf("unsupported tensor dtype %q", raw)
	}
}

func resolveShape(shape []any) ([]int64, error) {
	out := make([]int64, len(shape))
	for i, dim := range shape {
		switch v := dim.(type) {
		case float64:
			if v < 1 || v != math.Trunc(v) {
				return nil, fmt.Errorf("shape[%d]=%v is not a positive integer", i, v)
			}

			out[i] = int64(v)
		case int:
			if v < 1 {
				return nil, fmt.Errorf("shape[%d]=%d is not positive", i, v)
			}

			out[i] = int64(v)
		case int64:
			if v < 1 {
				return nil, fmt.Errorf("shape[%d]=%d is not positive", i, v)
			}

			out[i] = v
		case string:
			if strings.TrimSpace(v) == "" {
				return nil, fmt.Errorf("shape[%d] has empty symbolic dimension", i)
			}

			out[i] = 1
		default:
			return nil, fmt.Errorf("shape[%d] has unsupported type %T", i, dim)
		}
	}

	return out, nil
}

func validateShapeAgainstData(shape []int64, dataLen int) error {
	count, err := elementCount(shape)
	if err != nil {
		return err
	}

	if count != dataLen {
		return fmt.Errorf("shape %v expects %d elements, got %d", shape, count, dataLen)
	}

	return nil
}

func elementCount(shape []int64) (int, error) {
	if len(shape) == 0 {
		return 1, nil
	}

	count := int64(1)

	for i, dim := range shape {
		if dim < 1 {
			return 0, fmt.Errorf("shape[%d]=%d is not positive", i, dim)
		}

		if count > math.MaxInt64/dim {
			return 0, fmt.Errorf("shape %v overflows element count", shape)
		}

		count *= dim
	}

	if count > int64(math.MaxInt) {
		return 0, fmt.Errorf("shape %v exceeds platform int capacity", shape)
	}

	return int(count), nil
}

// ConcatTensorsDim1 concatenates two 3D float32 tensors along dimension 1.
// Both tensors must have shape [B, T_x, D] with matching B and D.
// Returns a tensor with shape [B, T_a + T_b, D].
func ConcatTensorsDim1(a, b *Tensor) (*Tensor, error) {
	aShape := a.Shape()

	bShape := b.Shape()
	if len(aShape) != 3 || len(bShape) != 3 {
		return nil, fmt.Errorf("ConcatTensorsDim1: both tensors must be 3D, got %dD and %dD", len(aShape), len(bShape))
	}

	if aShape[0] != bShape[0] {
		return nil, fmt.Errorf("ConcatTensorsDim1: batch dim mismatch: %d vs %d", aShape[0], bShape[0])
	}

	if aShape[2] != bShape[2] {
		return nil, fmt.Errorf("ConcatTensorsDim1: last dim mismatch: %d vs %d", aShape[2], bShape[2])
	}

	aData, err := ExtractFloat32(a)
	if err != nil {
		return nil, fmt.Errorf("ConcatTensorsDim1: extract a: %w", err)
	}

	bData, err := ExtractFloat32(b)
	if err != nil {
		return nil, fmt.Errorf("ConcatTensorsDim1: extract b: %w", err)
	}

	combined := make([]float32, 0, len(aData)+len(bData))
	combined = append(combined, aData...)
	combined = append(combined, bData...)

	outShape := []int64{aShape[0], aShape[1] + bShape[1], aShape[2]}

	return NewTensor(combined, outShape)
}
