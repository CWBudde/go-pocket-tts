package tensor

import (
	"errors"
	"fmt"
)

// Tensor is a dense, row-major float32 tensor used by the safetensors-native
// runtime path.
type Tensor struct {
	shape []int64
	data  []float32
}

// New creates a tensor from data and shape.
func New(data []float32, shape []int64) (*Tensor, error) {
	total, err := shapeElemCount(shape)
	if err != nil {
		return nil, err
	}

	if len(data) != total {
		return nil, fmt.Errorf("tensor: data length %d does not match shape %v (%d elements)", len(data), shape, total)
	}

	s := append([]int64(nil), shape...)
	d := append([]float32(nil), data...)

	return &Tensor{shape: s, data: d}, nil
}

// newOwned creates a Tensor taking ownership of the provided data and shape
// slices without copying. The caller must not retain or modify data or shape
// after this call. len(data) must equal the product of shape elements; this is
// the caller's responsibility and is not validated here.
func newOwned(data []float32, shape []int64) *Tensor {
	return &Tensor{shape: shape, data: data}
}

// Zeros creates a zero-initialized tensor.
func Zeros(shape []int64) (*Tensor, error) {
	total, err := shapeElemCount(shape)
	if err != nil {
		return nil, err
	}

	return &Tensor{
		shape: append([]int64(nil), shape...),
		data:  make([]float32, total),
	}, nil
}

// Full creates a tensor filled with value.
func Full(shape []int64, value float32) (*Tensor, error) {
	t, err := Zeros(shape)
	if err != nil {
		return nil, err
	}

	for i := range t.data {
		t.data[i] = value
	}

	return t, nil
}

func (t *Tensor) Shape() []int64 {
	if t == nil {
		return nil
	}

	return append([]int64(nil), t.shape...)
}

// Data returns a copy of the underlying tensor data.
func (t *Tensor) Data() []float32 {
	if t == nil {
		return nil
	}

	return append([]float32(nil), t.data...)
}

// RawData returns the underlying data slice.
// Callers must treat it as read-only.
func (t *Tensor) RawData() []float32 {
	if t == nil {
		return nil
	}

	return t.data
}

func (t *Tensor) ElemCount() int {
	if t == nil {
		return 0
	}

	return len(t.data)
}

func (t *Tensor) Rank() int {
	if t == nil {
		return 0
	}

	return len(t.shape)
}

// Clone returns a deep copy.
func (t *Tensor) Clone() *Tensor {
	if t == nil {
		return nil
	}

	dup, _ := New(t.data, t.shape)

	return dup
}

// Reshape returns a tensor with a new shape and shared backing data.
// Callers must treat RawData as read-only unless they intentionally want
// aliasing effects between the original and reshaped tensors.
func (t *Tensor) Reshape(shape []int64) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("tensor: reshape on nil tensor")
	}

	total, err := shapeElemCount(shape)
	if err != nil {
		return nil, err
	}

	if total != len(t.data) {
		return nil, fmt.Errorf("tensor: cannot reshape %v (%d elements) to %v (%d elements)", t.shape, len(t.data), shape, total)
	}

	return &Tensor{shape: append([]int64(nil), shape...), data: t.data}, nil
}
