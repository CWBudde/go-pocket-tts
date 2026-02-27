package tensor

import (
	"errors"
	"fmt"
)

// Narrow slices the tensor along a single dimension.
func (t *Tensor) Narrow(dim int, start, length int64) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("tensor: narrow on nil tensor")
	}

	dim, err := normalizeDim(dim, len(t.shape))
	if err != nil {
		return nil, fmt.Errorf("tensor: narrow: %w", err)
	}

	if start < 0 || length < 0 || start+length > t.shape[dim] {
		return nil, fmt.Errorf("tensor: narrow: range [%d:%d] out of bounds for dim %d size %d", start, start+length, dim, t.shape[dim])
	}

	outShape := append([]int64(nil), t.shape...)
	outShape[dim] = length

	out, err := Zeros(outShape)
	if err != nil {
		return nil, err
	}

	srcStrides := computeStrides(t.shape)
	outStrides := computeStrides(outShape)
	coord := make([]int64, len(outShape))
	srcCoord := make([]int64, len(t.shape))

	for i := range out.data {
		linearToCoord(int64(i), outShape, outStrides, coord)
		copy(srcCoord, coord)
		srcCoord[dim] += start
		srcOff := coordToLinear(srcCoord, srcStrides)
		out.data[i] = t.data[srcOff]
	}

	return out, nil
}

// Gather gathers indices along dim.
func (t *Tensor) Gather(dim int, indices []int64) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("tensor: gather on nil tensor")
	}

	if len(indices) == 0 {
		return nil, errors.New("tensor: gather requires at least one index")
	}

	dim, err := normalizeDim(dim, len(t.shape))
	if err != nil {
		return nil, fmt.Errorf("tensor: gather: %w", err)
	}

	for i, idx := range indices {
		if idx < 0 || idx >= t.shape[dim] {
			return nil, fmt.Errorf("tensor: gather index %d (%d) out of range for dim %d size %d", i, idx, dim, t.shape[dim])
		}
	}

	outShape := append([]int64(nil), t.shape...)
	outShape[dim] = int64(len(indices))

	out, err := Zeros(outShape)
	if err != nil {
		return nil, err
	}

	srcStrides := computeStrides(t.shape)
	outStrides := computeStrides(outShape)
	coord := make([]int64, len(outShape))
	srcCoord := make([]int64, len(t.shape))

	for i := range out.data {
		linearToCoord(int64(i), outShape, outStrides, coord)
		copy(srcCoord, coord)
		srcCoord[dim] = indices[coord[dim]]
		srcOff := coordToLinear(srcCoord, srcStrides)
		out.data[i] = t.data[srcOff]
	}

	return out, nil
}

// Transpose swaps dim1 and dim2.
func (t *Tensor) Transpose(dim1, dim2 int) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("tensor: transpose on nil tensor")
	}

	rank := len(t.shape)

	d1, err := normalizeDim(dim1, rank)
	if err != nil {
		return nil, fmt.Errorf("tensor: transpose dim1: %w", err)
	}

	d2, err := normalizeDim(dim2, rank)
	if err != nil {
		return nil, fmt.Errorf("tensor: transpose dim2: %w", err)
	}

	if d1 == d2 {
		return t.Clone(), nil
	}

	outShape := append([]int64(nil), t.shape...)
	outShape[d1], outShape[d2] = outShape[d2], outShape[d1]

	out, err := Zeros(outShape)
	if err != nil {
		return nil, err
	}

	srcStrides := computeStrides(t.shape)
	outStrides := computeStrides(outShape)
	outCoord := make([]int64, rank)
	srcCoord := make([]int64, rank)

	for i := range out.data {
		linearToCoord(int64(i), outShape, outStrides, outCoord)
		copy(srcCoord, outCoord)
		srcCoord[d1], srcCoord[d2] = outCoord[d2], outCoord[d1]
		srcOff := coordToLinear(srcCoord, srcStrides)
		out.data[i] = t.data[srcOff]
	}

	return out, nil
}

// Concat concatenates tensors along dim.
//
//nolint:funlen // Keep full shape validation and copy loop together for correctness/readability.
func Concat(tensors []*Tensor, dim int) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, errors.New("tensor: concat requires at least one tensor")
	}

	first := tensors[0]
	if first == nil {
		return nil, errors.New("tensor: concat tensor 0 is nil")
	}

	rank := len(first.shape)

	dim, err := normalizeDim(dim, rank)
	if err != nil {
		return nil, fmt.Errorf("tensor: concat: %w", err)
	}

	outShape := append([]int64(nil), first.shape...)
	outShape[dim] = 0

	for i, t := range tensors {
		if t == nil {
			return nil, fmt.Errorf("tensor: concat tensor %d is nil", i)
		}

		if len(t.shape) != rank {
			return nil, fmt.Errorf("tensor: concat tensor %d rank %d does not match rank %d", i, len(t.shape), rank)
		}

		for d := range rank {
			if d == dim {
				continue
			}

			if t.shape[d] != first.shape[d] {
				return nil, fmt.Errorf("tensor: concat tensor %d shape %v does not match base shape %v on dim %d", i, t.shape, first.shape, d)
			}
		}

		outShape[dim] += t.shape[dim]
	}

	out, err := Zeros(outShape)
	if err != nil {
		return nil, err
	}

	inner := int64(1)
	for i := dim + 1; i < rank; i++ {
		inner *= outShape[i]
	}

	outer := int64(1)
	for i := range dim {
		outer *= outShape[i]
	}

	outDim := outShape[dim]

	for o := range outer {
		writePos := int64(0)

		for _, t := range tensors {
			span := t.shape[dim] * inner
			srcBase := o * t.shape[dim] * inner
			dstBase := o*outDim*inner + writePos
			copy(out.data[dstBase:dstBase+span], t.data[srcBase:srcBase+span])
			writePos += span
		}
	}

	return out, nil
}
