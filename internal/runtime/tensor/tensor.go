package tensor

import (
	"errors"
	"fmt"
	"math"
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

// Reshape returns a tensor with a new shape and shared values.
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

	return &Tensor{shape: append([]int64(nil), shape...), data: append([]float32(nil), t.data...)}, nil
}

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

// BroadcastAdd performs element-wise add with NumPy-style broadcasting.
func BroadcastAdd(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float32) float32 { return x + y }, "add")
}

// BroadcastMul performs element-wise multiply with NumPy-style broadcasting.
func BroadcastMul(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float32) float32 { return x * y }, "mul")
}

// Softmax applies softmax along dim.
func Softmax(x *Tensor, dim int) (*Tensor, error) {
	if x == nil {
		return nil, errors.New("tensor: softmax on nil tensor")
	}

	if len(x.shape) == 0 {
		return nil, errors.New("tensor: softmax requires rank >= 1")
	}

	dim, err := normalizeDim(dim, len(x.shape))
	if err != nil {
		return nil, fmt.Errorf("tensor: softmax: %w", err)
	}

	axis := x.shape[dim]
	if axis <= 0 {
		return nil, fmt.Errorf("tensor: softmax axis dimension must be > 0, got %d", axis)
	}

	inner := int64(1)
	for i := dim + 1; i < len(x.shape); i++ {
		inner *= x.shape[i]
	}

	outer := int64(1)
	for i := range dim {
		outer *= x.shape[i]
	}

	out := x.Clone()

	for o := range outer {
		for in := range inner {
			base := o*axis*inner + in
			maxV := float32(math.Inf(-1))

			for k := range axis {
				v := out.data[base+k*inner]
				if v > maxV {
					maxV = v
				}
			}

			var sum float64

			for k := range axis {
				i := base + k*inner
				e := math.Exp(float64(out.data[i] - maxV))
				out.data[i] = float32(e)
				sum += e
			}

			if sum == 0 {
				return nil, errors.New("tensor: softmax encountered zero normalization sum")
			}

			inv := float32(1.0 / sum)

			for k := range axis {
				i := base + k*inner
				out.data[i] *= inv
			}
		}
	}

	return out, nil
}

// LayerNorm normalizes the last dimension and applies optional weight/bias.
func LayerNorm(x, weight, bias *Tensor, eps float32) (*Tensor, error) {
	if x == nil {
		return nil, errors.New("tensor: layernorm input is nil")
	}

	if x.Rank() < 1 {
		return nil, errors.New("tensor: layernorm requires rank >= 1")
	}

	if eps <= 0 {
		return nil, errors.New("tensor: layernorm eps must be > 0")
	}

	d := x.shape[len(x.shape)-1]
	if d <= 0 {
		return nil, errors.New("tensor: layernorm last dimension must be > 0")
	}

	if weight != nil {
		if weight.Rank() != 1 || weight.shape[0] != d {
			return nil, fmt.Errorf("tensor: layernorm weight shape %v does not match last dimension %d", weight.shape, d)
		}
	}

	if bias != nil {
		if bias.Rank() != 1 || bias.shape[0] != d {
			return nil, fmt.Errorf("tensor: layernorm bias shape %v does not match last dimension %d", bias.shape, d)
		}
	}

	out := x.Clone()
	dd := int(d)

	outer := len(x.data) / dd
	for o := range outer {
		start := o * dd
		slice := out.data[start : start+dd]

		var mean float64
		for _, v := range slice {
			mean += float64(v)
		}

		mean /= float64(dd)

		var variance float64

		for _, v := range slice {
			delta := float64(v) - mean
			variance += delta * delta
		}

		variance /= float64(dd)

		invStd := float32(1.0 / math.Sqrt(variance+float64(eps)))
		for i := range dd {
			n := (slice[i] - float32(mean)) * invStd
			if weight != nil {
				n *= weight.data[i]
			}

			if bias != nil {
				n += bias.data[i]
			}

			slice[i] = n
		}
	}

	return out, nil
}

// MatMul performs batched matrix multiplication with broadcasting over batch dims.
func MatMul(a, b *Tensor) (*Tensor, error) {
	if a == nil || b == nil {
		return nil, errors.New("tensor: matmul requires non-nil inputs")
	}

	if a.Rank() < 2 || b.Rank() < 2 {
		return nil, fmt.Errorf("tensor: matmul requires rank >= 2, got %d and %d", a.Rank(), b.Rank())
	}

	aShape := a.shape
	bShape := b.shape
	aRank := len(aShape)
	bRank := len(bShape)

	m := aShape[aRank-2]
	k := aShape[aRank-1]
	k2 := bShape[bRank-2]

	n := bShape[bRank-1]
	if k != k2 {
		return nil, fmt.Errorf("tensor: matmul mismatch: A shape %v and B shape %v (K dims %d vs %d)", aShape, bShape, k, k2)
	}

	aBatch := aShape[:aRank-2]
	bBatch := bShape[:bRank-2]

	batchShape, err := broadcastShape(aBatch, bBatch)
	if err != nil {
		return nil, fmt.Errorf("tensor: matmul batch broadcast: %w", err)
	}

	outShape := make([]int64, 0, len(batchShape)+2)
	outShape = append(outShape, batchShape...)
	outShape = append(outShape, m, n)

	out, err := Zeros(outShape)
	if err != nil {
		return nil, err
	}

	aStrides := computeStrides(aShape)
	bStrides := computeStrides(bShape)
	outStrides := computeStrides(outShape)

	batchCount, err := shapeElemCount(batchShape)
	if err != nil {
		return nil, err
	}

	batchCoords := make([]int64, len(batchShape))
	batchStrides := computeStrides(batchShape)

	for batchIdx := range batchCount {
		linearToCoord(int64(batchIdx), batchShape, batchStrides, batchCoords)
		aBatchOffset := broadcastBatchOffset(batchCoords, aShape[:aRank-2], aStrides[:aRank-2])
		bBatchOffset := broadcastBatchOffset(batchCoords, bShape[:bRank-2], bStrides[:bRank-2])
		outBatchOffset := coordToLinear(batchCoords, outStrides[:len(batchShape)])

		for i := range m {
			for j := range n {
				var sum float32

				for kk := range k {
					aIdx := aBatchOffset + i*aStrides[aRank-2] + kk*aStrides[aRank-1]
					bIdx := bBatchOffset + kk*bStrides[bRank-2] + j*bStrides[bRank-1]
					sum += a.data[aIdx] * b.data[bIdx]
				}

				outIdx := outBatchOffset + i*outStrides[len(outShape)-2] + j*outStrides[len(outShape)-1]
				out.data[outIdx] = sum
			}
		}
	}

	return out, nil
}

// Linear applies y = x * W^T + b where weight shape is [out, in].
func Linear(x, weight, bias *Tensor) (*Tensor, error) {
	if x == nil || weight == nil {
		return nil, errors.New("tensor: linear requires non-nil x and weight")
	}

	if x.Rank() < 1 {
		return nil, errors.New("tensor: linear requires x rank >= 1")
	}

	if weight.Rank() != 2 {
		return nil, fmt.Errorf("tensor: linear weight must be rank 2, got %d", weight.Rank())
	}

	in := x.shape[x.Rank()-1]

	out := weight.shape[0]
	if weight.shape[1] != in {
		return nil, fmt.Errorf("tensor: linear mismatch: x last dim %d, weight in dim %d", in, weight.shape[1])
	}

	if bias != nil {
		if bias.Rank() != 1 || bias.shape[0] != out {
			return nil, fmt.Errorf("tensor: linear bias shape %v does not match out dim %d", bias.shape, out)
		}
	}

	batch := len(x.data) / int(in)
	outData := make([]float32, batch*int(out))
	inI := int(in)
	outI := int(out)

	wData := weight.data
	for bIdx := range batch {
		xSlice := x.data[bIdx*inI : bIdx*inI+inI]

		yBase := bIdx * outI
		for o := range outI {
			sum := dotF32(xSlice, wData[o*inI:(o+1)*inI])
			if bias != nil {
				sum += bias.data[o]
			}

			outData[yBase+o] = sum
		}
	}

	// Build outShape in-place; reuse the first (rank-1) elements of x.shape.
	outShape := make([]int64, x.Rank())
	copy(outShape, x.shape[:x.Rank()-1])
	outShape[x.Rank()-1] = out

	return newOwned(outData, outShape), nil
}

func broadcastBinary(a, b *Tensor, fn func(x, y float32) float32, opName string) (*Tensor, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("tensor: broadcast %s requires non-nil inputs", opName)
	}

	outShape, err := broadcastShape(a.shape, b.shape)
	if err != nil {
		return nil, fmt.Errorf("tensor: broadcast %s: %w", opName, err)
	}

	out, err := Zeros(outShape)
	if err != nil {
		return nil, err
	}

	aPadShape := leftPadShape(a.shape, len(outShape))
	bPadShape := leftPadShape(b.shape, len(outShape))
	aPadStrides := computeStrides(aPadShape)
	bPadStrides := computeStrides(bPadShape)
	outStrides := computeStrides(outShape)
	coord := make([]int64, len(outShape))

	for i := range out.data {
		linearToCoord(int64(i), outShape, outStrides, coord)

		aOff := int64(0)
		bOff := int64(0)

		for d := range coord {
			ac := coord[d]
			if aPadShape[d] == 1 {
				ac = 0
			}

			bc := coord[d]
			if bPadShape[d] == 1 {
				bc = 0
			}

			aOff += ac * aPadStrides[d]
			bOff += bc * bPadStrides[d]
		}

		out.data[i] = fn(a.data[aOff], b.data[bOff])
	}

	return out, nil
}

func broadcastShape(a, b []int64) ([]int64, error) {
	outRank := max(len(a), len(b))

	out := make([]int64, outRank)
	for i := range outRank {
		ad := int64(1)
		if j := i - (outRank - len(a)); j >= 0 {
			ad = a[j]
		}

		bd := int64(1)
		if j := i - (outRank - len(b)); j >= 0 {
			bd = b[j]
		}

		switch {
		case ad == bd || ad == 1:
			out[i] = bd
		case bd == 1:
			out[i] = ad
		default:
			return nil, fmt.Errorf("cannot broadcast shapes %v and %v", a, b)
		}
	}

	return out, nil
}

func leftPadShape(shape []int64, rank int) []int64 {
	if len(shape) == rank {
		return append([]int64(nil), shape...)
	}

	out := make([]int64, rank)

	pad := rank - len(shape)
	for i := range pad {
		out[i] = 1
	}

	copy(out[pad:], shape)

	return out
}

func broadcastBatchOffset(batchCoords, srcBatchShape, srcBatchStrides []int64) int64 {
	if len(srcBatchShape) == 0 {
		return 0
	}

	outRank := len(batchCoords)
	srcRank := len(srcBatchShape)
	pad := outRank - srcRank
	var off int64

	for i := range srcRank {
		coord := batchCoords[pad+i]
		if srcBatchShape[i] == 1 {
			coord = 0
		}

		off += coord * srcBatchStrides[i]
	}

	return off
}

func shapeElemCount(shape []int64) (int, error) {
	total := int64(1)

	for i, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("tensor: shape %v has negative dimension at %d", shape, i)
		}

		total *= d
		if total > math.MaxInt32 && total > math.MaxInt64/2 {
			return 0, fmt.Errorf("tensor: shape %v too large", shape)
		}
	}

	if total > int64(^uint(0)>>1) {
		return 0, fmt.Errorf("tensor: shape %v exceeds platform int size", shape)
	}

	return int(total), nil
}

func normalizeDim(dim, rank int) (int, error) {
	if rank < 0 {
		return 0, fmt.Errorf("invalid rank %d", rank)
	}

	if dim < 0 {
		dim += rank
	}

	if dim < 0 || dim >= rank {
		return 0, fmt.Errorf("dim %d out of range for rank %d", dim, rank)
	}

	return dim, nil
}

func computeStrides(shape []int64) []int64 {
	if len(shape) == 0 {
		return nil
	}

	strides := make([]int64, len(shape))

	stride := int64(1)
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return strides
}

func linearToCoord(linear int64, shape, strides, out []int64) {
	if len(shape) == 0 {
		return
	}

	for i := range shape {
		if shape[i] == 0 {
			out[i] = 0
			continue
		}

		out[i] = (linear / strides[i]) % shape[i]
	}
}

func coordToLinear(coord, strides []int64) int64 {
	var off int64
	for i, c := range coord {
		off += c * strides[i]
	}

	return off
}
