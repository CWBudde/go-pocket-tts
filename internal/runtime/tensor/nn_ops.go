package tensor

import (
	"errors"
	"fmt"
	"math"
)

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
//
//nolint:funlen // Broadcasting and indexing logic is intentionally explicit.
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
