package tensor

import "fmt"

// BroadcastAdd performs element-wise add with NumPy-style broadcasting.
func BroadcastAdd(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float32) float32 { return x + y }, "add")
}

// BroadcastMul performs element-wise multiply with NumPy-style broadcasting.
func BroadcastMul(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float32) float32 { return x * y }, "mul")
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
