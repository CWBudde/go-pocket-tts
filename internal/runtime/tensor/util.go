package tensor

import (
	"fmt"
	"math"
)

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
