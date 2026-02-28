package ops

import (
	"errors"
	"fmt"
	"math"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

// CausalMask sets positions where key index > query index + offset to -Inf.
// Expected input shape: [..., query, key].
func CausalMask(scores *tensor.Tensor, offset int64) (*tensor.Tensor, error) {
	if scores == nil {
		return nil, errors.New("ops: causal mask scores is nil")
	}

	shape := scores.Shape()
	if len(shape) < 2 {
		return nil, fmt.Errorf("ops: causal mask requires rank >= 2, got %d", len(shape))
	}

	q := int(shape[len(shape)-2])

	k := int(shape[len(shape)-1])
	if q <= 0 || k <= 0 {
		return nil, fmt.Errorf("ops: causal mask requires positive query/key dims, got %d and %d", q, k)
	}

	out := scores.Clone()
	data := out.RawData()
	blocks := len(data) / (q * k)
	applyCausalMaskInPlace(data, q, k, blocks, offset)

	return out, nil
}

// Attention computes scaled dot-product attention.
// q shape: [..., tq, d], k shape: [..., tk, d], v shape: [..., tk, dv]
// output: [..., tq, dv]
func Attention(q, k, v *tensor.Tensor, causal bool, offset int64) (*tensor.Tensor, error) {
	if q == nil || k == nil || v == nil {
		return nil, errors.New("ops: attention requires non-nil q/k/v")
	}

	qShape := q.Shape()
	kShape := k.Shape()

	vShape := v.Shape()
	if len(qShape) < 2 || len(kShape) < 2 || len(vShape) < 2 {
		return nil, errors.New("ops: attention requires rank >= 2 inputs")
	}

	d := qShape[len(qShape)-1]
	if d != kShape[len(kShape)-1] {
		return nil, fmt.Errorf("ops: attention q/k depth mismatch %d vs %d", d, kShape[len(kShape)-1])
	}

	if kShape[len(kShape)-2] != vShape[len(vShape)-2] {
		return nil, fmt.Errorf("ops: attention key/value sequence mismatch %d vs %d", kShape[len(kShape)-2], vShape[len(vShape)-2])
	}

	kT, err := k.Transpose(-1, -2)
	if err != nil {
		return nil, fmt.Errorf("ops: attention transpose k: %w", err)
	}

	scores, err := tensor.MatMul(q, kT)
	if err != nil {
		return nil, fmt.Errorf("ops: attention q*k^T: %w", err)
	}

	scale := float32(1.0 / math.Sqrt(float64(d)))

	err = scaleMaskSoftmaxInPlace(scores, scale, causal, offset)
	if err != nil {
		return nil, fmt.Errorf("ops: attention scale/mask/softmax: %w", err)
	}

	out, err := tensor.MatMul(scores, v)
	if err != nil {
		return nil, fmt.Errorf("ops: attention probs*v: %w", err)
	}

	return out, nil
}

func applyCausalMaskInPlace(data []float32, q, k, blocks int, offset int64) {
	negInf := float32(math.Inf(-1))

	for b := range blocks {
		base := b * q * k
		for qi := range q {
			maxKey := int64(qi) + offset

			row := base + qi*k
			for ki := range k {
				if int64(ki) > maxKey {
					data[row+ki] = negInf
				}
			}
		}
	}
}

func scaleMaskSoftmaxInPlace(scores *tensor.Tensor, scale float32, causal bool, offset int64) error {
	if scores == nil {
		return errors.New("ops: softmax input is nil")
	}

	shape := scores.Shape()
	if len(shape) < 2 {
		return fmt.Errorf("ops: softmax expects rank >= 2, got %d", len(shape))
	}

	q := int(shape[len(shape)-2])
	k := int(shape[len(shape)-1])

	if q <= 0 || k <= 0 {
		return fmt.Errorf("ops: softmax expects positive query/key dims, got %d and %d", q, k)
	}

	data := scores.RawData()

	blocks := len(data) / (q * k)
	for b := range blocks {
		base := b * q * k
		for qi := range q {
			row := base + qi*k
			maxKey := int64(qi) + offset
			maxV := float32(math.Inf(-1))

			for ki := range k {
				idx := row + ki

				v := data[idx] * scale
				if causal && int64(ki) > maxKey {
					v = float32(math.Inf(-1))
				}

				data[idx] = v
				if v > maxV {
					maxV = v
				}
			}

			if math.IsInf(float64(maxV), -1) {
				for ki := range k {
					data[row+ki] = 0
				}

				continue
			}

			var sum float64

			for ki := range k {
				idx := row + ki
				e := math.Exp(float64(data[idx] - maxV))
				data[idx] = float32(e)
				sum += e
			}

			if sum == 0 || math.IsNaN(sum) {
				return errors.New("ops: softmax encountered zero normalization sum")
			}

			inv := float32(1.0 / sum)

			for ki := range k {
				idx := row + ki
				data[idx] *= inv
			}
		}
	}

	return nil
}
