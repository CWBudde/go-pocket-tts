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

	scaled := scores.Clone()

	scale := float32(1.0 / math.Sqrt(float64(d)))
	for i := range scaled.RawData() {
		scaled.RawData()[i] *= scale
	}

	if causal {
		scaled, err = CausalMask(scaled, offset)
		if err != nil {
			return nil, fmt.Errorf("ops: attention causal mask: %w", err)
		}
	}

	probs, err := tensor.Softmax(scaled, -1)
	if err != nil {
		return nil, fmt.Errorf("ops: attention softmax: %w", err)
	}

	out, err := tensor.MatMul(probs, v)
	if err != nil {
		return nil, fmt.Errorf("ops: attention probs*v: %w", err)
	}

	return out, nil
}
