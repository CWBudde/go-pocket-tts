package ops

import (
	"fmt"
	"math"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

// MLP computes linear(silu(linear(x))).
func MLP(x, w1, b1, w2, b2 *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := tensor.Linear(x, w1, b1)
	if err != nil {
		return nil, fmt.Errorf("ops: mlp first linear: %w", err)
	}

	hAct := h.Clone()
	for i, v := range hAct.RawData() {
		hAct.RawData()[i] = silu(v)
	}

	out, err := tensor.Linear(hAct, w2, b2)
	if err != nil {
		return nil, fmt.Errorf("ops: mlp second linear: %w", err)
	}

	return out, nil
}

func silu(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}
