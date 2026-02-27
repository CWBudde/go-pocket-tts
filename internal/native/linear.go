package native

import (
	"errors"
	"fmt"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

type Linear struct {
	Weight *tensor.Tensor // [out, in]
	Bias   *tensor.Tensor // optional [out]
}

func loadLinear(vb *VarBuilder, name string, withBias bool) (*Linear, error) {
	w, err := vb.Tensor(name + ".weight")
	if err != nil {
		return nil, err
	}

	if len(w.Shape()) != 2 {
		return nil, fmt.Errorf("native: linear %q weight must be rank-2, got %v", name, w.Shape())
	}
	var b *tensor.Tensor

	if withBias {
		t, ok, err := vb.TensorMaybe(name + ".bias")
		if err != nil {
			return nil, err
		}

		if ok {
			if len(t.Shape()) != 1 || t.Shape()[0] != w.Shape()[0] {
				return nil, fmt.Errorf("native: linear %q bias shape %v incompatible with weight %v", name, t.Shape(), w.Shape())
			}

			b = t
		}
	}

	return &Linear{Weight: w, Bias: b}, nil
}

func (l *Linear) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if l == nil || l.Weight == nil {
		return nil, errors.New("native: linear is not initialized")
	}

	return tensor.Linear(x, l.Weight, l.Bias)
}

type LayerNorm struct {
	Weight *tensor.Tensor
	Bias   *tensor.Tensor
	Eps    float32
}

func loadLayerNorm(vb *VarBuilder, name string, eps float32) (*LayerNorm, error) {
	w, err := vb.Tensor(name + ".weight")
	if err != nil {
		return nil, err
	}

	b, err := vb.Tensor(name + ".bias")
	if err != nil {
		return nil, err
	}

	if len(w.Shape()) != 1 || len(b.Shape()) != 1 || w.Shape()[0] != b.Shape()[0] {
		return nil, fmt.Errorf("native: layernorm %q invalid shapes weight=%v bias=%v", name, w.Shape(), b.Shape())
	}

	return &LayerNorm{Weight: w, Bias: b, Eps: eps}, nil
}

func (ln *LayerNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if ln == nil || ln.Weight == nil || ln.Bias == nil {
		return nil, errors.New("native: layernorm is not initialized")
	}

	return tensor.LayerNorm(x, ln.Weight, ln.Bias, ln.Eps)
}
