package native

import (
	"errors"
	"fmt"
	"strings"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
	"github.com/example/go-pocket-tts/internal/safetensors"
)

// VarBuilder provides xn-like hierarchical tensor lookup over safetensors.
type VarBuilder struct {
	store  *safetensors.Store
	prefix string
}

func OpenVarBuilder(path string, opts safetensors.StoreOptions) (*VarBuilder, error) {
	store, err := safetensors.OpenStore(path, opts)
	if err != nil {
		return nil, err
	}

	return &VarBuilder{store: store}, nil
}

func NewVarBuilder(store *safetensors.Store) *VarBuilder {
	return &VarBuilder{store: store}
}

func (vb *VarBuilder) Path(parts ...string) *VarBuilder {
	if vb == nil {
		return nil
	}

	prefix := vb.prefix

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		if prefix == "" {
			prefix = part
		} else {
			prefix += "." + part
		}
	}

	return &VarBuilder{store: vb.store, prefix: prefix}
}

func (vb *VarBuilder) Has(name string) bool {
	if vb == nil || vb.store == nil {
		return false
	}

	return vb.store.Has(vb.resolve(name))
}

func (vb *VarBuilder) Tensor(name string, wantShape ...int64) (*tensor.Tensor, error) {
	if vb == nil || vb.store == nil {
		return nil, errors.New("native varbuilder: uninitialized store")
	}

	fullName := vb.resolve(name)

	st, err := vb.store.Tensor(fullName)
	if err != nil {
		return nil, err
	}

	if len(wantShape) > 0 && !equalShape(st.Shape, wantShape) {
		return nil, fmt.Errorf("native varbuilder: tensor %q shape %v does not match expected %v", fullName, st.Shape, wantShape)
	}

	t, err := tensor.New(st.Data, st.Shape)
	if err != nil {
		return nil, fmt.Errorf("native varbuilder: tensor %q: %w", fullName, err)
	}

	return t, nil
}

func (vb *VarBuilder) TensorMaybe(name string, wantShape ...int64) (*tensor.Tensor, bool, error) {
	if !vb.Has(name) {
		return nil, false, nil
	}

	t, err := vb.Tensor(name, wantShape...)
	if err != nil {
		return nil, true, err
	}

	return t, true, nil
}

func (vb *VarBuilder) resolve(name string) string {
	name = strings.TrimSpace(name)
	if vb == nil || vb.prefix == "" {
		return name
	}

	if name == "" {
		return vb.prefix
	}

	return vb.prefix + "." + name
}

func equalShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}
