package native

import (
	"testing"

	"github.com/example/go-pocket-tts/internal/safetensors"
)

func TestDetectNumHeads(t *testing.T) {
	tests := []struct {
		name     string
		dModel   int64
		fallback int64
		want     int64
	}{
		{name: "dModel=1024 -> 16 heads", dModel: 1024, fallback: 8, want: 16},
		{name: "dModel=512 -> 16 heads", dModel: 512, fallback: 8, want: 16},
		{name: "dModel=256 -> 16 heads", dModel: 256, fallback: 8, want: 16},
		{name: "dModel=192 -> 16 heads", dModel: 192, fallback: 8, want: 16},
		{name: "dModel=8 -> 8 heads", dModel: 8, fallback: 4, want: 8},
		{name: "dModel=4 -> 4 heads", dModel: 4, fallback: 2, want: 4},
		{name: "dModel=3 -> 1 head", dModel: 3, fallback: 99, want: 1},
		{name: "dModel=1 -> 1 head", dModel: 1, fallback: 99, want: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vb := vbWithInProjWeight(t, tt.dModel)
			got := detectNumHeads(vb, tt.fallback)
			if got != tt.want {
				t.Fatalf("detectNumHeads(dModel=%d, fallback=%d) = %d; want %d",
					tt.dModel, tt.fallback, got, tt.want)
			}
		})
	}
}

func TestDetectNumHeads_Fallback(t *testing.T) {
	tests := []struct {
		name     string
		vb       *VarBuilder
		fallback int64
	}{
		{name: "nil VarBuilder", vb: nil, fallback: 42},
		{name: "missing tensor", vb: vbEmpty(t), fallback: 42},
		{name: "wrong shape (1D)", vb: vbWithShape(t, []int64{16}), fallback: 42},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := detectNumHeads(tt.vb, tt.fallback)
			if got != tt.fallback {
				t.Fatalf("detectNumHeads returned %d; want fallback %d", got, tt.fallback)
			}
		})
	}
}

// vbWithInProjWeight builds a VarBuilder containing a single tensor at the
// path detectNumHeads expects, with shape [3*dModel, dModel].
func vbWithInProjWeight(t *testing.T, dModel int64) *VarBuilder {
	t.Helper()

	nElems := 3 * dModel * dModel
	data := make([]byte, nElems*4) // F32 zeros

	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"transformer.layers.0.self_attn.in_proj.weight": {
			dtype: "F32",
			shape: []int64{3 * dModel, dModel},
			data:  data,
		},
	})

	st, err := safetensors.OpenStoreFromBytes(blob, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("open store: %v", err)
	}

	return NewVarBuilder(st)
}

// vbEmpty returns a VarBuilder with no tensors matching detectNumHeads's path.
func vbEmpty(t *testing.T) *VarBuilder {
	t.Helper()

	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"unrelated.tensor": {dtype: "F32", shape: []int64{1}, data: make([]byte, 4)},
	})

	st, err := safetensors.OpenStoreFromBytes(blob, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("open store: %v", err)
	}

	return NewVarBuilder(st)
}

// vbWithShape returns a VarBuilder with the expected tensor name but a custom shape.
func vbWithShape(t *testing.T, shape []int64) *VarBuilder {
	t.Helper()

	nElems := int64(1)
	for _, s := range shape {
		nElems *= s
	}

	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"transformer.layers.0.self_attn.in_proj.weight": {
			dtype: "F32",
			shape: shape,
			data:  make([]byte, nElems*4),
		},
	})

	st, err := safetensors.OpenStoreFromBytes(blob, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("open store: %v", err)
	}

	return NewVarBuilder(st)
}
