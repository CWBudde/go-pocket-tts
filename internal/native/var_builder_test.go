package native

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	"github.com/example/go-pocket-tts/internal/safetensors"
)

type tensorMeta struct {
	DType   string  `json:"dtype"`
	Shape   []int64 `json:"shape"`
	Offsets [2]int  `json:"data_offsets"`
}

func buildSafetensors(t *testing.T, tensors map[string]struct {
	dtype string
	shape []int64
	data  []byte
},
) []byte {
	t.Helper()

	head := map[string]tensorMeta{}
	offset := 0

	totalDataLen := 0
	for _, spec := range tensors {
		totalDataLen += len(spec.data)
	}

	blob := make([]byte, 0, totalDataLen)

	for name, spec := range tensors {
		start := offset
		end := start + len(spec.data)
		head[name] = tensorMeta{DType: spec.dtype, Shape: spec.shape, Offsets: [2]int{start, end}}
		offset = end

		blob = append(blob, spec.data...)
	}

	headJSON, err := json.Marshal(head)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}

	out := make([]byte, 8+len(headJSON)+len(blob))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headJSON)))
	copy(out[8:], headJSON)
	copy(out[8+len(headJSON):], blob)

	return out
}

func f32Bytes(vals []float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}

	return out
}

func TestVarBuilder_PathTensor(t *testing.T) {
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"flow_lm.bos_emb":                           {dtype: "F32", shape: []int64{2}, data: f32Bytes([]float32{1, 2})},
		"flow_lm.conditioner.embed.weight":          {dtype: "F32", shape: []int64{3, 2}, data: f32Bytes([]float32{10, 11, 20, 21, 30, 31})},
		"flow_lm.transformer.layers.0.norm1.weight": {dtype: "F32", shape: []int64{2}, data: f32Bytes([]float32{1, 1})},
	})

	st, err := safetensors.OpenStoreFromBytes(blob, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("open store: %v", err)
	}

	vb := NewVarBuilder(st)
	if !vb.Path("flow_lm").Has("bos_emb") {
		t.Fatalf("expected flow_lm.bos_emb to exist")
	}

	b, err := vb.Path("flow_lm").Tensor("bos_emb", 2)
	if err != nil {
		t.Fatalf("tensor: %v", err)
	}

	if got := b.Data(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("bos_emb data = %v", got)
	}
}

func TestLUTConditioner_EmbedTokens(t *testing.T) {
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"flow_lm.conditioner.embed.weight": {dtype: "F32", shape: []int64{4, 2}, data: f32Bytes([]float32{1, 2, 3, 4, 5, 6, 7, 8})},
	})

	st, err := safetensors.OpenStoreFromBytes(blob, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("open store: %v", err)
	}

	c, err := loadLUTConditioner(NewVarBuilder(st).Path("flow_lm"))
	if err != nil {
		t.Fatalf("load conditioner: %v", err)
	}

	emb, err := c.EmbedTokens([]int64{2, 0, 3})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}

	if got := emb.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 3 || got[2] != 2 {
		t.Fatalf("shape = %v", got)
	}

	want := []float32{5, 6, 1, 2, 7, 8}
	if got := emb.Data(); len(got) != len(want) {
		t.Fatalf("data len = %d want %d", len(got), len(want))
	} else {
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("data[%d]=%v want %v", i, got[i], want[i])
			}
		}
	}
}
