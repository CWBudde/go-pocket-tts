package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

// Tensor holds a single tensor loaded from a safetensors file.
type Tensor struct {
	Name  string
	Shape []int64
	Data  []float32
}

// LoadFirstTensor reads a safetensors file and returns the first float32 tensor.
// The safetensors format is: 8-byte LE header length → JSON header → raw tensor data.
func LoadFirstTensor(path string) (*Tensor, error) {
	store, err := OpenStore(path, StoreOptions{})
	if err != nil {
		return nil, err
	}
	defer store.Close()

	names := store.Names()
	if len(names) == 0 {
		return nil, errors.New("safetensors: no tensors found")
	}

	return store.Tensor(names[0])
}

// LoadFirstTensorFromBytes decodes a safetensors payload and returns the first
// float32 tensor.
func LoadFirstTensorFromBytes(data []byte) (*Tensor, error) {
	store, err := OpenStoreFromBytes(data, StoreOptions{})
	if err != nil {
		return nil, err
	}
	defer store.Close()

	names := store.Names()
	if len(names) == 0 {
		return nil, errors.New("safetensors: no tensors found")
	}

	return store.Tensor(names[0])
}

// LoadVoiceEmbedding loads a voice embedding from a safetensors file and
// ensures the result has 3D shape [1, T, D]. If the tensor is 2D [T, D],
// it is reshaped to [1, T, D]. Returns the float32 data and the 3D shape.
func LoadVoiceEmbedding(path string) ([]float32, []int64, error) {
	tensor, err := LoadFirstTensor(path)
	if err != nil {
		return nil, nil, err
	}

	return normalizeVoiceEmbeddingShape(tensor)
}

// LoadVoiceEmbeddingFromBytes loads a voice embedding from safetensors payload
// bytes and ensures the result has 3D shape [1, T, D].
func LoadVoiceEmbeddingFromBytes(data []byte) ([]float32, []int64, error) {
	tensor, err := LoadFirstTensorFromBytes(data)
	if err != nil {
		return nil, nil, err
	}

	return normalizeVoiceEmbeddingShape(tensor)
}

// requiredModelKeys is a subset of tensor keys that must be present in a valid
// PocketTTS safetensors model file.
var requiredModelKeys = []string{
	"text_emb.weight",
	"flow_transformer.layers.0.self_attn.q_proj.weight",
	"lsd_decode.net.0.weight",
	"mimi_decode.model.decoder.model.0.conv.conv.weight",
}

// ValidateModelKeys reads only the header of a safetensors file and verifies
// that it contains the expected tensor keys for a PocketTTS model. This is a
// lightweight check that does not load any tensor data.
func ValidateModelKeys(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open %s: %w", path, err)
	}

	defer func() {
		_ = f.Close()
	}()

	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		return fmt.Errorf("read header length: %w", err)
	}

	if headerLen > 100*1024*1024 { // sanity: header should not exceed 100 MB
		return fmt.Errorf("header length %d exceeds 100 MB limit", headerLen)
	}

	headerBuf := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBuf); err != nil {
		return fmt.Errorf("read header: %w", err)
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBuf, &header); err != nil {
		return fmt.Errorf("parse header: %w", err)
	}

	var missing []string

	for _, key := range requiredModelKeys {
		if _, ok := header[key]; !ok {
			missing = append(missing, key)
		}
	}

	if len(missing) > 0 {
		return fmt.Errorf("missing required tensors: %v", missing)
	}

	return nil
}

func normalizeVoiceEmbeddingShape(tensor *Tensor) ([]float32, []int64, error) {
	switch len(tensor.Shape) {
	case 2:
		// [T, D] → [1, T, D]
		shape := []int64{1, tensor.Shape[0], tensor.Shape[1]}
		return tensor.Data, shape, nil
	case 3:
		return tensor.Data, tensor.Shape, nil
	default:
		return nil, nil, fmt.Errorf("safetensors: voice embedding has %dD shape %v, expected 2D or 3D", len(tensor.Shape), tensor.Shape)
	}
}
