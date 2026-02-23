package safetensors

import (
	"fmt"
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
		return nil, fmt.Errorf("safetensors: no tensors found")
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
		return nil, fmt.Errorf("safetensors: no tensors found")
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
