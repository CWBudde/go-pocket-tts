package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Tensor holds a single tensor loaded from a safetensors file.
type Tensor struct {
	Name  string
	Shape []int64
	Data  []float32
}

// headerEntry is the JSON representation of a tensor in the safetensors header.
type headerEntry struct {
	DType   string  `json:"dtype"`
	Shape   []int64 `json:"shape"`
	Offsets [2]int  `json:"data_offsets"`
}

// LoadFirstTensor reads a safetensors file and returns the first float32 tensor.
// The safetensors format is: 8-byte LE header length → JSON header → raw tensor data.
func LoadFirstTensor(path string) (*Tensor, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: read %s: %w", path, err)
	}

	if len(data) < 8 {
		return nil, fmt.Errorf("safetensors: file too short (%d bytes)", len(data))
	}

	headerLen := binary.LittleEndian.Uint64(data[:8])
	headerEnd := 8 + int(headerLen)
	if headerEnd > len(data) {
		return nil, fmt.Errorf("safetensors: header length %d exceeds file size %d", headerLen, len(data))
	}

	var header map[string]headerEntry
	if err := json.Unmarshal(data[8:headerEnd], &header); err != nil {
		return nil, fmt.Errorf("safetensors: parse header: %w", err)
	}

	// Remove the __metadata__ key if present (it's not a tensor).
	delete(header, "__metadata__")

	if len(header) == 0 {
		return nil, fmt.Errorf("safetensors: no tensors found in %s", path)
	}

	// Pick the first tensor.
	var name string
	var entry headerEntry
	for name, entry = range header {
		break
	}

	if entry.DType != "F32" {
		return nil, fmt.Errorf("safetensors: tensor %q has unsupported dtype %q (only F32 supported)", name, entry.DType)
	}

	// Validate data offsets.
	dataStart := headerEnd + entry.Offsets[0]
	dataEnd := headerEnd + entry.Offsets[1]
	if dataEnd > len(data) {
		return nil, fmt.Errorf("safetensors: tensor %q data [%d:%d] exceeds file size %d", name, dataStart, dataEnd, len(data))
	}

	// Compute expected element count from shape.
	numElements := int64(1)
	for _, dim := range entry.Shape {
		numElements *= dim
	}
	expectedBytes := int(numElements) * 4
	actualBytes := dataEnd - dataStart
	if actualBytes < expectedBytes {
		return nil, fmt.Errorf("safetensors: tensor %q needs %d bytes but data has %d", name, expectedBytes, actualBytes)
	}

	// Convert raw bytes to float32 slice.
	floats := make([]float32, numElements)
	raw := data[dataStart:dataEnd]
	for i := range floats {
		floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}

	return &Tensor{
		Name:  name,
		Shape: entry.Shape,
		Data:  floats,
	}, nil
}

// LoadVoiceEmbedding loads a voice embedding from a safetensors file and
// ensures the result has 3D shape [1, T, D]. If the tensor is 2D [T, D],
// it is reshaped to [1, T, D]. Returns the float32 data and the 3D shape.
func LoadVoiceEmbedding(path string) ([]float32, []int64, error) {
	tensor, err := LoadFirstTensor(path)
	if err != nil {
		return nil, nil, err
	}

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
