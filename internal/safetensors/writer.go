package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
)

// EncodeTensors serializes float32 tensors into safetensors format.
func EncodeTensors(tensors []Tensor) ([]byte, error) {
	if len(tensors) == 0 {
		return nil, errors.New("safetensors: no tensors to encode")
	}

	sorted := make([]Tensor, len(tensors))
	copy(sorted, tensors)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Name < sorted[j].Name
	})

	header := make(map[string]storeHeaderEntry, len(sorted))
	raw := make([]byte, 0, estimateTensorBytes(sorted))

	for _, tensor := range sorted {
		name := strings.TrimSpace(tensor.Name)
		if name == "" {
			return nil, errors.New("safetensors: tensor name must not be empty")
		}

		if _, exists := header[name]; exists {
			return nil, fmt.Errorf("safetensors: duplicate tensor name %q", name)
		}

		elemCount, err := shapeElementCount(tensor.Shape)
		if err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}

		if int64(len(tensor.Data)) != elemCount {
			return nil, fmt.Errorf(
				"safetensors: tensor %q shape %v expects %d elements, got %d",
				name,
				tensor.Shape,
				elemCount,
				len(tensor.Data),
			)
		}

		start := len(raw)

		raw = append(raw, make([]byte, len(tensor.Data)*4)...)
		for i, v := range tensor.Data {
			binary.LittleEndian.PutUint32(raw[start+i*4:], math.Float32bits(v))
		}

		end := len(raw)

		header[name] = storeHeaderEntry{
			DType:   dtypeF32,
			Shape:   append([]int64(nil), tensor.Shape...),
			Offsets: [2]int{start, end},
		}
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return nil, fmt.Errorf("safetensors: encode header: %w", err)
	}

	out := make([]byte, 0, 8+len(headerJSON)+len(raw))
	lenPrefix := make([]byte, 8)
	binary.LittleEndian.PutUint64(lenPrefix, uint64(len(headerJSON)))
	out = append(out, lenPrefix...)
	out = append(out, headerJSON...)
	out = append(out, raw...)

	return out, nil
}

// WriteFile writes float32 tensors into a .safetensors file.
func WriteFile(path string, tensors []Tensor) error {
	data, err := EncodeTensors(tensors)
	if err != nil {
		return err
	}

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("safetensors: write %s: %w", path, err)
	}

	return nil
}

func estimateTensorBytes(tensors []Tensor) int {
	total := 0
	for _, tensor := range tensors {
		total += len(tensor.Data) * 4
	}

	return total
}
