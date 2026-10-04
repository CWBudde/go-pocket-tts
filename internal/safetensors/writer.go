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

// EncodeTensors serializes tensors into safetensors format. A tensor is
// written as F32 unless its DType is I64; I64 data must hold integers.
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

		dtype, err := appendTensorData(&raw, tensor)
		if err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}

		header[name] = storeHeaderEntry{
			DType:   dtype,
			Shape:   append([]int64(nil), tensor.Shape...),
			Offsets: [2]int{start, len(raw)},
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

// appendTensorData appends the little-endian data of tensor to raw and
// returns the dtype it was written as.
func appendTensorData(raw *[]byte, tensor Tensor) (string, error) {
	start := len(*raw)

	switch dtype := strings.ToUpper(tensor.DType); dtype {
	case "", dtypeF32:
		*raw = append(*raw, make([]byte, len(tensor.Data)*4)...)
		for i, v := range tensor.Data {
			binary.LittleEndian.PutUint32((*raw)[start+i*4:], math.Float32bits(v))
		}

		return dtypeF32, nil
	case dtypeI64:
		*raw = append(*raw, make([]byte, len(tensor.Data)*8)...)
		for i, v := range tensor.Data {
			// float32 holds every integer up to 2^24 exactly; the
			// int64 range check also rejects ±Inf.
			f := float64(v)
			if f != math.Trunc(f) || f < math.MinInt64 || f >= math.MaxInt64 {
				return "", fmt.Errorf("I64 element %d is %v, not an integer", i, v)
			}

			binary.LittleEndian.PutUint64((*raw)[start+i*8:], uint64(int64(f)))
		}

		return dtypeI64, nil
	default:
		return "", fmt.Errorf("cannot write dtype %q (supported: %s, %s)", tensor.DType, dtypeF32, dtypeI64)
	}
}

// EncodeVoiceModelState serializes a voice model state the way upstream
// export_model_state does: one tensor per module key, named module/key. Module
// names and keys must not contain "/", which upstream splits on.
func EncodeVoiceModelState(state *VoiceModelState) ([]byte, error) {
	if state == nil || len(state.Modules) == 0 {
		return nil, errors.New("safetensors: voice model state is empty")
	}

	var tensors []Tensor

	for module, keys := range state.Modules {
		if module == "" || strings.Contains(module, "/") {
			return nil, fmt.Errorf("safetensors: invalid voice model state module name %q", module)
		}

		for key, t := range keys {
			if key == "" || strings.Contains(key, "/") {
				return nil, fmt.Errorf("safetensors: invalid voice model state key %q in module %q", key, module)
			}

			if t == nil {
				return nil, fmt.Errorf("safetensors: voice model state %s/%s is nil", module, key)
			}

			out := *t
			out.Name = module + "/" + key
			tensors = append(tensors, out)
		}
	}

	return EncodeTensors(tensors)
}

// WriteVoiceModelState writes state into a .safetensors file that upstream
// pocket-tts and LoadVoiceModelState read; see EncodeVoiceModelState.
func WriteVoiceModelState(path string, state *VoiceModelState) error {
	data, err := EncodeVoiceModelState(state)
	if err != nil {
		return err
	}

	err = os.WriteFile(path, data, 0o600)
	if err != nil {
		return fmt.Errorf("safetensors: write %s: %w", path, err)
	}

	return nil
}

// WriteFile writes tensors into a .safetensors file; see EncodeTensors.
func WriteFile(path string, tensors []Tensor) error {
	data, err := EncodeTensors(tensors)
	if err != nil {
		return err
	}

	err = os.WriteFile(path, data, 0o600)
	if err != nil {
		return fmt.Errorf("safetensors: write %s: %w", path, err)
	}

	return nil
}

func estimateTensorBytes(tensors []Tensor) int {
	total := 0

	for _, tensor := range tensors {
		width := 4
		if strings.EqualFold(tensor.DType, dtypeI64) {
			width = 8
		}

		total += len(tensor.Data) * width
	}

	return total
}
