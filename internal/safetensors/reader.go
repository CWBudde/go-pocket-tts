package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

// Tensor holds a single tensor loaded from a safetensors file.
type Tensor struct {
	Name  string
	Shape []int64
	Data  []float32
}

type VoiceFileKind string

const (
	VoiceFileUnknown    VoiceFileKind = "unknown"
	VoiceFileEmbedding  VoiceFileKind = "embedding"
	VoiceFileModelState VoiceFileKind = "model_state"
)

type VoiceModelState struct {
	Modules map[string]map[string]*Tensor
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
	kind, err := InspectVoiceFile(path)
	if err != nil {
		return nil, nil, err
	}

	if kind == VoiceFileModelState {
		return nil, nil, errors.New("safetensors: voice file contains upstream model state, not a legacy audio_prompt embedding")
	}

	tensor, err := LoadFirstTensor(path)
	if err != nil {
		return nil, nil, err
	}

	return normalizeVoiceEmbeddingShape(tensor)
}

// LoadVoiceEmbeddingFromBytes loads a voice embedding from safetensors payload
// bytes and ensures the result has 3D shape [1, T, D].
func LoadVoiceEmbeddingFromBytes(data []byte) ([]float32, []int64, error) {
	kind, err := InspectVoiceFileBytes(data)
	if err != nil {
		return nil, nil, err
	}

	if kind == VoiceFileModelState {
		return nil, nil, errors.New("safetensors: voice file contains upstream model state, not a legacy audio_prompt embedding")
	}

	tensor, err := LoadFirstTensorFromBytes(data)
	if err != nil {
		return nil, nil, err
	}

	return normalizeVoiceEmbeddingShape(tensor)
}

func InspectVoiceFile(path string) (VoiceFileKind, error) {
	store, err := OpenStore(path, StoreOptions{})
	if err != nil {
		return VoiceFileUnknown, err
	}
	defer store.Close()

	return classifyVoiceTensorNames(store.Names()), nil
}

func InspectVoiceFileBytes(data []byte) (VoiceFileKind, error) {
	store, err := OpenStoreFromBytes(data, StoreOptions{})
	if err != nil {
		return VoiceFileUnknown, err
	}
	defer store.Close()

	return classifyVoiceTensorNames(store.Names()), nil
}

func LoadVoiceModelState(path string) (*VoiceModelState, error) {
	store, err := OpenStore(path, StoreOptions{})
	if err != nil {
		return nil, err
	}
	defer store.Close()

	kind := classifyVoiceTensorNames(store.Names())
	if kind != VoiceFileModelState {
		return nil, fmt.Errorf("safetensors: voice file kind %q is not upstream model state", kind)
	}

	return loadVoiceModelStateFromStore(store)
}

func LoadVoiceModelStateFromBytes(data []byte) (*VoiceModelState, error) {
	store, err := OpenStoreFromBytes(data, StoreOptions{})
	if err != nil {
		return nil, err
	}
	defer store.Close()

	kind := classifyVoiceTensorNames(store.Names())
	if kind != VoiceFileModelState {
		return nil, fmt.Errorf("safetensors: voice file kind %q is not upstream model state", kind)
	}

	return loadVoiceModelStateFromStore(store)
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

	err = binary.Read(f, binary.LittleEndian, &headerLen)
	if err != nil {
		return fmt.Errorf("read header length: %w", err)
	}

	if headerLen > 100*1024*1024 { // sanity: header should not exceed 100 MB
		return fmt.Errorf("header length %d exceeds 100 MB limit", headerLen)
	}

	headerBuf := make([]byte, headerLen)

	_, err = io.ReadFull(f, headerBuf)
	if err != nil {
		return fmt.Errorf("read header: %w", err)
	}

	var header map[string]json.RawMessage

	err = json.Unmarshal(headerBuf, &header)
	if err != nil {
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

func classifyVoiceTensorNames(names []string) VoiceFileKind {
	hasAudioPrompt := false
	hasModelState := false

	for _, name := range names {
		if name == "audio_prompt" {
			hasAudioPrompt = true
			continue
		}

		if isModelStateTensorName(name) {
			hasModelState = true
		}
	}

	if hasModelState {
		return VoiceFileModelState
	}

	if hasAudioPrompt || len(names) > 0 {
		return VoiceFileEmbedding
	}

	return VoiceFileUnknown
}

func isModelStateTensorName(name string) bool {
	slash := strings.LastIndex(name, "/")
	if slash <= 0 || slash == len(name)-1 {
		return false
	}

	tensorKey := name[slash+1:]
	switch tensorKey {
	case "cache", "offset", "current_end":
		return true
	default:
		return false
	}
}

func loadVoiceModelStateFromStore(store *Store) (*VoiceModelState, error) {
	state := &VoiceModelState{Modules: make(map[string]map[string]*Tensor)}
	for _, name := range store.Names() {
		slash := strings.LastIndex(name, "/")
		if slash <= 0 || slash == len(name)-1 {
			return nil, fmt.Errorf("safetensors: invalid model-state tensor name %q", name)
		}

		moduleName := name[:slash]
		tensorKey := name[slash+1:]

		t, err := store.Tensor(name)
		if err != nil {
			return nil, err
		}

		if tensorKey == "current_end" {
			tensorKey = "offset"
			t = &Tensor{
				Name:  moduleName + "/offset",
				Shape: []int64{1},
				Data:  []float32{float32(firstDim(t.Shape))},
			}
		}

		module := state.Modules[moduleName]
		if module == nil {
			module = make(map[string]*Tensor)
			state.Modules[moduleName] = module
		}

		module[tensorKey] = t
	}

	return state, nil
}

func firstDim(shape []int64) int64 {
	if len(shape) == 0 {
		return 0
	}

	return shape[0]
}
