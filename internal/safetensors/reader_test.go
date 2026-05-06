package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Helpers to build synthetic .safetensors files for testing
// ---------------------------------------------------------------------------

// tensorMeta describes a single tensor in the safetensors header.
type tensorMeta struct {
	DType   string  `json:"dtype"`
	Shape   []int64 `json:"shape"`
	Offsets [2]int  `json:"data_offsets"`
}

// buildSafetensors creates a valid .safetensors binary blob with the given
// tensors. Each entry in tensors maps tensor name → (dtype, shape, raw bytes).
func buildSafetensors(t *testing.T, tensors map[string]struct {
	dtype string
	shape []int64
	data  []byte
},
) []byte {
	t.Helper()

	// Build header and compute offsets.
	header := make(map[string]tensorMeta)

	totalDataLen := 0
	for _, info := range tensors {
		totalDataLen += len(info.data)
	}

	rawData := make([]byte, 0, totalDataLen)
	for name, info := range tensors {
		start := len(rawData)
		rawData = append(rawData, info.data...)
		header[name] = tensorMeta{
			DType:   info.dtype,
			Shape:   info.shape,
			Offsets: [2]int{start, start + len(info.data)},
		}
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}

	// 8-byte LE header length + JSON header + tensor data.
	lenBuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(lenBuf, uint64(len(headerJSON)))
	buf := make([]byte, 0, len(lenBuf)+len(headerJSON)+len(rawData))
	buf = append(buf, lenBuf...)
	buf = append(buf, headerJSON...)
	buf = append(buf, rawData...)

	return buf
}

// float32Bytes converts a float32 slice to little-endian bytes.
func float32Bytes(vals []float32) []byte {
	buf := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}

	return buf
}

func int64Bytes(vals []int64) []byte {
	buf := make([]byte, len(vals)*8)
	for i, v := range vals {
		binary.LittleEndian.PutUint64(buf[i*8:], uint64(v))
	}

	return buf
}

func equalInt64s(a, b []int64) bool {
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

// writeTempSafetensors writes raw bytes to a temp file and returns the path.
func writeTempSafetensors(t *testing.T, data []byte) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "test.safetensors")

	err := os.WriteFile(path, data, 0o644)
	if err != nil {
		t.Fatalf("write temp safetensors: %v", err)
	}

	return path
}

// ---------------------------------------------------------------------------
// Tests for LoadFirstTensor
// ---------------------------------------------------------------------------

func TestLoadFirstTensor_Float32_2D(t *testing.T) {
	// 2D tensor [2, 3] with known values.
	vals := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"voice_emb": {dtype: "F32", shape: []int64{2, 3}, data: float32Bytes(vals)},
	})

	path := writeTempSafetensors(t, blob)

	tensor, err := LoadFirstTensor(path)
	if err != nil {
		t.Fatalf("LoadFirstTensor: %v", err)
	}

	if tensor.Name != "voice_emb" {
		t.Errorf("Name = %q, want %q", tensor.Name, "voice_emb")
	}

	if len(tensor.Shape) != 2 || tensor.Shape[0] != 2 || tensor.Shape[1] != 3 {
		t.Errorf("Shape = %v, want [2 3]", tensor.Shape)
	}

	if len(tensor.Data) != 6 {
		t.Fatalf("Data length = %d, want 6", len(tensor.Data))
	}

	for i, v := range vals {
		if tensor.Data[i] != v {
			t.Errorf("Data[%d] = %v, want %v", i, tensor.Data[i], v)
		}
	}
}

func TestLoadFirstTensor_Float32_3D(t *testing.T) {
	// 3D tensor [1, 2, 4] with known values.
	vals := make([]float32, 8)
	for i := range vals {
		vals[i] = float32(i) * 0.5
	}

	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"embedding": {dtype: "F32", shape: []int64{1, 2, 4}, data: float32Bytes(vals)},
	})

	path := writeTempSafetensors(t, blob)

	tensor, err := LoadFirstTensor(path)
	if err != nil {
		t.Fatalf("LoadFirstTensor: %v", err)
	}

	if len(tensor.Shape) != 3 || tensor.Shape[0] != 1 || tensor.Shape[1] != 2 || tensor.Shape[2] != 4 {
		t.Errorf("Shape = %v, want [1 2 4]", tensor.Shape)
	}

	if len(tensor.Data) != 8 {
		t.Fatalf("Data length = %d, want 8", len(tensor.Data))
	}
}

func TestLoadFirstTensor_MultiTensor_ReturnsFirst(t *testing.T) {
	// File with two tensors — should return one of them without error.
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"alpha": {dtype: "F32", shape: []int64{1, 2}, data: float32Bytes([]float32{1.0, 2.0})},
		"beta":  {dtype: "F32", shape: []int64{1, 3}, data: float32Bytes([]float32{3.0, 4.0, 5.0})},
	})

	path := writeTempSafetensors(t, blob)

	tensor, err := LoadFirstTensor(path)
	if err != nil {
		t.Fatalf("LoadFirstTensor: %v", err)
	}

	// Should return one of the tensors (map iteration order is non-deterministic).
	if tensor.Name != "alpha" && tensor.Name != "beta" {
		t.Errorf("Name = %q, want alpha or beta", tensor.Name)
	}
}

func TestLoadFirstTensor_EmptyFile(t *testing.T) {
	path := writeTempSafetensors(t, []byte{})

	_, err := LoadFirstTensor(path)
	if err == nil {
		t.Fatal("expected error for empty file")
	}
}

func TestLoadFirstTensor_TruncatedHeader(t *testing.T) {
	// Only 4 bytes — not enough for the 8-byte length prefix.
	path := writeTempSafetensors(t, []byte{0, 0, 0, 0})

	_, err := LoadFirstTensor(path)
	if err == nil {
		t.Fatal("expected error for truncated header")
	}
}

func TestLoadFirstTensor_NoTensors(t *testing.T) {
	// Valid header but empty tensor map.
	headerJSON := []byte("{}")
	lenBuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(lenBuf, uint64(len(headerJSON)))
	buf := make([]byte, 0, len(lenBuf)+len(headerJSON))
	buf = append(buf, lenBuf...)
	buf = append(buf, headerJSON...)

	path := writeTempSafetensors(t, buf)

	_, err := LoadFirstTensor(path)
	if err == nil {
		t.Fatal("expected error for file with no tensors")
	}
}

func TestLoadFirstTensor_UnsupportedDtype(t *testing.T) {
	data := make([]byte, 4)
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"tensor": {dtype: "U32", shape: []int64{1}, data: data},
	})

	path := writeTempSafetensors(t, blob)

	_, err := LoadFirstTensor(path)
	if err == nil {
		t.Fatal("expected error for unsupported dtype U32")
	}
}

func TestLoadFirstTensor_FileNotFound(t *testing.T) {
	_, err := LoadFirstTensor("/nonexistent/path/voice.safetensors")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadFirstTensor_InvalidJSON(t *testing.T) {
	// Valid length prefix but invalid JSON.
	headerJSON := []byte("{invalid json")
	lenBuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(lenBuf, uint64(len(headerJSON)))
	buf := make([]byte, 0, len(lenBuf)+len(headerJSON))
	buf = append(buf, lenBuf...)
	buf = append(buf, headerJSON...)

	path := writeTempSafetensors(t, buf)

	_, err := LoadFirstTensor(path)
	if err == nil {
		t.Fatal("expected error for invalid JSON header")
	}
}

func TestLoadFirstTensor_DataTruncated(t *testing.T) {
	// Header says tensor is 12 bytes but file has only 4 bytes of data.
	vals := []float32{1.0}
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"tensor": {dtype: "F32", shape: []int64{1, 3}, data: float32Bytes(vals)}, // shape says 3 floats = 12 bytes, but only 4 bytes provided
	})

	// Manually truncate the data section.
	path := writeTempSafetensors(t, blob[:len(blob)-8]) // remove 8 bytes from end

	_, err := LoadFirstTensor(path)
	if err == nil {
		t.Fatal("expected error for truncated tensor data")
	}
}

// ---------------------------------------------------------------------------
// Tests for LoadVoiceEmbedding (2D→3D reshape)
// ---------------------------------------------------------------------------

func TestLoadVoiceEmbedding_2D_ReshapesTo3D(t *testing.T) {
	// 2D tensor [3, 1024] → should reshape to [1, 3, 1024].
	vals := make([]float32, 3*1024)
	for i := range vals {
		vals[i] = float32(i) * 0.001
	}

	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"voice": {dtype: "F32", shape: []int64{3, 1024}, data: float32Bytes(vals)},
	})

	path := writeTempSafetensors(t, blob)

	data, shape, err := LoadVoiceEmbedding(path)
	if err != nil {
		t.Fatalf("LoadVoiceEmbedding: %v", err)
	}

	if len(shape) != 3 || shape[0] != 1 || shape[1] != 3 || shape[2] != 1024 {
		t.Errorf("shape = %v, want [1 3 1024]", shape)
	}

	if len(data) != 3*1024 {
		t.Fatalf("data length = %d, want %d", len(data), 3*1024)
	}
	// Data values should be preserved.
	if data[0] != vals[0] || data[len(data)-1] != vals[len(vals)-1] {
		t.Error("data values not preserved after reshape")
	}
}

func TestLoadVoiceEmbedding_3D_PassesThrough(t *testing.T) {
	// Already 3D [1, 2, 1024] → should pass through unchanged.
	vals := make([]float32, 2*1024)
	for i := range vals {
		vals[i] = float32(i) * 0.01
	}

	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"voice": {dtype: "F32", shape: []int64{1, 2, 1024}, data: float32Bytes(vals)},
	})

	path := writeTempSafetensors(t, blob)

	data, shape, err := LoadVoiceEmbedding(path)
	if err != nil {
		t.Fatalf("LoadVoiceEmbedding: %v", err)
	}

	if len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != 1024 {
		t.Errorf("shape = %v, want [1 2 1024]", shape)
	}

	if len(data) != 2*1024 {
		t.Fatalf("data length = %d, want %d", len(data), 2*1024)
	}
}

func TestInspectVoiceFile_ModelState(t *testing.T) {
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"transformer.layers.0.self_attn/cache": {
			dtype: "F32",
			shape: []int64{2, 1, 2, 1, 1},
			data:  float32Bytes([]float32{1, 2, 3, 4}),
		},
		"transformer.layers.0.self_attn/offset": {
			dtype: "I64",
			shape: []int64{1},
			data:  int64Bytes([]int64{2}),
		},
	})

	path := writeTempSafetensors(t, blob)

	kind, err := InspectVoiceFile(path)
	if err != nil {
		t.Fatalf("InspectVoiceFile: %v", err)
	}

	if kind != VoiceFileModelState {
		t.Fatalf("kind = %q, want %q", kind, VoiceFileModelState)
	}

	state, err := LoadVoiceModelState(path)
	if err != nil {
		t.Fatalf("LoadVoiceModelState: %v", err)
	}

	module := state.Modules["transformer.layers.0.self_attn"]
	if module == nil {
		t.Fatal("expected module state for transformer layer")
	}

	if got := module["offset"].Data; len(got) != 1 || got[0] != 2 {
		t.Fatalf("offset data = %v, want [2]", got)
	}

	if _, _, err := LoadVoiceEmbedding(path); err == nil || !strings.Contains(err.Error(), "model state") {
		t.Fatalf("LoadVoiceEmbedding(model state) err = %v, want model-state error", err)
	}
}

func TestLoadVoiceModelState_LegacyCurrentEndBecomesOffset(t *testing.T) {
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"layer/cache": {
			dtype: "F32",
			shape: []int64{2, 1, 3, 1, 1},
			data:  float32Bytes(make([]float32, 6)),
		},
		"layer/current_end": {
			dtype: "F32",
			shape: []int64{3},
			data:  float32Bytes([]float32{0, 0, 0}),
		},
	})

	state, err := LoadVoiceModelState(writeTempSafetensors(t, blob))
	if err != nil {
		t.Fatalf("LoadVoiceModelState: %v", err)
	}

	offset := state.Modules["layer"]["offset"]
	if offset == nil {
		t.Fatal("expected current_end to be imported as offset")
	}

	if !equalInt64s(offset.Shape, []int64{1}) || len(offset.Data) != 1 || offset.Data[0] != 3 {
		t.Fatalf("offset tensor = shape %v data %v, want shape [1] data [3]", offset.Shape, offset.Data)
	}
}

func TestLoadVoiceEmbedding_1D_ReturnsError(t *testing.T) {
	vals := []float32{1.0, 2.0, 3.0}
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"voice": {dtype: "F32", shape: []int64{3}, data: float32Bytes(vals)},
	})

	path := writeTempSafetensors(t, blob)

	_, _, err := LoadVoiceEmbedding(path)
	if err == nil {
		t.Fatal("expected error for 1D tensor")
	}
}

func TestLoadVoiceEmbedding_4D_ReturnsError(t *testing.T) {
	vals := make([]float32, 24)
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"voice": {dtype: "F32", shape: []int64{1, 2, 3, 4}, data: float32Bytes(vals)},
	})

	path := writeTempSafetensors(t, blob)

	_, _, err := LoadVoiceEmbedding(path)
	if err == nil {
		t.Fatal("expected error for 4D tensor")
	}
}

// ---------------------------------------------------------------------------
// Task 19.4 additional tests
// ---------------------------------------------------------------------------

// TestLoadVoiceEmbedding_DataValuesPreserved verifies that the exact float32
// values survive the safetensors round-trip without corruption.
func TestLoadVoiceEmbedding_DataValuesPreserved(t *testing.T) {
	// Use a small [2, 4] tensor with recognisable, non-trivial values.
	vals := []float32{1.5, -0.25, 3.14159, 0.0, -1.0, 42.0, 0.001, -999.9}
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"voice": {dtype: "F32", shape: []int64{2, 4}, data: float32Bytes(vals)},
	})

	path := writeTempSafetensors(t, blob)

	data, shape, err := LoadVoiceEmbedding(path)
	if err != nil {
		t.Fatalf("LoadVoiceEmbedding: %v", err)
	}

	// Shape should be reshaped from [2, 4] → [1, 2, 4].
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != 4 {
		t.Errorf("shape = %v, want [1 2 4]", shape)
	}

	// Each value must be bit-exact.
	if len(data) != len(vals) {
		t.Fatalf("data length = %d, want %d", len(data), len(vals))
	}

	for i, want := range vals {
		if data[i] != want {
			t.Errorf("data[%d] = %v, want %v", i, data[i], want)
		}
	}
}

// TestLoadFirstTensor_MetadataKeyIgnored verifies that the special
// __metadata__ entry in the safetensors header is stripped and does not
// prevent loading the real tensor.
func TestLoadFirstTensor_MetadataKeyIgnored(t *testing.T) {
	vals := []float32{1.0, 2.0, 3.0}
	// Build raw tensor data.
	rawData := float32Bytes(vals)

	// Build header JSON manually so we can include the __metadata__ key
	// alongside the real tensor.
	headerMap := map[string]any{
		"__metadata__": map[string]any{
			"format": "pt",
		},
		"voice_emb": map[string]any{
			"dtype":        "F32",
			"shape":        []int{1, 3},
			"data_offsets": []int{0, len(rawData)},
		},
	}

	headerJSON, err := json.Marshal(headerMap)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}

	lenBuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(lenBuf, uint64(len(headerJSON)))
	buf := make([]byte, 0, len(lenBuf)+len(headerJSON)+len(rawData))
	buf = append(buf, lenBuf...)
	buf = append(buf, headerJSON...)
	buf = append(buf, rawData...)

	path := writeTempSafetensors(t, buf)

	tensor, err := LoadFirstTensor(path)
	if err != nil {
		t.Fatalf("LoadFirstTensor with __metadata__: %v", err)
	}

	if tensor.Name != "voice_emb" {
		t.Errorf("tensor name = %q, want %q", tensor.Name, "voice_emb")
	}

	if len(tensor.Data) != 3 {
		t.Fatalf("data length = %d, want 3", len(tensor.Data))
	}

	for i, want := range vals {
		if tensor.Data[i] != want {
			t.Errorf("data[%d] = %v, want %v", i, tensor.Data[i], want)
		}
	}
}

// ---------------------------------------------------------------------------
// ValidateModelKeys
// ---------------------------------------------------------------------------

func TestValidateModelKeys_AllKeysPresent(t *testing.T) {
	// Build a file that contains all required keys.
	tensors := make(map[string]struct {
		dtype string
		shape []int64
		data  []byte
	})

	dummyData := float32Bytes([]float32{1.0})
	for _, key := range requiredModelKeys {
		tensors[key] = struct {
			dtype string
			shape []int64
			data  []byte
		}{"F32", []int64{1}, dummyData}
	}

	data := buildSafetensors(t, tensors)
	path := writeTempSafetensors(t, data)

	err := ValidateModelKeys(path)
	if err != nil {
		t.Errorf("ValidateModelKeys should pass; got: %v", err)
	}
}

func TestValidateModelKeys_MissingKey(t *testing.T) {
	// Build a file that only has some of the required keys.
	tensors := map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"text_emb.weight": {"F32", []int64{1}, float32Bytes([]float32{1.0})},
	}
	data := buildSafetensors(t, tensors)
	path := writeTempSafetensors(t, data)

	err := ValidateModelKeys(path)
	if err == nil {
		t.Fatal("expected error for missing keys")
	}

	if !strings.Contains(err.Error(), "missing required tensors") {
		t.Errorf("error should mention missing tensors; got: %v", err)
	}
}

func TestValidateModelKeys_FileNotFound(t *testing.T) {
	err := ValidateModelKeys("/nonexistent/model.safetensors")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestValidateModelKeys_InvalidFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.safetensors")

	err := os.WriteFile(path, []byte("not a safetensors file"), 0o644)
	if err != nil {
		t.Fatal(err)
	}

	err = ValidateModelKeys(path)
	if err == nil {
		t.Fatal("expected error for invalid file")
	}
}
