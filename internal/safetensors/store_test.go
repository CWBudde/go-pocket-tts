package safetensors

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

func TestStore_TensorByName_F32(t *testing.T) {
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"alpha": {dtype: "F32", shape: []int64{2}, data: float32Bytes([]float32{1, 2})},
		"beta":  {dtype: "F32", shape: []int64{1, 3}, data: float32Bytes([]float32{3, 4, 5})},
	})

	store, err := OpenStoreFromBytes(blob, StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStoreFromBytes: %v", err)
	}
	defer store.Close()

	names := store.Names()
	if strings.Join(names, "|") != "alpha|beta" {
		t.Fatalf("Names() = %v; want [alpha beta]", names)
	}

	tensor, err := store.Tensor("beta")
	if err != nil {
		t.Fatalf("Tensor(beta): %v", err)
	}

	if len(tensor.Shape) != 2 || tensor.Shape[0] != 1 || tensor.Shape[1] != 3 {
		t.Fatalf("beta shape = %v; want [1 3]", tensor.Shape)
	}

	if len(tensor.Data) != 3 || tensor.Data[0] != 3 || tensor.Data[2] != 5 {
		t.Fatalf("beta data = %v; want [3 4 5]", tensor.Data)
	}
}

func TestStore_DTypeConversion_F16AndBF16(t *testing.T) {
	f16Data := float16Bytes([]uint16{0x3c00, 0xc000, 0x3800}) // 1.0, -2.0, 0.5
	bf16Data := bfloat16BytesFromFloat32([]float32{1.0, -2.0, 0.5})

	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"half":  {dtype: "F16", shape: []int64{3}, data: f16Data},
		"bhalf": {dtype: "BF16", shape: []int64{3}, data: bf16Data},
	})

	store, err := OpenStoreFromBytes(blob, StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStoreFromBytes: %v", err)
	}
	defer store.Close()

	half, err := store.Tensor("half")
	if err != nil {
		t.Fatalf("Tensor(half): %v", err)
	}

	assertFloatSliceNear(t, half.Data, []float32{1.0, -2.0, 0.5}, 1e-4)

	bhalf, err := store.Tensor("bhalf")
	if err != nil {
		t.Fatalf("Tensor(bhalf): %v", err)
	}

	assertFloatSliceNear(t, bhalf.Data, []float32{1.0, -2.0, 0.5}, 1e-4)
}

func TestStore_RemapLenientAndStrict(t *testing.T) {
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"model.weight": {dtype: "F32", shape: []int64{1}, data: float32Bytes([]float32{1})},
		"other.bias":   {dtype: "F32", shape: []int64{1}, data: float32Bytes([]float32{2})},
	})

	mapper := func(name string) (string, bool) {
		if after, ok := strings.CutPrefix(name, "model."); ok {
			return after, true
		}

		return "", false
	}

	lenient, err := OpenStoreFromBytes(blob, StoreOptions{
		KeyMapper: mapper,
		RemapMode: RemapLenient,
	})
	if err != nil {
		t.Fatalf("OpenStoreFromBytes lenient: %v", err)
	}
	defer lenient.Close()

	if !lenient.Has("weight") || lenient.Has("other.bias") {
		t.Fatalf("lenient remap names = %v; want [weight]", lenient.Names())
	}

	_, err = OpenStoreFromBytes(blob, StoreOptions{
		KeyMapper: mapper,
		RemapMode: RemapStrict,
	})
	if err == nil {
		t.Fatal("strict remap should fail when key mapper rejects a tensor")
	}
}

func TestStore_StrictRemapCollisionFails(t *testing.T) {
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"a": {dtype: "F32", shape: []int64{1}, data: float32Bytes([]float32{1})},
		"b": {dtype: "F32", shape: []int64{1}, data: float32Bytes([]float32{2})},
	})
	mapper := func(_ string) (string, bool) { return "same", true }

	_, err := OpenStoreFromBytes(blob, StoreOptions{KeyMapper: mapper, RemapMode: RemapStrict})
	if err == nil {
		t.Fatal("strict remap should fail on mapped name collision")
	}
}

func TestStore_TensorWithShapeAndMissingDiagnostics(t *testing.T) {
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"alpha": {dtype: "F32", shape: []int64{2}, data: float32Bytes([]float32{1, 2})},
	})

	store, err := OpenStoreFromBytes(blob, StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStoreFromBytes: %v", err)
	}
	defer store.Close()

	_, err = store.TensorWithShape("alpha", []int64{1, 2})
	if err == nil {
		t.Fatal("TensorWithShape should fail on shape mismatch")
	}

	_, err = store.Tensor("missing")
	if err == nil {
		t.Fatal("Tensor(missing) should fail")
	}

	if !strings.Contains(err.Error(), "available: alpha") {
		t.Fatalf("missing tensor error should include available names, got: %v", err)
	}
}

func TestStore_CorruptionAndUnsupportedDTypeErrors(t *testing.T) {
	// Unsupported dtype.
	unsupported := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"x": {dtype: "I64", shape: []int64{1}, data: make([]byte, 8)},
	})

	_, err := OpenStoreFromBytes(unsupported, StoreOptions{})
	if err == nil {
		t.Fatal("OpenStoreFromBytes should fail for unsupported dtype")
	}

	// Invalid offset range (end < start).
	header := `{"bad":{"dtype":"F32","shape":[1],"data_offsets":[4,2]}}`
	data := make([]byte, 8+len(header)+4)
	binary.LittleEndian.PutUint64(data[:8], uint64(len(header)))
	copy(data[8:], []byte(header))

	_, err = OpenStoreFromBytes(data, StoreOptions{})
	if err == nil {
		t.Fatal("OpenStoreFromBytes should fail for invalid offsets")
	}
}

func TestStore_ReadAll(t *testing.T) {
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"a": {dtype: "F32", shape: []int64{1}, data: float32Bytes([]float32{1})},
		"b": {dtype: "F32", shape: []int64{1}, data: float32Bytes([]float32{2})},
	})

	store, err := OpenStoreFromBytes(blob, StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStoreFromBytes: %v", err)
	}
	defer store.Close()

	all, err := store.ReadAll()
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}

	if len(all) != 2 {
		t.Fatalf("ReadAll len = %d; want 2", len(all))
	}
}

func float16Bytes(bits []uint16) []byte {
	buf := make([]byte, len(bits)*2)
	for i, b := range bits {
		binary.LittleEndian.PutUint16(buf[i*2:], b)
	}

	return buf
}

func bfloat16BytesFromFloat32(vals []float32) []byte {
	buf := make([]byte, len(vals)*2)
	for i, v := range vals {
		bits := math.Float32bits(v)
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(bits>>16))
	}

	return buf
}

func assertFloatSliceNear(t *testing.T, got, want []float32, tol float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
	}

	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > tol {
			t.Fatalf("value[%d]=%v want=%v diff=%v tol=%v", i, got[i], want[i], diff, tol)
		}
	}
}
