package safetensors

import (
	"encoding/binary"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestWriteFile_RoundTripSingleTensor(t *testing.T) {
	path := filepath.Join(t.TempDir(), "voice.safetensors")

	want := Tensor{
		Name:  "audio_prompt",
		Shape: []int64{1, 2, 4},
		Data:  []float32{1.5, -0.25, 3.25, 4.0, -1.0, 0.5, 2.5, 9.0},
	}

	err := WriteFile(path, []Tensor{want})
	if err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	got, err := LoadFirstTensor(path)
	if err != nil {
		t.Fatalf("LoadFirstTensor: %v", err)
	}

	if got.Name != want.Name {
		t.Fatalf("tensor name = %q, want %q", got.Name, want.Name)
	}

	if len(got.Shape) != len(want.Shape) || got.Shape[0] != 1 || got.Shape[1] != 2 || got.Shape[2] != 4 {
		t.Fatalf("tensor shape = %v, want %v", got.Shape, want.Shape)
	}

	if len(got.Data) != len(want.Data) {
		t.Fatalf("tensor data length = %d, want %d", len(got.Data), len(want.Data))
	}

	for i := range got.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("data[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestEncodeTensors_MultipleRoundTrip(t *testing.T) {
	blob, err := EncodeTensors([]Tensor{
		{Name: "b", Shape: []int64{2}, Data: []float32{3, 4}},
		{Name: "a", Shape: []int64{1, 2}, Data: []float32{1, 2}},
	})
	if err != nil {
		t.Fatalf("EncodeTensors: %v", err)
	}

	store, err := OpenStoreFromBytes(blob, StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStoreFromBytes: %v", err)
	}
	defer store.Close()

	names := store.Names()
	if len(names) != 2 || names[0] != "a" || names[1] != "b" {
		t.Fatalf("Names() = %v, want [a b]", names)
	}
}

func TestEncodeTensors_ValidationErrors(t *testing.T) {
	_, err := EncodeTensors(nil)
	if err == nil {
		t.Fatal("EncodeTensors(nil) should fail")
	}

	_, err = EncodeTensors([]Tensor{{Name: "", Shape: []int64{1}, Data: []float32{1}}})
	if err == nil {
		t.Fatal("empty tensor name should fail")
	}

	_, err = EncodeTensors([]Tensor{
		{Name: "x", Shape: []int64{1}, Data: []float32{1}},
		{Name: "x", Shape: []int64{1}, Data: []float32{2}},
	})
	if err == nil {
		t.Fatal("duplicate tensor names should fail")
	}

	_, err = EncodeTensors([]Tensor{{Name: "x", Shape: []int64{1, 2}, Data: []float32{1}}})
	if err == nil {
		t.Fatal("shape/data mismatch should fail")
	}
}

// TestEncodeTensors_I64 checks that an I64 tensor is written with 8-byte
// little-endian integers and comes back with its dtype.
func TestEncodeTensors_I64(t *testing.T) {
	blob, err := EncodeTensors([]Tensor{
		{Name: "offset", DType: "I64", Shape: []int64{2}, Data: []float32{124, -3}},
		{Name: "x", Shape: []int64{1}, Data: []float32{1.5}},
	})
	if err != nil {
		t.Fatalf("EncodeTensors: %v", err)
	}

	headerEnd, header, err := decodeHeader(blob)
	if err != nil {
		t.Fatalf("decodeHeader: %v", err)
	}

	entry, err := parseHeaderEntry(header["offset"])
	if err != nil {
		t.Fatalf("parse offset entry: %v", err)
	}

	if entry.DType != "I64" || entry.Offsets[1]-entry.Offsets[0] != 16 {
		t.Fatalf("offset entry = %+v, want I64 with 16 data bytes", entry)
	}

	raw := blob[headerEnd+entry.Offsets[0]:]
	if a, b := int64(binary.LittleEndian.Uint64(raw)), int64(binary.LittleEndian.Uint64(raw[8:])); a != 124 || b != -3 {
		t.Fatalf("offset data = [%d %d], want [124 -3]", a, b)
	}

	store, err := OpenStoreFromBytes(blob, StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStoreFromBytes: %v", err)
	}
	defer store.Close()

	for _, tc := range []struct {
		name  string
		dtype string
		data  []float32
	}{
		{name: "offset", dtype: "I64", data: []float32{124, -3}},
		{name: "x", dtype: "F32", data: []float32{1.5}},
	} {
		got, err := store.Tensor(tc.name)
		if err != nil {
			t.Fatalf("Tensor(%s): %v", tc.name, err)
		}

		if got.DType != tc.dtype || !slices.Equal(got.Data, tc.data) {
			t.Errorf("%s = %s %v, want %s %v", tc.name, got.DType, got.Data, tc.dtype, tc.data)
		}
	}
}

func TestEncodeTensors_DTypeErrors(t *testing.T) {
	for _, tc := range []struct {
		name   string
		tensor Tensor
		want   string
	}{
		{name: "fractional I64", tensor: Tensor{Name: "x", DType: "I64", Shape: []int64{1}, Data: []float32{1.5}}, want: "integer"},
		{name: "NaN I64", tensor: Tensor{Name: "x", DType: "I64", Shape: []int64{1}, Data: []float32{float32(math.NaN())}}, want: "integer"},
		{name: "F16", tensor: Tensor{Name: "x", DType: "F16", Shape: []int64{1}, Data: []float32{1}}, want: "F16"},
	} {
		_, err := EncodeTensors([]Tensor{tc.tensor})
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want one mentioning %q", tc.name, err, tc.want)
		}
	}
}

// TestWriteVoiceModelState_RoundTrip writes a state the way upstream
// export_model_state does (module/key names, int64 offset and pad) and loads
// it back.
func TestWriteVoiceModelState_RoundTrip(t *testing.T) {
	const module = "transformer.layers.0.self_attn"

	state := &VoiceModelState{Modules: map[string]map[string]*Tensor{
		module: {
			"cache":  {Shape: []int64{2, 1, 2, 1, 2}, Data: []float32{1, 2, 3, 4, 5, 6, 7, 8}},
			"offset": {DType: "I64", Shape: []int64{1}, Data: []float32{2}},
			"pad":    {DType: "I64", Shape: []int64{1}, Data: []float32{0}},
		},
	}}

	path := filepath.Join(t.TempDir(), "voice.safetensors")

	err := WriteVoiceModelState(path, state)
	if err != nil {
		t.Fatalf("WriteVoiceModelState: %v", err)
	}

	kind, err := InspectVoiceFile(path)
	if err != nil || kind != VoiceFileModelState {
		t.Fatalf("InspectVoiceFile = %q, %v; want %q", kind, err, VoiceFileModelState)
	}

	store, err := OpenStore(path, StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer store.Close()

	wantNames := []string{module + "/cache", module + "/offset", module + "/pad"}
	if got := store.Names(); !slices.Equal(got, wantNames) {
		t.Fatalf("names = %v, want %v", got, wantNames)
	}

	got, err := LoadVoiceModelState(path)
	if err != nil {
		t.Fatalf("LoadVoiceModelState: %v", err)
	}

	for key, want := range state.Modules[module] {
		g := got.Modules[module][key]
		if g == nil {
			t.Fatalf("%s missing after the round trip", key)
		}

		wantDType := want.DType
		if wantDType == "" {
			wantDType = "F32"
		}

		if g.DType != wantDType || !slices.Equal(g.Shape, want.Shape) || !slices.Equal(g.Data, want.Data) {
			t.Errorf("%s = %s %v %v, want %s %v %v", key, g.DType, g.Shape, g.Data, wantDType, want.Shape, want.Data)
		}
	}
}

func TestEncodeVoiceModelState_Errors(t *testing.T) {
	one := func() *Tensor { return &Tensor{Shape: []int64{1}, Data: []float32{1}} }

	for _, tc := range []struct {
		name  string
		state *VoiceModelState
	}{
		{name: "nil", state: nil},
		{name: "empty", state: &VoiceModelState{}},
		{name: "empty module name", state: &VoiceModelState{Modules: map[string]map[string]*Tensor{"": {"offset": one()}}}},
		{name: "slash in module name", state: &VoiceModelState{Modules: map[string]map[string]*Tensor{"a/b": {"offset": one()}}}},
		{name: "slash in key", state: &VoiceModelState{Modules: map[string]map[string]*Tensor{"a": {"x/offset": one()}}}},
		{name: "nil tensor", state: &VoiceModelState{Modules: map[string]map[string]*Tensor{"a": {"offset": nil}}}},
	} {
		_, err := EncodeVoiceModelState(tc.state)
		if err == nil {
			t.Errorf("%s: EncodeVoiceModelState succeeded, want an error", tc.name)
		}
	}
}
