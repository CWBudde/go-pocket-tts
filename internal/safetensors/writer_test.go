package safetensors

import (
	"path/filepath"
	"testing"
)

func TestWriteFile_RoundTripSingleTensor(t *testing.T) {
	path := filepath.Join(t.TempDir(), "voice.safetensors")

	want := Tensor{
		Name:  "audio_prompt",
		Shape: []int64{1, 2, 4},
		Data:  []float32{1.5, -0.25, 3.25, 4.0, -1.0, 0.5, 2.5, 9.0},
	}

	if err := WriteFile(path, []Tensor{want}); err != nil {
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
	if _, err := EncodeTensors(nil); err == nil {
		t.Fatal("EncodeTensors(nil) should fail")
	}

	if _, err := EncodeTensors([]Tensor{{Name: "", Shape: []int64{1}, Data: []float32{1}}}); err == nil {
		t.Fatal("empty tensor name should fail")
	}

	if _, err := EncodeTensors([]Tensor{
		{Name: "x", Shape: []int64{1}, Data: []float32{1}},
		{Name: "x", Shape: []int64{1}, Data: []float32{2}},
	}); err == nil {
		t.Fatal("duplicate tensor names should fail")
	}

	if _, err := EncodeTensors([]Tensor{{Name: "x", Shape: []int64{1, 2}, Data: []float32{1}}}); err == nil {
		t.Fatal("shape/data mismatch should fail")
	}
}
