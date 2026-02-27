package onnx

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestRunnerRoundTrip(t *testing.T) {
	libPath := os.Getenv("POCKETTTS_ORT_LIB")
	if libPath == "" {
		libPath = os.Getenv("ORT_LIBRARY_PATH")
	}

	if libPath == "" {
		t.Skip("no ORT library available; set POCKETTTS_ORT_LIB")
	}

	identityModel := filepath.Join("..", "model", "testdata", "identity_float32.onnx")
	if _, err := os.Stat(identityModel); err != nil {
		t.Skipf("identity model not found: %v", err)
	}

	session := Session{
		Name: "identity",
		Path: identityModel,
		Inputs: []NodeInfo{
			{Name: "input", DType: "float", Shape: []any{float64(1), float64(3)}},
		},
		Outputs: []NodeInfo{
			{Name: "output", DType: "float", Shape: []any{float64(1), float64(3)}},
		},
	}

	runner, err := NewRunner(session, RunnerConfig{
		LibraryPath: libPath,
		APIVersion:  23,
	})
	if err != nil {
		t.Fatalf("NewRunner: %v", err)
	}
	defer runner.Close()

	input, err := NewTensor([]float32{1.0, 2.0, 3.0}, []int64{1, 3})
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}

	outputs, err := runner.Run(context.Background(), map[string]*Tensor{"input": input})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	out, ok := outputs["output"]
	if !ok {
		t.Fatal("missing 'output' key in results")
	}

	data, err := ExtractFloat32(out)
	if err != nil {
		t.Fatalf("ExtractFloat32: %v", err)
	}

	if len(data) != 3 {
		t.Fatalf("expected 3 elements, got %d", len(data))
	}

	for i, want := range []float32{1.0, 2.0, 3.0} {
		if data[i] != want {
			t.Errorf("data[%d] = %f, want %f", i, data[i], want)
		}
	}
}

func TestRunnerCloseIsIdempotent(t *testing.T) {
	libPath := os.Getenv("POCKETTTS_ORT_LIB")
	if libPath == "" {
		t.Skip("no ORT library available")
	}

	identityModel := filepath.Join("..", "model", "testdata", "identity_float32.onnx")
	if _, err := os.Stat(identityModel); err != nil {
		t.Skipf("identity model not found: %v", err)
	}

	session := Session{
		Name: "identity",
		Path: identityModel,
		Inputs: []NodeInfo{
			{Name: "input", DType: "float", Shape: []any{float64(1), float64(3)}},
		},
		Outputs: []NodeInfo{
			{Name: "output", DType: "float", Shape: []any{float64(1), float64(3)}},
		},
	}

	runner, err := NewRunner(session, RunnerConfig{LibraryPath: libPath, APIVersion: 23})
	if err != nil {
		t.Fatalf("NewRunner: %v", err)
	}

	runner.Close()
	runner.Close() // second close should not panic
}
