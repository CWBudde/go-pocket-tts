//go:build integration

package onnx

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/example/go-pocket-tts/internal/config"
)

// ortLibPath returns the ORT library path, skipping if unavailable.
func ortLibPath(t *testing.T) string {
	t.Helper()
	info, err := DetectRuntime(config.RuntimeConfig{})
	if err != nil {
		t.Skipf("ONNX Runtime library not detected: %v", err)
	}
	return info.LibraryPath
}

// identitySession returns a Session pointing at the testdata identity model.
// The identity_float32.onnx uses input name "x" and output name "y" with
// shape [1, 4] float32, as defined in internal/model/testdata/identity_manifest.json.
func identitySession(t *testing.T) Session {
	t.Helper()
	return Session{
		Name: "identity",
		Path: filepath.Join("..", "model", "testdata", "identity_float32.onnx"),
		Inputs: []NodeInfo{
			{Name: "x", DType: "float32", Shape: []any{float64(1), float64(4)}},
		},
		Outputs: []NodeInfo{
			{Name: "y", DType: "float32", Shape: []any{float64(1), float64(4)}},
		},
	}
}

// TestRunnerIntegration_RoundTrip verifies that Runner can load the identity
// ONNX model and execute an inference pass with float32 tensors.
func TestRunnerIntegration_RoundTrip(t *testing.T) {
	libPath := ortLibPath(t)

	runner, err := NewRunner(identitySession(t), RunnerConfig{
		LibraryPath: libPath,
		APIVersion:  23,
	})
	if err != nil {
		t.Fatalf("NewRunner: %v", err)
	}
	defer runner.Close()

	want := []float32{1.5, 2.5, 3.5, 4.5}
	input, err := NewTensor(want, []int64{1, 4})
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}

	outputs, err := runner.Run(context.Background(), map[string]*Tensor{"x": input})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	out, ok := outputs["y"]
	if !ok {
		t.Fatalf("missing 'y' key in results; got keys: %v", mapKeys(outputs))
	}

	got, err := ExtractFloat32(out)
	if err != nil {
		t.Fatalf("ExtractFloat32: %v", err)
	}

	if len(got) != len(want) {
		t.Fatalf("output length %d, want %d", len(got), len(want))
	}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("output[%d] = %f, want %f", i, got[i], w)
		}
	}
}

// TestRunnerIntegration_Int64 verifies that Runner handles int64 tensors correctly.
func TestRunnerIntegration_Int64(t *testing.T) {
	// The identity_float32.onnx only supports float32; this test verifies that
	// passing int64 is properly rejected with a clear error (not a panic).
	libPath := ortLibPath(t)

	runner, err := NewRunner(identitySession(t), RunnerConfig{
		LibraryPath: libPath,
		APIVersion:  23,
	})
	if err != nil {
		t.Fatalf("NewRunner: %v", err)
	}
	defer runner.Close()

	intInput, err := NewTensor([]int64{1, 2, 3, 4}, []int64{1, 4})
	if err != nil {
		t.Fatalf("NewTensor int64: %v", err)
	}

	// Expect an error since the model expects float32 input.
	_, err = runner.Run(context.Background(), map[string]*Tensor{"x": intInput})
	if err == nil {
		t.Fatal("expected error running int64 tensor against float32-only model; got nil")
	}
	t.Logf("correctly rejected int64 input: %v", err)
}

// TestRunnerIntegration_Close is idempotent.
func TestRunnerIntegration_Close(t *testing.T) {
	libPath := ortLibPath(t)

	runner, err := NewRunner(identitySession(t), RunnerConfig{
		LibraryPath: libPath,
		APIVersion:  23,
	})
	if err != nil {
		t.Fatalf("NewRunner: %v", err)
	}

	runner.Close()
	runner.Close() // must not panic
}

// TestEngineIntegration_LoadAndRun verifies that Engine loads a manifest,
// creates runners, and can execute a named graph.
func TestEngineIntegration_LoadAndRun(t *testing.T) {
	libPath := ortLibPath(t)

	// Write a temporary manifest pointing at the identity model.
	tmp := t.TempDir()
	manifestPath := writeTestManifest(t, tmp)

	engine, err := NewEngine(manifestPath, RunnerConfig{
		LibraryPath: libPath,
		APIVersion:  23,
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer engine.Close()

	runner, ok := engine.Runner("identity")
	if !ok {
		t.Fatal("expected 'identity' runner in engine")
	}
	if runner.Name() != "identity" {
		t.Errorf("runner name = %q, want %q", runner.Name(), "identity")
	}

	// Execute a round-trip through the engine's runner.
	// The underlying identity_float32.onnx uses shape [1, 4] regardless of
	// what writeTestManifest declares in the manifest metadata.
	want := []float32{7.0, 8.0, 9.0, 10.0}
	input, err := NewTensor(want, []int64{1, 4})
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}

	// The underlying identity_float32.onnx uses "x"/"y" regardless of
	// what the manifest metadata says.
	outputs, err := runner.Run(context.Background(), map[string]*Tensor{"x": input})
	if err != nil {
		t.Fatalf("runner.Run via Engine: %v", err)
	}

	out, ok := outputs["y"]
	if !ok {
		t.Fatalf("missing 'y' in results; keys: %v", mapKeys(outputs))
	}
	got, err := ExtractFloat32(out)
	if err != nil {
		t.Fatalf("ExtractFloat32: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("output length %d, want %d", len(got), len(want))
	}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("output[%d] = %f, want %f", i, got[i], w)
		}
	}
}

func mapKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
