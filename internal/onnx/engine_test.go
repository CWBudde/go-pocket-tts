package onnx

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTestManifest(t *testing.T, dir string) string {
	t.Helper()
	onnxFile := filepath.Join(dir, "identity.onnx")

	src := filepath.Join("..", "model", "testdata", "identity_float32.onnx")
	data, err := os.ReadFile(src)
	if err != nil {
		t.Fatalf("read identity model: %v", err)
	}
	if err := os.WriteFile(onnxFile, data, 0o644); err != nil {
		t.Fatalf("write identity model: %v", err)
	}

	manifest := `{
  "graphs": [
    {
      "name": "identity",
      "filename": "identity.onnx",
      "inputs": [{"name":"input","dtype":"float","shape":[1,3]}],
      "outputs": [{"name":"output","dtype":"float","shape":[1,3]}]
    }
  ]
}`
	mPath := filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(mPath, []byte(manifest), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return mPath
}

func TestNewEngineLoadsRunners(t *testing.T) {
	libPath := os.Getenv("POCKETTTS_ORT_LIB")
	if libPath == "" {
		libPath = os.Getenv("ORT_LIBRARY_PATH")
	}
	if libPath == "" {
		t.Skip("no ORT library available")
	}

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
		t.Fatal("expected 'identity' runner")
	}
	if runner.Name() != "identity" {
		t.Fatalf("expected name 'identity', got %q", runner.Name())
	}
}

func TestNewEngineRejectsMissingManifest(t *testing.T) {
	_, err := NewEngine("/nonexistent/manifest.json", RunnerConfig{
		LibraryPath: "/fake",
		APIVersion:  23,
	})
	if err == nil {
		t.Fatal("expected error for missing manifest")
	}
}
