package model

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/onnx"
)

func TestVerifyONNXRunsNativeVerifier(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "tiny.onnx")
	manifestPath := filepath.Join(tmp, "manifest.json")

	if err := os.WriteFile(modelPath, []byte("fake-onnx"), 0o644); err != nil {
		t.Fatalf("write model file: %v", err)
	}

	manifest := `{
  "graphs": [
    {
      "name": "tiny",
      "filename": "tiny.onnx",
      "inputs": [{"name":"x","dtype":"float32","shape":[1,4]}],
      "outputs": [{"name":"y","dtype":"float32","shape":[1,4]}]
    }
  ]
}`
	if err := os.WriteFile(manifestPath, []byte(manifest), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	orig := runNativeVerify
	t.Cleanup(func() { runNativeVerify = orig })

	var called bool
	runNativeVerify = func(sessions []onnx.Session, opts VerifyOptions) error {
		called = true
		if len(sessions) != 1 || sessions[0].Name != "tiny" {
			t.Fatalf("unexpected sessions: %+v", sessions)
		}
		if opts.ManifestPath != manifestPath {
			t.Fatalf("unexpected manifest path: %s", opts.ManifestPath)
		}
		return nil
	}

	var out bytes.Buffer
	err := VerifyONNX(VerifyOptions{
		ManifestPath: manifestPath,
		ORTLibrary:   "/tmp/libonnxruntime.so",
		Stdout:       &out,
		Stderr:       &out,
	})
	if err != nil {
		t.Fatalf("VerifyONNX failed: %v", err)
	}
	if !called {
		t.Fatal("expected native verifier to be called")
	}
}

func TestVerifyONNXRejectsInvalidInputShape(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "tiny.onnx")
	manifestPath := filepath.Join(tmp, "manifest.json")

	if err := os.WriteFile(modelPath, []byte("fake-onnx"), 0o644); err != nil {
		t.Fatalf("write model file: %v", err)
	}

	manifest := `{
  "graphs": [
    {
      "name": "bad",
      "filename": "tiny.onnx",
      "inputs": [{"name":"x","dtype":"float32","shape":[0,4]}],
      "outputs": []
    }
  ]
}`
	if err := os.WriteFile(manifestPath, []byte(manifest), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	orig := runNativeVerify
	t.Cleanup(func() { runNativeVerify = orig })
	runNativeVerify = func(_ []onnx.Session, _ VerifyOptions) error { return nil }

	err := VerifyONNX(VerifyOptions{ManifestPath: manifestPath})
	if err == nil {
		t.Fatal("expected shape validation error")
	}
	if !strings.Contains(err.Error(), "not a positive integer") {
		t.Fatalf("unexpected error: %v", err)
	}
}
