//go:build integration

package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/example/go-pocket-tts/internal/onnx"
	"github.com/example/go-pocket-tts/internal/safetensors"
	"github.com/example/go-pocket-tts/internal/testutil"
)

func findRepoFile(t *testing.T, rel string) string {
	t.Helper()

	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatalf("abs path: %v", err)
	}
	for {
		candidate := filepath.Join(dir, rel)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Skipf("%s not found", rel)
	return ""
}

func findModelWeights(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("POCKETTTS_MODEL_SAFETENSORS"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	candidates := []string{
		"models/tts_b6369a24.safetensors",
		"models/model.safetensors",
	}
	for _, c := range candidates {
		if p := findRepoFileOptional(c); p != "" {
			return p
		}
	}
	t.Skip("model safetensors not found; set POCKETTTS_MODEL_SAFETENSORS or download models")
	return ""
}

func findRepoFileOptional(rel string) string {
	dir, err := filepath.Abs(".")
	if err != nil {
		return ""
	}
	for {
		candidate := filepath.Join(dir, rel)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
}

func TestExportVoiceIntegration_NativeONNXPath(t *testing.T) {
	testutil.RequireONNXRuntime(t)

	manifestPath := findRepoFile(t, filepath.Join("models", "onnx", "manifest.json"))
	weightsPath := findModelWeights(t)
	inputPath := findRepoFile(t, filepath.Join("cmd", "pockettts", "testdata", "silence_100ms.wav"))

	sm, err := onnx.NewSessionManager(manifestPath)
	if err != nil {
		t.Fatalf("NewSessionManager: %v", err)
	}
	if _, ok := sm.Session("mimi_encoder"); !ok {
		t.Skip("mimi_encoder graph missing from manifest")
	}

	out := filepath.Join(t.TempDir(), "voice.safetensors")
	cmd := NewRootCmd()
	cmd.SetArgs([]string{
		"export-voice",
		"--paths-onnx-manifest=" + manifestPath,
		"--model-safetensors=" + weightsPath,
		"--input=" + inputPath,
		"--out=" + out,
		"--id=itest",
		"--license=integration",
	})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("export-voice failed: %v", err)
	}

	data, shape, err := safetensors.LoadVoiceEmbedding(out)
	if err != nil {
		t.Fatalf("LoadVoiceEmbedding: %v", err)
	}
	if len(shape) != 3 || shape[0] != 1 || shape[2] != onnx.VoiceEmbeddingDim {
		t.Fatalf("shape = %v, want [1 T %d]", shape, onnx.VoiceEmbeddingDim)
	}
	if len(data) != int(shape[1]*shape[2]) {
		t.Fatalf("data length = %d, want %d", len(data), shape[1]*shape[2])
	}
	if shape[1] < 1 {
		t.Fatalf("shape[1] = %d; want > 0", shape[1])
	}
}
