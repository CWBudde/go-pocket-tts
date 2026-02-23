//go:build integration

package onnx

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func modelSafetensorsPath(t *testing.T) string {
	t.Helper()

	if p := os.Getenv("POCKETTTS_MODEL_SAFETENSORS"); strings.TrimSpace(p) != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatalf("abs path: %v", err)
	}
	for {
		candidates := []string{
			filepath.Join(dir, "models", "tts_b6369a24.safetensors"),
			filepath.Join(dir, "models", "model.safetensors"),
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				return c
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	t.Skip("model safetensors not found; set POCKETTTS_MODEL_SAFETENSORS or download models")
	return ""
}

func silenceFixturePath(t *testing.T) string {
	t.Helper()

	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatalf("abs path: %v", err)
	}
	for {
		candidate := filepath.Join(dir, "cmd", "pockettts", "testdata", "silence_100ms.wav")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	t.Skip("silence fixture not found")
	return ""
}

func TestEncodeVoiceIntegration_OutputShape(t *testing.T) {
	libPath := ortLibPath(t)
	manifestPath := textConditionerManifestPath(t)
	weightsPath := modelSafetensorsPath(t)
	audioPath := silenceFixturePath(t)

	engine, err := NewEngine(manifestPath, RunnerConfig{
		LibraryPath:      libPath,
		APIVersion:       23,
		ModelWeightsPath: weightsPath,
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer engine.Close()

	if _, ok := engine.Runner("mimi_encoder"); !ok {
		t.Skip("mimi_encoder graph not present in manifest; skipping")
	}

	samples, err := loadVoiceAudioSamples(audioPath)
	if err != nil {
		t.Fatalf("loadVoiceAudioSamples: %v", err)
	}

	embedding, err := engine.encodeVoiceSamples(context.Background(), samples)
	if err != nil {
		t.Fatalf("encodeVoiceSamples: %v", err)
	}

	shape := embedding.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[2] != VoiceEmbeddingDim {
		t.Fatalf("embedding shape = %v, want [1 T %d]", shape, VoiceEmbeddingDim)
	}
	if shape[1] < 1 {
		t.Fatalf("embedding shape[1] = %d; want > 0", shape[1])
	}

	data, err := ExtractFloat32(embedding)
	if err != nil {
		t.Fatalf("ExtractFloat32: %v", err)
	}
	if len(data) != int(shape[1]*VoiceEmbeddingDim) {
		t.Fatalf("embedding len = %d, want %d", len(data), shape[1]*VoiceEmbeddingDim)
	}
	for i, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("embedding[%d] is not finite (%v)", i, v)
		}
	}
}
