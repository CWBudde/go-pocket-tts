//go:build integration

package onnx

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/example/go-pocket-tts/internal/tokenizer"
)

// tokenizerModelPath finds models/tokenizer.model by walking up from the
// package directory. Skips the test if the file is not found.
func tokenizerModelPath(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatalf("abs path: %v", err)
	}
	for {
		candidate := filepath.Join(dir, "models", "tokenizer.model")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Skip("models/tokenizer.model not found; skipping text_conditioner integration test")
	return ""
}

// textConditionerManifestPath locates the real ONNX manifest that includes the
// text_conditioner graph. Skips the test if not found.
func textConditionerManifestPath(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatalf("abs path: %v", err)
	}
	for {
		candidate := filepath.Join(dir, "models", "onnx", "manifest.json")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Skip("models/onnx/manifest.json not found; skipping text_conditioner integration test")
	return ""
}

// TestTextConditionerIntegration_OutputShape tokenizes a sentence, runs the
// text_conditioner graph, and verifies the output has shape [1, T, 1024].
func TestTextConditionerIntegration_OutputShape(t *testing.T) {
	libPath := ortLibPath(t)
	tokPath := tokenizerModelPath(t)
	manifestPath := textConditionerManifestPath(t)

	tok, err := tokenizer.NewSentencePieceTokenizer(tokPath)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer: %v", err)
	}

	const inputText = "Hello world."
	tokenIDs, err := tok.Encode(inputText)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(tokenIDs) == 0 {
		t.Fatal("tokenizer returned no token IDs")
	}
	t.Logf("tokenized %q → %d tokens: %v", inputText, len(tokenIDs), tokenIDs)

	engine, err := NewEngine(manifestPath, RunnerConfig{
		LibraryPath: libPath,
		APIVersion:  23,
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer engine.Close()

	if _, ok := engine.Runner("text_conditioner"); !ok {
		t.Skip("text_conditioner graph not present in manifest; skipping")
	}

	emb, err := engine.TextConditioner(context.Background(), tokenIDs)
	if err != nil {
		t.Fatalf("TextConditioner: %v", err)
	}

	shape := emb.Shape()
	T := int64(len(tokenIDs))
	if len(shape) != 3 || shape[0] != 1 || shape[1] != T || shape[2] != 1024 {
		t.Errorf("text_embeddings shape = %v, want [1 %d 1024]", shape, T)
	}
	t.Logf("text_embeddings shape = %v ✓", shape)
}
