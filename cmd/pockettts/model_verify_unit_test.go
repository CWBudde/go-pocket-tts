package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
)

func TestVerifyNativeSafetensors_MissingModel(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Paths.ModelPath = filepath.Join(t.TempDir(), "missing.safetensors")

	err := verifyNativeSafetensors(cfg)
	if err == nil || !strings.Contains(err.Error(), "model file not found") {
		t.Fatalf("expected missing model error, got: %v", err)
	}
}

// The smoke load must use the selected model config: the drifting checkpoint
// has no time embeddings and fails to load as the default lsd head.
func TestVerifyNativeSafetensors_DriftingCheckpoint(t *testing.T) {
	modelPath := filepath.Join("..", "..", "models", "english_drifting_26-09", "model.safetensors")

	_, err := os.Stat(modelPath)
	if err != nil {
		t.Skipf("drifting checkpoint not downloaded: %v", err)
	}

	mc, err := modelcfg.Lookup("english_drifting_26-09")
	if err != nil {
		t.Fatal(err)
	}

	cfg := config.DefaultConfig()
	cfg.Model = mc
	cfg.Paths.ModelPath = modelPath
	cfg.Paths.TokenizerModel = filepath.Join("..", "..", "models", "english_drifting_26-09", "tokenizer.model")

	err = verifyNativeSafetensors(cfg)
	if err != nil {
		t.Fatalf("verifyNativeSafetensors: %v", err)
	}
}

func TestVerifyONNX_MissingManifest(t *testing.T) {
	cfg := config.DefaultConfig()
	missingManifest := filepath.Join(t.TempDir(), "missing", "manifest.json")

	err := verifyONNX(missingManifest, cfg, 23)
	if err == nil || !strings.Contains(err.Error(), "model verify failed") {
		t.Fatalf("expected wrapped verify error, got: %v", err)
	}
}

func TestNewModelVerifyCmd_InvalidBackend(t *testing.T) {
	orig := activeCfg

	t.Cleanup(func() { activeCfg = orig })

	activeCfg = config.DefaultConfig()

	cmd := newModelVerifyCmd()
	cmd.SetArgs([]string{"--backend", "bogus"})

	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "invalid backend") {
		t.Fatalf("expected invalid backend error, got: %v", err)
	}
}

func TestNewModelVerifyCmd_DefaultBackendNative(t *testing.T) {
	orig := activeCfg

	t.Cleanup(func() { activeCfg = orig })

	activeCfg = config.DefaultConfig()
	activeCfg.TTS.Backend = config.BackendNative
	activeCfg.Paths.ModelPath = filepath.Join(t.TempDir(), "missing.safetensors")

	cmd := newModelVerifyCmd()
	cmd.SetArgs(nil)

	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "model file not found") {
		t.Fatalf("expected native verify missing model error, got: %v", err)
	}
}

func TestNewModelVerifyCmd_DefaultBackendONNX(t *testing.T) {
	orig := activeCfg

	t.Cleanup(func() { activeCfg = orig })

	activeCfg = config.DefaultConfig()
	activeCfg.TTS.Backend = config.BackendNativeONNX

	cmd := newModelVerifyCmd()
	cmd.SetArgs([]string{"--manifest", filepath.Join(t.TempDir(), "missing-manifest.json")})

	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "model verify failed") {
		t.Fatalf("expected onnx verify error, got: %v", err)
	}
}
