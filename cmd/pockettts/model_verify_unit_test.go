package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/config"
)

func TestVerifyNativeSafetensors_MissingModel(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Paths.ModelPath = filepath.Join(t.TempDir(), "missing.safetensors")

	err := verifyNativeSafetensors(cfg)
	if err == nil || !strings.Contains(err.Error(), "model file not found") {
		t.Fatalf("expected missing model error, got: %v", err)
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
