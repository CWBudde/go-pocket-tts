//go:build integration

package model

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/config"
	internalonnx "github.com/example/go-pocket-tts/internal/onnx"
)

func TestVerifyONNXIntegration(t *testing.T) {
	if _, err := internalonnx.DetectRuntime(config.RuntimeConfig{}); err != nil {
		t.Skipf("ONNX Runtime library not detected: %v", err)
	}

	manifestPath := filepath.Join("testdata", "identity_manifest.json")
	err := VerifyONNX(VerifyOptions{
		ManifestPath: manifestPath,
	})
	if err != nil {
		if strings.Contains(err.Error(), "Unsupported model IR version") {
			t.Skipf("skipping due to ORT/IR compatibility: %v", err)
		}
		t.Fatalf("VerifyONNX integration failed: %v", err)
	}
}
