package tts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/onnx"
	"github.com/example/go-pocket-tts/internal/text"
)

// ---------------------------------------------------------------------------
// NewService — error paths
// ---------------------------------------------------------------------------

func TestNewService_MissingORTLibrary(t *testing.T) {
	cfg := config.DefaultConfig()
	// Point at a nonexistent ORT library and manifest to trigger error.
	cfg.Runtime.ORTLibraryPath = "/nonexistent/libonnxruntime.so"
	cfg.Paths.ONNXManifest = "/nonexistent/manifest.json"

	_, err := NewService(cfg)
	if err == nil {
		t.Error("NewService with missing ORT library should return error")
	}
}

// ---------------------------------------------------------------------------
// Service.Synthesize — tested via a directly-constructed Service
// (bypassing NewService to avoid ORT/manifest dependency)
// ---------------------------------------------------------------------------

func newTestService(t *testing.T) *Service {
	t.Helper()

	// Try real construction first.
	libPath := os.Getenv("POCKETTTS_ORT_LIB")
	if libPath == "" {
		libPath = os.Getenv("ORT_LIBRARY_PATH")
	}
	if libPath != "" {
		cfg := config.DefaultConfig()
		cfg.Runtime.ORTLibraryPath = libPath
		svc, err := NewService(cfg)
		if err == nil {
			return svc
		}
	}

	// No ORT available — build a stub Service with a manifest-based engine.
	// Create a temp dir with identity model + manifest.
	tmp := t.TempDir()
	src := filepath.Join("..", "onnx", "..", "model", "testdata", "identity_float32.onnx")
	data, err := os.ReadFile(src)
	if err != nil {
		t.Skipf("identity model unavailable: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "identity.onnx"), data, 0o644); err != nil {
		t.Fatalf("write identity model: %v", err)
	}
	manifest := `{"graphs":[{"name":"identity","filename":"identity.onnx","inputs":[{"name":"input","dtype":"float","shape":[1,3]}],"outputs":[{"name":"output","dtype":"float","shape":[1,3]}]}]}`
	if err := os.WriteFile(filepath.Join(tmp, "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	if libPath == "" {
		t.Skipf("no ORT library available; set POCKETTTS_ORT_LIB")
	}

	engine, err := onnx.NewEngine(filepath.Join(tmp, "manifest.json"), onnx.RunnerConfig{
		LibraryPath: libPath,
		APIVersion:  23,
	})
	if err != nil {
		t.Skipf("cannot create engine: %v", err)
	}
	return &Service{
		engine:       engine,
		preprocessor: text.NewPreprocessor(),
	}
}

func TestSynthesize_EmptyInput(t *testing.T) {
	svc := newTestService(t)

	_, err := svc.Synthesize("")
	if err == nil {
		t.Error("Synthesize(\"\") = nil; want error for empty input")
	}
	if !strings.Contains(err.Error(), "tokens") {
		t.Errorf("error %q should mention tokens", err.Error())
	}
}

func TestSynthesize_WhitespaceOnly(t *testing.T) {
	svc := newTestService(t)

	_, err := svc.Synthesize("   \t\n  ")
	if err == nil {
		t.Error("Synthesize(whitespace) = nil; want error (whitespace produces no tokens)")
	}
}

func TestSynthesize_ValidInput_ReturnsErrorUntilPhase18(t *testing.T) {
	svc := newTestService(t)

	// Engine.Infer is a temporary shim that returns an error.
	// This test documents that behavior — it will be updated in Phase 18
	// when the real generation pipeline is implemented.
	_, err := svc.Synthesize("hello world")
	if err == nil {
		t.Fatal("Synthesize should return error (Infer not yet implemented)")
	}
	if !strings.Contains(err.Error(), "not yet implemented") {
		t.Errorf("unexpected error: %v", err)
	}
}
