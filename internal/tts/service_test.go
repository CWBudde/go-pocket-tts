package tts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/onnx"
	"github.com/example/go-pocket-tts/internal/tokenizer"
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
	// Use the real tokenizer model if available, otherwise a nil tokenizer
	// is acceptable for these tests (they only test Synthesize error paths).
	var tok tokenizer.Tokenizer
	if tokPath := config.DefaultConfig().Paths.TokenizerModel; tokPath != "" {
		if t, err := tokenizer.NewSentencePieceTokenizer(tokPath); err == nil {
			tok = t
		}
	}
	return &Service{
		engine:    engine,
		tokenizer: tok,
	}
}

// TestNewService_MissingTokenizerModel verifies that NewService returns an error
// when the tokenizer model path points to a missing file.
func TestNewService_MissingTokenizerModel(t *testing.T) {
	libPath := os.Getenv("POCKETTTS_ORT_LIB")
	if libPath == "" {
		libPath = os.Getenv("ORT_LIBRARY_PATH")
	}
	if libPath == "" {
		t.Skip("no ORT library available; set POCKETTTS_ORT_LIB")
	}

	cfg := config.DefaultConfig()
	cfg.Runtime.ORTLibraryPath = libPath
	cfg.Paths.TokenizerModel = "/nonexistent/tokenizer.model"
	// ONNXManifest is also missing, but tokenizer error should surface first.

	_, err := NewService(cfg)
	if err == nil {
		t.Error("NewService with missing tokenizer model should return error")
	}
}

func TestSynthesize_EmptyInput(t *testing.T) {
	svc := newTestService(t)

	_, err := svc.Synthesize("", "")
	if err == nil {
		t.Error("Synthesize(\"\") = nil; want error for empty input")
	}
	if !strings.Contains(err.Error(), "tokens") {
		t.Errorf("error %q should mention tokens", err.Error())
	}
}

func TestSynthesize_WhitespaceOnly(t *testing.T) {
	svc := newTestService(t)

	_, err := svc.Synthesize("   \t\n  ", "")
	if err == nil {
		t.Error("Synthesize(whitespace) = nil; want error (whitespace produces no tokens)")
	}
}

func TestSynthesize_ValidInput_ReturnsErrorWithTestEngine(t *testing.T) {
	svc := newTestService(t)

	// The test engine uses an identity model, not the real TTS graphs,
	// so Synthesize returns an error from the generation pipeline
	// (missing text_conditioner or other graph).
	_, err := svc.Synthesize("hello world", "")
	if err == nil {
		t.Fatal("Synthesize with test engine should return error (missing TTS graphs)")
	}
}

// ---------------------------------------------------------------------------
// Task 19.4 tests: voice path loading through Service.Synthesize
// ---------------------------------------------------------------------------

// fakeTokenizer always returns a small fixed token sequence regardless of input.
type fakeTokenizer struct{}

func (f fakeTokenizer) Encode(_ string) ([]int64, error) {
	return []int64{1, 2, 3}, nil
}

// newVoiceTestService builds a Service with a fake tokenizer and no real engine.
// This is sufficient for testing the voice loading error paths in Synthesize,
// which occur before the engine is invoked.
func newVoiceTestService() *Service {
	return &Service{
		engine:    nil,
		tokenizer: fakeTokenizer{},
	}
}

// TestSynthesize_BadSafetensorsPath_ReturnsError verifies that Synthesize
// propagates the safetensors load error when the voice path points to a
// missing file.
func TestSynthesize_BadSafetensorsPath_ReturnsError(t *testing.T) {
	svc := newVoiceTestService()

	// Point at a non-existent .safetensors file.
	_, err := svc.Synthesize("hello world", "/nonexistent/voice.safetensors")
	if err == nil {
		t.Fatal("Synthesize with missing voice file = nil; want error")
	}
	if !strings.Contains(err.Error(), "load voice embedding") {
		t.Errorf("error %q should mention 'load voice embedding'", err.Error())
	}
}

// TestSynthesize_InvalidSafetensorsFile_ReturnsError verifies that Synthesize
// propagates parse errors when the voice path points to a file with invalid
// safetensors content.
func TestSynthesize_InvalidSafetensorsFile_ReturnsError(t *testing.T) {
	svc := newVoiceTestService()

	// Write a file that is definitely not a valid safetensors file.
	tmp := filepath.Join(t.TempDir(), "bad.safetensors")
	if err := os.WriteFile(tmp, []byte("not a safetensors file"), 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}

	_, err := svc.Synthesize("hello world", tmp)
	if err == nil {
		t.Fatal("Synthesize with invalid safetensors = nil; want error")
	}
	if !strings.Contains(err.Error(), "load voice embedding") {
		t.Errorf("error %q should mention 'load voice embedding'", err.Error())
	}
}

// TestSynthesize_EmptyVoicePath_SkipsEmbeddingLoad verifies that a whitespace-
// only voice path is treated as "no voice" and does not trigger a safetensors
// load. We use a non-existent voice path with explicit whitespace as input to
// Synthesize — if the whitespace trimming were broken, it would attempt to open
// the path and return a "load voice embedding" error instead of the expected
// safetensors file-not-found error for a non-empty path.
//
// This test specifically checks the boundary: "   " (whitespace only) must
// behave identically to "" (empty string), both skipping the safetensors load.
// The existing TestSynthesize_BadSafetensorsPath_ReturnsError proves the
// positive case: a non-whitespace path DOES attempt loading.
func TestSynthesize_EmptyVoicePath_SkipsEmbeddingLoad(t *testing.T) {
	svc := newTestService(t)

	_, err := svc.Synthesize("hello world", "   ")
	// With the test engine, we expect a pipeline error (missing TTS graphs),
	// NOT a safetensors error.
	if err != nil && strings.Contains(err.Error(), "load voice embedding") {
		t.Errorf("whitespace voice path should not attempt safetensors load, got: %v", err)
	}
}
