package tts

import (
	"math"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/onnx"
	"github.com/example/go-pocket-tts/internal/text"
)

// ---------------------------------------------------------------------------
// NewService — error paths that don't require a real ORT library
// ---------------------------------------------------------------------------

func TestNewService_InvalidThreadCount(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Runtime.Threads = 0 // NewEngine rejects this before Bootstrap is called

	_, err := NewService(cfg)
	if err == nil {
		t.Error("NewService(threads=0) = nil; want error")
	}
}

func TestNewService_NegativeThreadCount(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Runtime.Threads = -1

	_, err := NewService(cfg)
	if err == nil {
		t.Error("NewService(threads=-1) = nil; want error")
	}
}

// ---------------------------------------------------------------------------
// Service.Synthesize — tested via a directly-constructed Service
// (bypassing NewService to avoid the ORT Bootstrap dependency)
// ---------------------------------------------------------------------------

// newTestService constructs a Service with a real Engine (using a stub
// config) by exploiting that onnx.NewEngine only requires Threads >= 1
// and a Bootstrap call that may fail gracefully.
//
// If Bootstrap fails (no ORT library present), we fall back to building
// a minimal Service by hand so Synthesize can still be tested.
func newTestService(t *testing.T) *Service {
	t.Helper()

	cfg := config.DefaultConfig()
	cfg.Runtime.Threads = 1

	svc, err := NewService(cfg)
	if err == nil {
		return svc
	}

	// ORT runtime not available — build a Service directly with a stub engine.
	// onnx.Engine is an exported type with an exported Infer method, so we
	// can construct one via its package constructor using a synthetic path
	// that will fail Bootstrap but still let us test the token-level logic.
	// Instead: build Service directly using struct literal (white-box, same package).
	stubEngine := buildStubEngine(t)
	return &Service{
		engine:       stubEngine,
		preprocessor: text.NewPreprocessor(),
	}
}

// buildStubEngine creates a minimal *onnx.Engine for tests.
// We use the exported NewEngine path with a fake-but-syntactically-valid
// library path so Bootstrap returns an error, then fall back to a direct
// struct construction via the onnx package's exported surface.
//
// Since onnx.Engine is unexported beyond its package boundary (it's an
// exported type but its fields are unexported), the only way to get a
// valid *onnx.Engine is through onnx.NewEngine. We accept that if ORT
// is absent, Engine-based tests will be skipped.
func buildStubEngine(t *testing.T) *onnx.Engine {
	t.Helper()
	cfg := config.DefaultConfig()
	cfg.Runtime.Threads = 1
	engine, err := onnx.NewEngine(cfg.Runtime)
	if err != nil {
		t.Skipf("onnx.NewEngine unavailable (no ORT library): %v", err)
	}
	return engine
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

func TestSynthesize_ValidInput(t *testing.T) {
	svc := newTestService(t)

	samples, err := svc.Synthesize("hello world")
	if err != nil {
		t.Fatalf("Synthesize(\"hello world\") error = %v", err)
	}
	if len(samples) == 0 {
		t.Error("Synthesize returned empty samples")
	}
	// Samples should be finite float32 values.
	for i, s := range samples {
		if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
			t.Errorf("sample[%d] = %v; want finite value", i, s)
			break
		}
	}
}

func TestSynthesize_OutputLength(t *testing.T) {
	svc := newTestService(t)

	// The Engine stub generates 512 samples per token.
	// "hi" has 2 non-space characters → 2 tokens → 1024 samples.
	samples, err := svc.Synthesize("hi")
	if err != nil {
		t.Fatalf("Synthesize(\"hi\") error = %v", err)
	}
	// Token count = len("hi") = 2; expected samples = 2 * 512 = 1024.
	wantSamples := 2 * 512
	if len(samples) != wantSamples {
		t.Errorf("sample count = %d; want %d", len(samples), wantSamples)
	}
}

func TestSynthesize_PunctuationTokenized(t *testing.T) {
	svc := newTestService(t)

	// Punctuation is not whitespace, so "a!" should produce 2 tokens.
	samples, err := svc.Synthesize("a!")
	if err != nil {
		t.Fatalf("Synthesize(\"a!\") error = %v", err)
	}
	wantSamples := 2 * 512
	if len(samples) != wantSamples {
		t.Errorf("sample count = %d; want %d", len(samples), wantSamples)
	}
}
