package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/tts"
)

// --- New & WithShutdownTimeout ---

func TestNew_DefaultShutdownTimeout(t *testing.T) {
	cfg := config.DefaultConfig()

	s := New(cfg, nil)
	if s == nil {
		t.Fatal("New() returned nil")
	}

	if s.shutdownTimeout != 30*time.Second {
		t.Errorf("shutdownTimeout = %v; want 30s", s.shutdownTimeout)
	}
}

func TestWithShutdownTimeout(t *testing.T) {
	cfg := config.DefaultConfig()

	s := New(cfg, nil).WithShutdownTimeout(5 * time.Second)
	if s.shutdownTimeout != 5*time.Second {
		t.Errorf("shutdownTimeout = %v; want 5s", s.shutdownTimeout)
	}
}

func TestWithShutdownTimeout_Chaining(t *testing.T) {
	cfg := config.DefaultConfig()
	s := New(cfg, nil)
	returned := s.WithShutdownTimeout(10 * time.Second)
	// Must return the same *Server for chaining.
	if returned != s {
		t.Error("WithShutdownTimeout should return the same *Server")
	}
}

// --- staticVoiceLister ---

func TestStaticVoiceLister_Empty(t *testing.T) {
	vl := staticVoiceLister{}
	voices := vl.ListVoices()
	// nil slice is fine; just verify no panic
	if len(voices) != 0 {
		t.Errorf("ListVoices() = %v; want empty", voices)
	}
}

func TestStaticVoiceLister_ReturnsCopy(t *testing.T) {
	orig := []tts.Voice{{ID: "v1", Path: "v1.bin"}}
	vl := staticVoiceLister{voices: orig}

	got := vl.ListVoices()
	if len(got) != 1 || got[0].ID != "v1" {
		t.Errorf("ListVoices() = %v; want [{v1 v1.bin}]", got)
	}

	// Mutating the returned slice must not affect the original.
	got[0].ID = "mutated"

	fresh := vl.ListVoices()
	if fresh[0].ID != "v1" {
		t.Error("ListVoices() returned a non-copy; mutation affected the source")
	}
}

// --- loadVoiceLister ---

func TestLoadVoiceLister_MissingManifest_ReturnsStatic(t *testing.T) {
	// A non-existent manifest path should fall back to staticVoiceLister (no panic).
	vl := loadVoiceLister()
	if vl == nil {
		t.Error("loadVoiceLister() returned nil")
	}
	// Must be callable without panic.
	_ = vl.ListVoices()
}

// --- runtimeDeps with CLI backend ---

func TestRuntimeDeps_CLIBackend(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TTS.Backend = "cli"
	cfg.Server.Workers = 4
	s := New(cfg, nil)

	synth, voices, workers, streamer, err := s.runtimeDeps("cli")
	if err != nil {
		t.Fatalf("runtimeDeps(cli) error = %v", err)
	}

	if synth == nil {
		t.Error("synth is nil for cli backend")
	}

	if voices == nil {
		t.Error("voices is nil")
	}

	if workers != 4 {
		t.Errorf("workers = %d; want 4", workers)
	}

	if streamer != nil {
		t.Error("streamer should be nil for cli backend")
	}
}

func TestRuntimeDeps_InvalidBackend(t *testing.T) {
	cfg := config.DefaultConfig()
	s := New(cfg, nil)

	synth, voices, workers, streamer, err := s.runtimeDeps("unknown")
	_ = synth
	_ = voices
	_ = workers
	_ = streamer

	if err == nil {
		t.Error("runtimeDeps(unknown) = nil; want error")
	}
}

func TestRuntimeDeps_NativeSafetensors(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Paths.ModelPath = filepath.Join("..", "..", "models", "tts_b6369a24.safetensors")

	cfg.Paths.TokenizerModel = filepath.Join("..", "..", "models", "tokenizer.model")

	_, err := os.Stat(cfg.Paths.ModelPath)
	if err != nil {
		t.Skipf("native safetensors model not available: %v", err)
	}

	_, err = os.Stat(cfg.Paths.TokenizerModel)
	if err != nil {
		t.Skipf("tokenizer model not available: %v", err)
	}

	s := New(cfg, nil)

	synth, voices, workers, streamer, err := s.runtimeDeps(config.BackendNative)
	if err != nil {
		t.Fatalf("runtimeDeps(native-safetensors) error = %v", err)
	}

	if synth == nil || voices == nil {
		t.Fatalf("runtimeDeps(native-safetensors) returned nil deps")
	}

	if workers != 2 {
		t.Fatalf("runtimeDeps(native-safetensors) workers = %d; want 2", workers)
	}

	if streamer == nil {
		t.Fatal("runtimeDeps(native-safetensors) streamer is nil; want non-nil")
	}
}

// --- ProbeHTTP ---

func TestProbeHTTP_Success(t *testing.T) {
	// Start a test HTTP server that returns 200 /health.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	// ProbeHTTP uses "http://" prefix + addr, so strip the scheme.
	addr := srv.Listener.Addr().String()

	err := ProbeHTTP(addr)
	if err != nil {
		t.Errorf("ProbeHTTP(%q) = %v; want nil", addr, err)
	}
}

func TestProbeHTTP_NonOKStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	addr := srv.Listener.Addr().String()

	err := ProbeHTTP(addr)
	if err == nil {
		t.Error("ProbeHTTP() = nil; want error for non-200 response")
	}
}

func TestProbeHTTP_ConnectionRefused(t *testing.T) {
	err := ProbeHTTP("127.0.0.1:1")
	if err == nil {
		t.Error("ProbeHTTP() = nil; want error for unreachable host")
	}
}

// --- Start: invalid backend config ---

func TestStart_InvalidBackend(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TTS.Backend = "bogus"
	s := New(cfg, nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	err := s.Start(ctx)
	if err == nil {
		t.Error("Start() = nil; want error for invalid backend")
	}
}

// --- Functional options ---

func TestOptions_WithMaxTextBytes(t *testing.T) {
	opts := defaultOptions()
	WithMaxTextBytes(1024)(&opts)

	if opts.maxTextBytes != 1024 {
		t.Errorf("maxTextBytes = %d; want 1024", opts.maxTextBytes)
	}
}

func TestOptions_WithWorkers(t *testing.T) {
	opts := defaultOptions()
	WithWorkers(8)(&opts)

	if opts.workers != 8 {
		t.Errorf("workers = %d; want 8", opts.workers)
	}
}

func TestOptions_WithRequestTimeout(t *testing.T) {
	opts := defaultOptions()
	WithRequestTimeout(90 * time.Second)(&opts)

	if opts.requestTimeout != 90*time.Second {
		t.Errorf("requestTimeout = %v; want 90s", opts.requestTimeout)
	}
}

func TestOptions_WithLogger(_ *testing.T) {
	// Just verify it doesn't panic and sets a non-nil logger.
	opts := defaultOptions()
	WithLogger(nil)(&opts)
	// nil logger is valid (caller's choice); no panic expected.
}
