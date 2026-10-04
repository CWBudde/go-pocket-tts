package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
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
	vl := loadVoiceLister(filepath.Join(t.TempDir(), "manifest.json"))
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

func TestRuntimeDeps_VoiceManifestFromConfig(t *testing.T) {
	dir := t.TempDir()

	err := os.WriteFile(filepath.Join(dir, "alice.safetensors"), []byte("voice-data"), 0o644)
	if err != nil {
		t.Fatal(err)
	}

	manifestPath := filepath.Join(dir, "manifest.json")

	err = os.WriteFile(manifestPath,
		[]byte(`{"voices":[{"id":"alice","path":"alice.safetensors","license":"MIT"}]}`), 0o644)
	if err != nil {
		t.Fatal(err)
	}

	cfg := config.DefaultConfig()
	cfg.Paths.VoiceManifest = manifestPath
	s := New(cfg, nil)

	// Only the voice lister matters here; the other deps have their own tests.
	_, voices, _, _, err := s.runtimeDeps("cli") //nolint:dogsled // see above
	if err != nil {
		t.Fatalf("runtimeDeps(cli) error = %v", err)
	}

	got := voices.ListVoices()
	if len(got) != 1 || got[0].ID != "alice" {
		t.Errorf("ListVoices() = %+v; want the voice from cfg.Paths.VoiceManifest", got)
	}
}

// /voices lists manifest IDs, so the native synthesizer must accept them:
// tts.Service only understands file paths.
func TestNativeSynthesizer_ResolvesManifestVoiceIDs(t *testing.T) {
	dir := t.TempDir()
	voiceFile := filepath.Join(dir, "alice.safetensors")

	err := os.WriteFile(voiceFile, []byte("voice-data"), 0o644)
	if err != nil {
		t.Fatal(err)
	}

	manifestPath := filepath.Join(dir, "manifest.json")

	err = os.WriteFile(manifestPath,
		[]byte(`{"voices":[{"id":"alice","path":"alice.safetensors","license":"MIT"}]}`), 0o644)
	if err != nil {
		t.Fatal(err)
	}

	cfg := config.DefaultConfig()
	cfg.Paths.VoiceManifest = manifestPath
	// A non-nil service skips model loading; voicePath never calls it.
	s := New(cfg, &tts.Service{})

	synth, _, _, _, err := s.runtimeDeps(config.BackendNative) //nolint:dogsled // only the synthesizer matters
	if err != nil {
		t.Fatalf("runtimeDeps(native) error = %v", err)
	}

	ns, ok := synth.(*nativeSynthesizer)
	if !ok {
		t.Fatalf("synth = %T; want *nativeSynthesizer", synth)
	}

	for voice, want := range map[string]string{
		"alice":                voiceFile,
		"":                     "", // no model config, so no default voice
		"/abs/bob.safetensors": "/abs/bob.safetensors",
		"not-in-manifest":      "not-in-manifest",
	} {
		got, err := ns.voicePath(voice)
		if err != nil || got != want {
			t.Errorf("voicePath(%q) = %q, %v; want %q", voice, got, err, want)
		}
	}

	// Without a manifest every value passes through.
	none := &nativeSynthesizer{}

	got, err := none.voicePath("alice")
	if err != nil || got != "alice" {
		t.Errorf("voicePath without manifest = %q, %v; want alice", got, err)
	}
}

// writeTestVoice writes a minimal voice embedding that tts.CheckVoiceFile
// accepts.
func writeTestVoice(t *testing.T, path string) {
	t.Helper()

	err := safetensors.WriteFile(path, []safetensors.Tensor{
		{Name: "audio_prompt", Shape: []int64{1, 1, 2}, Data: []float32{1, 2}},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// A request without a voice uses the default voice, like upstream serve
// (get_default_voice_for_language); without one the native model stops
// almost at once. It is resolved and loaded once at startup, so a broken
// default voice keeps serve from starting.
func TestRuntimeDeps_DefaultVoice(t *testing.T) {
	dir := t.TempDir()
	juergen := filepath.Join(dir, "juergen.safetensors")
	anna := filepath.Join(dir, "anna.safetensors")

	writeTestVoice(t, juergen)
	writeTestVoice(t, anna)

	corrupt := filepath.Join(dir, "corrupt.safetensors")

	err := os.WriteFile(corrupt, []byte("voice-data"), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	manifestPath := filepath.Join(dir, "manifest.json")

	err = os.WriteFile(manifestPath, []byte(`{"voices":[`+
		`{"id":"juergen","path":"juergen.safetensors","license":"CC-BY-4.0"},`+
		`{"id":"anna","path":"anna.safetensors","license":"CC-BY-4.0"},`+
		`{"id":"broken","path":"corrupt.safetensors","license":"CC-BY-4.0"},`+
		`{"id":"gone","path":"gone.safetensors","license":"CC-BY-4.0"}]}`), 0o644)
	if err != nil {
		t.Fatal(err)
	}

	type setup struct {
		backend, manifest, modelDefault, flag string
	}

	runtimeDeps := func(t *testing.T, s setup) (*nativeSynthesizer, error) {
		t.Helper()

		cfg := config.DefaultConfig()
		cfg.Paths.VoiceManifest = s.manifest
		cfg.Model = &modelcfg.ModelConfig{DefaultVoice: s.modelDefault}
		cfg.Server.DefaultVoice = s.flag

		synth, _, _, _, err := New(cfg, &tts.Service{}).runtimeDeps(s.backend)
		if err != nil {
			return nil, err
		}

		ns, ok := synth.(*nativeSynthesizer)
		if !ok {
			t.Fatalf("synth = %T; want *nativeSynthesizer", synth)
		}

		return ns, nil
	}

	native := config.BackendNative

	for name, tc := range map[string]struct {
		setup

		want string
	}{
		"model default":             {setup{native, manifestPath, "juergen", ""}, juergen},
		"--default-voice ID":        {setup{native, manifestPath, "juergen", "anna"}, anna},
		"--default-voice path":      {setup{native, filepath.Join(dir, "absent.json"), "juergen", anna}, anna},
		"no model default":          {setup{native, manifestPath, "", ""}, ""},
		"native-onnx gets no voice": {setup{config.BackendNativeONNX, manifestPath, "juergen", ""}, ""},
	} {
		ns, err := runtimeDeps(t, tc.setup)
		if err != nil {
			t.Errorf("%s: runtimeDeps error = %v", name, err)
			continue
		}

		got, err := ns.voicePath("")
		if err != nil || got != tc.want {
			t.Errorf("%s: voicePath(no voice) = %q, %v; want %q", name, got, err, tc.want)
		}
	}

	for name, tc := range map[string]struct {
		setup

		errSub string
	}{
		"model default not in manifest":  {setup{native, manifestPath, "alba", ""}, "model download"},
		"missing manifest":               {setup{native, filepath.Join(dir, "absent.json"), "juergen", ""}, "model download"},
		"voice file missing":             {setup{native, manifestPath, "gone", ""}, "model download"},
		"corrupt voice file":             {setup{native, manifestPath, "broken", ""}, "default voice"},
		"--default-voice unknown ID":     {setup{native, manifestPath, "juergen", "nobody"}, "--default-voice"},
		"--default-voice missing path":   {setup{native, manifestPath, "juergen", filepath.Join(dir, "x.safetensors")}, "--default-voice"},
		"--default-voice on native-onnx": {setup{config.BackendNativeONNX, manifestPath, "juergen", "anna"}, "native-safetensors"},
		"--default-voice on cli":         {setup{config.BackendCLI, manifestPath, "juergen", "anna"}, "native-safetensors"},
	} {
		_, err := runtimeDeps(t, tc.setup)
		if err == nil || !strings.Contains(err.Error(), tc.errSub) {
			t.Errorf("%s: runtimeDeps error = %v; want one containing %q", name, err, tc.errSub)
		}
	}
}

// Start must fail before listening when the default voice cannot be loaded.
func TestStart_BrokenDefaultVoiceFailsFast(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Server.ListenAddr = "127.0.0.1:0"
	cfg.Paths.VoiceManifest = filepath.Join(t.TempDir(), "absent.json")
	cfg.Model = &modelcfg.ModelConfig{DefaultVoice: "juergen"}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := New(cfg, &tts.Service{}).Start(ctx)
	if err == nil || !strings.Contains(err.Error(), "juergen") {
		t.Fatalf("Start error = %v; want a default voice error", err)
	}

	if ctx.Err() != nil {
		t.Fatal("Start blocked until the context ended; want an immediate error")
	}
}
