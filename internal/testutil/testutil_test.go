package testutil_test

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/example/go-pocket-tts/internal/testutil"
)

func TestSilenceWAVPath_FileExists(t *testing.T) {
	// Walk up from internal/testutil to the repo root and check the fixture.
	root := filepath.Join("..", "..")
	p := filepath.Join(root, testutil.SilenceWAVPath())
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("silence fixture not found at %q: %v", p, err)
	}
}

func TestRequirePocketTTS_SkipsWhenAbsent(t *testing.T) {
	t.Setenv("POCKETTTS_TTS_CLI_PATH", "/nonexistent/pocket-tts-binary")

	if !captureSkip(func(tb testing.TB) { testutil.RequirePocketTTS(tb) }) {
		t.Error("expected RequirePocketTTS to skip when binary is absent")
	}
}

func TestRequireONNXRuntime_SkipsWhenAbsent(t *testing.T) {
	t.Setenv("ORT_LIBRARY_PATH", "/nonexistent/libonnxruntime.so")
	t.Setenv("POCKETTTS_ORT_LIB", "")

	if !captureSkip(func(tb testing.TB) { testutil.RequireONNXRuntime(tb) }) {
		t.Error("expected RequireONNXRuntime to skip when library is absent")
	}
}

func TestRequireVoiceFile_SkipsWhenManifestAbsent(t *testing.T) {
	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	t.Cleanup(func() { os.Chdir(orig) }) //nolint:errcheck
	if err := os.Chdir(t.TempDir()); err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	if !captureSkip(func(tb testing.TB) { testutil.RequireVoiceFile(tb, "any-voice") }) {
		t.Error("expected RequireVoiceFile to skip when manifest is absent")
	}
}

// captureSkip runs fn in a fresh goroutine with a stub TB and returns true if
// the function called Skip/Skipf. Because the real testing.T.Skipf calls
// runtime.Goexit(), we run fn in an isolated goroutine so Goexit only
// terminates that goroutine and does not propagate to the parent test.
func captureSkip(fn func(testing.TB)) (skipped bool) {
	stub := &stubTB{}
	done := make(chan struct{})
	go func() {
		defer close(done)
		fn(stub)
	}()
	<-done
	return stub.skipped
}

// stubTB is a minimal testing.TB that records Skip calls and terminates the
// calling goroutine (via runtime.Goexit) exactly as the real testing.T does.
type stubTB struct {
	testing.TB // intentionally nil — only Skip methods are called
	skipped    bool
}

func (s *stubTB) Helper()                 {}
func (s *stubTB) Log(_ ...any)            {}
func (s *stubTB) Logf(_ string, _ ...any) {}

func (s *stubTB) Skip(_ ...any) {
	s.skipped = true
	runtime.Goexit()
}

func (s *stubTB) Skipf(_ string, _ ...any) {
	s.skipped = true
	runtime.Goexit()
}

func (s *stubTB) SkipNow() {
	s.skipped = true
	runtime.Goexit()
}
