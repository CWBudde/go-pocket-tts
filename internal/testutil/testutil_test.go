package testutil_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/example/go-pocket-tts/internal/testutil"
)

func TestSilenceWAVPath_FileExists(t *testing.T) {
	// Walk up from internal/testutil to the repo root and check the fixture.
	// When tests run, cwd is the package directory; go up two levels.
	root := filepath.Join("..", "..")
	p := filepath.Join(root, testutil.SilenceWAVPath())
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("silence fixture not found at %q: %v", p, err)
	}
}

func TestRequirePocketTTS_SkipsWhenAbsent(t *testing.T) {
	// Temporarily point the binary lookup at something that cannot exist.
	orig := os.Getenv("POCKETTTS_TTS_CLI_PATH")
	t.Setenv("POCKETTTS_TTS_CLI_PATH", "/nonexistent/pocket-tts-binary")
	defer func() {
		if orig == "" {
			os.Unsetenv("POCKETTTS_TTS_CLI_PATH")
		}
	}()

	skipped := false
	fakeT := &skipTracker{TB: t, onSkip: func() { skipped = true }}
	testutil.RequirePocketTTS(fakeT)
	if !skipped {
		t.Error("expected RequirePocketTTS to skip when binary is absent")
	}
}

func TestRequireONNXRuntime_SkipsWhenAbsent(t *testing.T) {
	// Ensure env vars point nowhere.
	t.Setenv("ORT_LIBRARY_PATH", "/nonexistent/libonnxruntime.so")

	skipped := false
	fakeT := &skipTracker{TB: t, onSkip: func() { skipped = true }}
	testutil.RequireONNXRuntime(fakeT)
	if !skipped {
		t.Error("expected RequireONNXRuntime to skip when library is absent")
	}
}

func TestRequireVoiceFile_SkipsWhenManifestAbsent(t *testing.T) {
	// Run from a temp dir that has no voices/manifest.json.
	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	t.Cleanup(func() { os.Chdir(orig) }) //nolint:errcheck
	if err := os.Chdir(t.TempDir()); err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	skipped := false
	fakeT := &skipTracker{TB: t, onSkip: func() { skipped = true }}
	testutil.RequireVoiceFile(fakeT, "any-voice")
	if !skipped {
		t.Error("expected RequireVoiceFile to skip when manifest is absent")
	}
}

// skipTracker is a minimal testing.TB implementation that intercepts Skip calls.
type skipTracker struct {
	testing.TB
	onSkip func()
}

func (s *skipTracker) Helper() {}

func (s *skipTracker) Skipf(_ string, _ ...any) {
	s.onSkip()
	// Do NOT call s.TB.Skip — that would actually skip the outer test.
}
