// Package testutil provides shared skip helpers for integration tests.
//
// Each helper calls t.Skip with a clear human-readable reason when the named
// prerequisite is absent, so integration tests remain runnable in partial
// environments without failing noisily.
//
// Typical usage:
//
//	func TestMyIntegration(t *testing.T) {
//	    testutil.RequirePocketTTS(t)
//	    testutil.RequireVoiceFile(t, "en-default")
//	    ...
//	}
package testutil

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/example/go-pocket-tts/internal/tts"
)

// RequirePocketTTS skips the test if the pocket-tts binary is not found in
// PATH or the path given by the POCKETTTS_TTS_CLI_PATH environment variable.
func RequirePocketTTS(t *testing.T) {
	t.Helper()
	exe := os.Getenv("POCKETTTS_TTS_CLI_PATH")
	if exe == "" {
		exe = "pocket-tts"
	}
	if _, err := exec.LookPath(exe); err != nil {
		t.Skipf("pocket-tts binary not available (%q not in PATH); set POCKETTTS_TTS_CLI_PATH to override", exe)
	}
}

// RequireONNXRuntime skips the test if no ONNX Runtime shared library can be
// located. It checks (in order): the ORT_LIBRARY_PATH env var, then the
// POCKETTTS_ORT_LIB env var, then common system library paths.
func RequireONNXRuntime(t *testing.T) {
	t.Helper()
	for _, env := range []string{"ORT_LIBRARY_PATH", "POCKETTTS_ORT_LIB"} {
		if p := os.Getenv(env); p != "" {
			if _, err := os.Stat(p); err == nil {
				return // found
			}
			t.Skipf("ONNX Runtime library not found at %s=%q", env, p)
		}
	}
	// Fall back to common system locations.
	candidates := []string{
		"/usr/lib/libonnxruntime.so",
		"/usr/local/lib/libonnxruntime.so",
		"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return // found
		}
	}
	t.Skip("ONNX Runtime shared library not found; set ORT_LIBRARY_PATH or POCKETTTS_ORT_LIB")
}

// RequireVoiceFile skips the test if the voice identified by id cannot be
// resolved from voices/manifest.json relative to the current working directory.
func RequireVoiceFile(t *testing.T, id string) {
	t.Helper()
	manifestPath := filepath.Join("voices", "manifest.json")
	vm, err := tts.NewVoiceManager(manifestPath)
	if err != nil {
		t.Skipf("voice manifest not available at %q: %v", manifestPath, err)
	}
	if _, err := vm.ResolvePath(id); err != nil {
		t.Skipf("voice %q not available: %v", id, err)
	}
}

// SilenceWAVPath returns the path to the committed 100 ms silence fixture WAV
// relative to the repository root. Callers should use this as a stand-in audio
// prompt when pocket-tts export-voice is not available.
func SilenceWAVPath() string {
	return filepath.Join("cmd", "pockettts", "testdata", "silence_100ms.wav")
}
