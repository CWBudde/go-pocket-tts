//go:build integration

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/testutil"
)

// runDoctorCapture executes the doctor command with the given extra args and
// returns the combined stdout output and the execution error (if any).
// The doctor command writes directly to os.Stdout/os.Stderr, so we redirect
// those descriptors via a pipe for the duration of the call.
func runDoctorCapture(t testing.TB, args ...string) (stdout string, err error) {
	t.Helper()

	pr, pw, pipeErr := os.Pipe()
	if pipeErr != nil {
		t.Fatalf("os.Pipe: %v", pipeErr)
	}
	origStdout := os.Stdout
	origStderr := os.Stderr
	os.Stdout = pw
	os.Stderr = pw // capture stderr into the same buffer for simplicity

	root := NewRootCmd()
	root.SetArgs(append([]string{"doctor"}, args...))
	execErr := root.Execute()

	pw.Close()
	os.Stdout = origStdout
	os.Stderr = origStderr

	var buf bytes.Buffer
	if _, readErr := buf.ReadFrom(pr); readErr != nil {
		t.Fatalf("read pipe: %v", readErr)
	}
	pr.Close()

	return buf.String(), execErr
}

// ---------------------------------------------------------------------------
// TestDoctorPasses_CLI
// ---------------------------------------------------------------------------

// TestDoctorPasses_CLI runs pockettts doctor in CLI mode against a valid
// environment (pocket-tts binary present) and asserts exit 0 with
// "doctor checks passed" in output.
func TestDoctorPasses_CLI(t *testing.T) {
	testutil.RequirePocketTTS(t)

	// Use an existing voice file so the voice-file check passes too.
	// model verify is skipped gracefully when no ONNX manifest is present.
	voiceFile := setupTempVoiceManifest(t, true)
	_ = voiceFile

	out, err := runDoctorCapture(t, "--backend", "cli")
	if err != nil {
		t.Fatalf("doctor failed: %v\noutput:\n%s", err, out)
	}
	if !strings.Contains(out, "doctor checks passed") {
		t.Errorf("expected 'doctor checks passed' in output, got:\n%s", out)
	}
}

// ---------------------------------------------------------------------------
// TestDoctorPasses_Native
// ---------------------------------------------------------------------------

// TestDoctorPasses_Native runs pockettts doctor in native mode, which does not
// require pocket-tts or Python. Asserts exit 0 with "doctor checks passed".
func TestDoctorPasses_Native(t *testing.T) {
	// Native mode skips pocket-tts and Python checks.
	// model verify is skipped gracefully when no ONNX manifest is present.
	// No external dependencies required.
	setupTempVoiceManifest(t, true)

	out, err := runDoctorCapture(t, "--backend", "native")
	if err != nil {
		t.Fatalf("doctor --backend native failed: %v\noutput:\n%s", err, out)
	}
	if !strings.Contains(out, "doctor checks passed") {
		t.Errorf("expected 'doctor checks passed' in output, got:\n%s", out)
	}
	if !strings.Contains(out, "backend: native-safetensors") {
		t.Errorf("expected 'backend: native-safetensors' in output, got:\n%s", out)
	}
}

// ---------------------------------------------------------------------------
// TestDoctorFails_MissingVoiceFile
// ---------------------------------------------------------------------------

// TestDoctorFails_MissingVoiceFile points the manifest at a non-existent voice
// file and asserts exit non-zero with a failure message in output.
func TestDoctorFails_MissingVoiceFile(t *testing.T) {
	// Create a manifest whose voice path does not exist on disk.
	setupTempVoiceManifest(t, false /* file absent */)

	out, err := runDoctorCapture(t, "--backend", "native")
	if err == nil {
		t.Fatalf("expected doctor to fail with missing voice file, but it passed\noutput:\n%s", out)
	}
	lower := strings.ToLower(out)
	if !strings.Contains(lower, "not found") && !strings.Contains(lower, "fail") {
		t.Errorf("expected failure message about missing voice file in output, got:\n%s", out)
	}
}

// ---------------------------------------------------------------------------
// TestDoctorFails_BadPocketTTS
// ---------------------------------------------------------------------------

// TestDoctorFails_BadPocketTTS provides a fake pocket-tts that exits 1 and
// asserts that the failure is surfaced in the doctor output.
func TestDoctorFails_BadPocketTTS(t *testing.T) {
	tmp := t.TempDir()
	fakeTTS := filepath.Join(tmp, "pocket-tts")
	// Script that exits 1 unconditionally, simulating a broken binary.
	script := "#!/bin/sh\nexit 1\n"
	if err := os.WriteFile(fakeTTS, []byte(script), 0o755); err != nil {
		t.Fatalf("WriteFile fake pocket-tts: %v", err)
	}

	// Point PATH to the temp dir so our fake is found first.
	origPath := os.Getenv("PATH")
	t.Setenv("PATH", tmp+string(os.PathListSeparator)+origPath)

	// Set up a valid voice manifest so that check doesn't interfere.
	setupTempVoiceManifest(t, true)

	out, err := runDoctorCapture(t, "--backend", "cli")
	if err == nil {
		t.Fatalf("expected doctor to fail with bad pocket-tts, but it passed\noutput:\n%s", out)
	}
	lower := strings.ToLower(out)
	if !strings.Contains(lower, "fail") && !strings.Contains(lower, "not found") && !strings.Contains(lower, "pocket-tts") {
		t.Errorf("expected pocket-tts failure message in output, got:\n%s", out)
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// setupTempVoiceManifest writes a temporary voices/manifest.json in a fresh
// temp directory, changes the working directory to it, and registers cleanup.
// If voiceFileExists is true, the voice file is created on disk too.
// Returns the path to the voice file.
func setupTempVoiceManifest(t testing.TB, voiceFileExists bool) string {
	t.Helper()

	tmp := t.TempDir()
	voiceDir := filepath.Join(tmp, "voices")
	if err := os.MkdirAll(voiceDir, 0o755); err != nil {
		t.Fatalf("MkdirAll voices/: %v", err)
	}

	voiceRelPath := filepath.Join("voices", "test.safetensors")
	voiceAbsPath := filepath.Join(tmp, voiceRelPath)

	manifest := map[string]any{
		"voices": []map[string]any{
			{"id": "test", "path": voiceRelPath, "license": "MIT"},
		},
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	if err := os.WriteFile(filepath.Join(voiceDir, "manifest.json"), data, 0o644); err != nil {
		t.Fatalf("WriteFile manifest: %v", err)
	}

	if voiceFileExists {
		if err := os.WriteFile(voiceAbsPath, []byte("dummy"), 0o644); err != nil {
			t.Fatalf("WriteFile voice file: %v", err)
		}
	}

	// Also create the ONNX models directory so the model verify step finds no
	// manifest (not a hard failure — it prints an error but we handle it via
	// the native-mode skip).
	_ = os.MkdirAll(filepath.Join(tmp, "models", "onnx"), 0o755)

	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(tmp); err != nil {
		t.Fatalf("Chdir %q: %v", tmp, err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(orig); err != nil {
			t.Logf("Chdir restore failed: %v", err)
		}
	})

	return voiceAbsPath
}
