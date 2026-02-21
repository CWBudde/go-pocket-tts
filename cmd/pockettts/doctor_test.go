package main

import (
	"os"
	"path/filepath"
	"testing"
)


func TestProbePocketTTSVersion_MissingExecutable(t *testing.T) {
	_, err := probePocketTTSVersion("/nonexistent/pocket-tts-binary")
	if err == nil {
		t.Fatal("expected error for missing executable")
	}
}

func TestProbePocketTTSVersion_RealExecutable(t *testing.T) {
	// Create a tiny script that exits 0 and prints a fixed string, simulating
	// a pocket-tts binary that honours --version.
	tmp := t.TempDir()
	script := filepath.Join(tmp, "fake-tts")
	if err := os.WriteFile(script, []byte("#!/bin/sh\necho 'fake-tts 1.2.3'\n"), 0o755); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	got, err := probePocketTTSVersion(script)
	if err != nil {
		t.Fatalf("probePocketTTSVersion: %v", err)
	}
	if got != "fake-tts 1.2.3" {
		t.Errorf("unexpected version output: %q", got)
	}
}

func TestProbePythonVersion_ReturnsVersion(t *testing.T) {
	// python3 or python must be present in CI / dev environments.
	ver, err := probePythonVersion()
	if err != nil {
		t.Skipf("python not available: %v", err)
	}
	if ver == "" {
		t.Error("expected non-empty version string")
	}
}

func TestCollectVoiceFiles_NoManifest(t *testing.T) {
	// Change to a temp dir that has no voices/manifest.json.
	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(orig); err != nil {
			t.Logf("chdir restore failed: %v", err)
		}
	})
	if err := os.Chdir(t.TempDir()); err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	files := collectVoiceFiles()
	// With no manifest, should return nil/empty (not panic).
	if files != nil && len(files) != 0 {
		t.Errorf("expected nil/empty slice without manifest, got %v", files)
	}
}

func TestCollectVoiceFiles_WithManifest(t *testing.T) {
	tmp := t.TempDir()
	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(orig); err != nil {
			t.Logf("chdir restore failed: %v", err)
		}
	})
	if err := os.Chdir(tmp); err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	voiceDir := filepath.Join(tmp, "voices")
	if err := os.MkdirAll(voiceDir, 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	manifest := `{"voices":[{"id":"test","path":"voices/test.bin","license":"MIT"}]}`
	if err := os.WriteFile(filepath.Join(voiceDir, "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	files := collectVoiceFiles()
	if len(files) != 1 {
		t.Errorf("expected 1 voice file, got %d: %v", len(files), files)
	}
	if files[0] != "voices/test.bin" {
		t.Errorf("unexpected path: %q", files[0])
	}
}
