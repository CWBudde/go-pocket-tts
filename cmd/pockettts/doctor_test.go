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

	writeErr := os.WriteFile(script, []byte("#!/bin/sh\necho 'fake-tts 1.2.3'\n"), 0o755)
	if writeErr != nil {
		t.Fatalf("WriteFile: %v", writeErr)
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
		err := os.Chdir(orig)
		if err != nil {
			t.Logf("chdir restore failed: %v", err)
		}
	})

	err = os.Chdir(t.TempDir())
	if err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	files := collectVoiceFiles()
	// With no manifest, should return nil/empty (not panic).
	if len(files) != 0 {
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
		err := os.Chdir(orig)
		if err != nil {
			t.Logf("chdir restore failed: %v", err)
		}
	})

	err = os.Chdir(tmp)
	if err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	voiceDir := filepath.Join(tmp, "voices")

	err = os.MkdirAll(voiceDir, 0o755)
	if err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	// Voice file is stored as "test.bin" relative to the manifest directory.
	voiceFile := filepath.Join(voiceDir, "test.bin")

	err = os.WriteFile(voiceFile, []byte("dummy"), 0o644)
	if err != nil {
		t.Fatalf("WriteFile voice: %v", err)
	}

	manifest := `{"voices":[{"id":"test","path":"test.bin","license":"MIT"}]}`

	err = os.WriteFile(filepath.Join(voiceDir, "manifest.json"), []byte(manifest), 0o644)
	if err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	files := collectVoiceFiles()
	if len(files) != 1 {
		t.Errorf("expected 1 voice file, got %d: %v", len(files), files)
	}
	// Path must be absolute so the doctor check works regardless of CWD.
	if !filepath.IsAbs(files[0]) {
		t.Errorf("expected absolute path, got %q", files[0])
	}
}

// TestCollectVoiceFiles_PathResolvedRelativeToManifest verifies that
// collectVoiceFiles resolves paths relative to the manifest directory, not to
// the working directory. This is the regression test for the bug where
// voice files like "mimi.safetensors" were checked against CWD instead of
// the "voices/" directory next to the manifest.
func TestCollectVoiceFiles_PathResolvedRelativeToManifest(t *testing.T) {
	// Layout inside the fake project root (<tmp>):
	//   <tmp>/
	//     voices/
	//       manifest.json   (references "mimi.safetensors")
	//       mimi.safetensors   (voice file lives HERE, next to the manifest)
	//
	// CWD is set to <tmp> so "voices/manifest.json" resolves, but
	// "mimi.safetensors" does NOT exist at the CWD level — only inside voices/.
	tmp := t.TempDir()

	voiceDir := filepath.Join(tmp, "voices")

	err := os.MkdirAll(voiceDir, 0o755)
	if err != nil {
		t.Fatalf("MkdirAll voiceDir: %v", err)
	}

	// Voice file lives next to the manifest, not at the project root.
	err = os.WriteFile(filepath.Join(voiceDir, "mimi.safetensors"), []byte("dummy"), 0o644)
	if err != nil {
		t.Fatalf("WriteFile voice: %v", err)
	}

	manifest := `{"voices":[{"id":"mimi","path":"mimi.safetensors","license":"CC-BY-4.0"}]}`

	err = os.WriteFile(filepath.Join(voiceDir, "manifest.json"), []byte(manifest), 0o644)
	if err != nil {
		t.Fatalf("WriteFile manifest: %v", err)
	}

	// CWD = project root (<tmp>). "mimi.safetensors" does NOT exist here,
	// only inside "voices/". The old buggy code returned the raw path
	// "mimi.safetensors" which would be stat'd against CWD and not found.
	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}

	t.Cleanup(func() {
		err := os.Chdir(orig)
		if err != nil {
			t.Logf("chdir restore: %v", err)
		}
	})

	err = os.Chdir(tmp)
	if err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	files := collectVoiceFiles()
	if len(files) != 1 {
		t.Fatalf("expected 1 resolved path, got %d: %v", len(files), files)
	}

	// The returned path must be absolute and point to the actual file.
	if !filepath.IsAbs(files[0]) {
		t.Errorf("path is not absolute: %q", files[0])
	}

	_, err = os.Stat(files[0])
	if err != nil {
		t.Errorf("returned path does not exist: %q (%v)", files[0], err)
	}
}
