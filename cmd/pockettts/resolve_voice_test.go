package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveVoiceOrPath_Empty(t *testing.T) {
	got, err := resolveVoiceOrPath("")
	if err != nil {
		t.Fatalf("resolveVoiceOrPath(\"\") returned error: %v", err)
	}
	if got != "" {
		t.Errorf("expected empty string, got %q", got)
	}
}

func TestResolveVoiceOrPath_NoManifest(t *testing.T) {
	// When manifest is missing, the raw voice value is returned as-is.
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

	got, err := resolveVoiceOrPath("my-voice")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "my-voice" {
		t.Errorf("expected passthrough %q, got %q", "my-voice", got)
	}
}

func TestResolveVoiceOrPath_KnownVoiceInManifest(t *testing.T) {
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
	// Voice path is relative to the manifest's directory (voices/).
	// VoiceManager resolves: filepath.Join("voices", "alice.bin") → "<tmp>/voices/alice.bin".
	if err := os.WriteFile(filepath.Join(voiceDir, "alice.bin"), []byte("voice-data"), 0o644); err != nil {
		t.Fatalf("WriteFile alice.bin: %v", err)
	}
	manifest := `{"voices":[{"id":"alice","path":"alice.bin","license":"MIT"}]}`
	if err := os.WriteFile(filepath.Join(voiceDir, "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	got, err := resolveVoiceOrPath("alice")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// VoiceManager.ResolvePath returns filepath.Clean(baseDir + path).
	// baseDir = filepath.Dir("voices/manifest.json") = "voices"
	// result  = filepath.Clean("voices/alice.bin") = "voices/alice.bin"
	wantPath := filepath.Join("voices", "alice.bin")
	if got != wantPath {
		t.Errorf("expected resolved path %q, got %q", wantPath, got)
	}
}

func TestResolveVoiceOrPath_UnknownVoicePassesThrough(t *testing.T) {
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
	manifest := `{"voices":[{"id":"alice","path":"voices/alice.bin","license":"MIT"}]}`
	if err := os.WriteFile(filepath.Join(voiceDir, "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	// "bob" is not in manifest → treated as raw CLI voice value
	got, err := resolveVoiceOrPath("bob")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "bob" {
		t.Errorf("expected passthrough %q, got %q", "bob", got)
	}
}
