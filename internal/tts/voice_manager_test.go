package tts

import (
	"os"
	"path/filepath"
	"testing"
)

// --- NewVoiceManager error paths ---

func TestNewVoiceManager_EmptyPath(t *testing.T) {
	_, err := NewVoiceManager("")
	if err == nil {
		t.Error("NewVoiceManager(\"\") = nil; want error")
	}
}

func TestNewVoiceManager_MissingFile(t *testing.T) {
	_, err := NewVoiceManager("/nonexistent/manifest.json")
	if err == nil {
		t.Error("NewVoiceManager(missing) = nil; want error")
	}
}

func TestNewVoiceManager_InvalidJSON(t *testing.T) {
	tmp := t.TempDir()
	manifestPath := filepath.Join(tmp, "manifest.json")
	if err := os.WriteFile(manifestPath, []byte("{bad json"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err := NewVoiceManager(manifestPath)
	if err == nil {
		t.Error("NewVoiceManager(invalid json) = nil; want error")
	}
}

func TestNewVoiceManager_EmptyVoiceID(t *testing.T) {
	tmp := t.TempDir()
	manifestPath := filepath.Join(tmp, "manifest.json")
	manifest := `{"voices":[{"id":"","path":"v.bin","license":""}]}`
	if err := os.WriteFile(manifestPath, []byte(manifest), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err := NewVoiceManager(manifestPath)
	if err == nil {
		t.Error("NewVoiceManager(empty id) = nil; want error")
	}
}

func TestNewVoiceManager_EmptyVoicePath(t *testing.T) {
	tmp := t.TempDir()
	manifestPath := filepath.Join(tmp, "manifest.json")
	manifest := `{"voices":[{"id":"v1","path":"","license":""}]}`
	if err := os.WriteFile(manifestPath, []byte(manifest), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err := NewVoiceManager(manifestPath)
	if err == nil {
		t.Error("NewVoiceManager(empty path) = nil; want error")
	}
}

func TestNewVoiceManager_DuplicateID(t *testing.T) {
	tmp := t.TempDir()
	manifestPath := filepath.Join(tmp, "manifest.json")
	manifest := `{"voices":[
		{"id":"v1","path":"a.bin","license":""},
		{"id":"v1","path":"b.bin","license":""}
	]}`
	if err := os.WriteFile(manifestPath, []byte(manifest), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	_, err := NewVoiceManager(manifestPath)
	if err == nil {
		t.Error("NewVoiceManager(duplicate id) = nil; want error")
	}
}

func TestNewVoiceManager_EmptyVoicesList(t *testing.T) {
	tmp := t.TempDir()
	manifestPath := filepath.Join(tmp, "manifest.json")
	manifest := `{"voices":[]}`
	if err := os.WriteFile(manifestPath, []byte(manifest), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	mgr, err := NewVoiceManager(manifestPath)
	if err != nil {
		t.Fatalf("NewVoiceManager(empty list) error = %v", err)
	}
	if len(mgr.ListVoices()) != 0 {
		t.Error("expected empty voice list")
	}
}

// --- ResolvePath ---

func TestResolvePath_AbsolutePath(t *testing.T) {
	tmp := t.TempDir()
	voiceFile := filepath.Join(tmp, "voice.bin")
	if err := os.WriteFile(voiceFile, []byte("data"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	manifestPath := filepath.Join(tmp, "manifest.json")
	manifest := `{"voices":[{"id":"v1","path":"` + voiceFile + `","license":""}]}`
	if err := os.WriteFile(manifestPath, []byte(manifest), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	mgr, err := NewVoiceManager(manifestPath)
	if err != nil {
		t.Fatalf("NewVoiceManager: %v", err)
	}

	got, err := mgr.ResolvePath("v1")
	if err != nil {
		t.Fatalf("ResolvePath error = %v", err)
	}
	if got != voiceFile {
		t.Errorf("ResolvePath = %q; want %q", got, voiceFile)
	}
}

func TestResolvePath_MissingVoiceFile(t *testing.T) {
	tmp := t.TempDir()
	manifestPath := filepath.Join(tmp, "manifest.json")
	// Path is relative but the file does not exist on disk.
	manifest := `{"voices":[{"id":"v1","path":"missing.bin","license":""}]}`
	if err := os.WriteFile(manifestPath, []byte(manifest), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	mgr, err := NewVoiceManager(manifestPath)
	if err != nil {
		t.Fatalf("NewVoiceManager: %v", err)
	}

	_, err = mgr.ResolvePath("v1")
	if err == nil {
		t.Error("ResolvePath(missing file) = nil; want error")
	}
}

// --- ListVoices returns independent copies ---

func TestListVoices_ReturnsCopy(t *testing.T) {
	tmp := t.TempDir()
	voiceFile := filepath.Join(tmp, "v.bin")
	os.WriteFile(voiceFile, []byte("data"), 0o644)

	manifestPath := filepath.Join(tmp, "manifest.json")
	manifest := `{"voices":[{"id":"v1","path":"v.bin","license":"MIT"}]}`
	os.WriteFile(manifestPath, []byte(manifest), 0o644)

	mgr, _ := NewVoiceManager(manifestPath)
	first := mgr.ListVoices()
	first[0].ID = "mutated"

	second := mgr.ListVoices()
	if second[0].ID != "v1" {
		t.Error("ListVoices did not return an independent copy")
	}
}
