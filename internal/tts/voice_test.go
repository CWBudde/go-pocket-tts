package tts

import (
	"os"
	"path/filepath"
	"testing"
)

func TestVoiceManagerListAndResolve(t *testing.T) {
	tmp := t.TempDir()

	voiceFile := filepath.Join(tmp, "mimi.safetensors")

	err := os.WriteFile(voiceFile, []byte("voice"), 0o644)
	if err != nil {
		t.Fatalf("write voice file: %v", err)
	}

	manifestPath := filepath.Join(tmp, "manifest.json")

	manifest := `{
  "voices": [
    {"id": "mimi", "path": "mimi.safetensors", "license": "CC-BY-4.0"}
  ]
}`

	err = os.WriteFile(manifestPath, []byte(manifest), 0o644)
	if err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	mgr, err := NewVoiceManager(manifestPath)
	if err != nil {
		t.Fatalf("new voice manager: %v", err)
	}

	voices := mgr.ListVoices()
	if len(voices) != 1 {
		t.Fatalf("expected 1 voice, got %d", len(voices))
	}

	if voices[0].ID != "mimi" {
		t.Fatalf("unexpected voice id: %q", voices[0].ID)
	}

	resolved, err := mgr.ResolvePath("mimi")
	if err != nil {
		t.Fatalf("resolve voice path: %v", err)
	}

	if resolved != voiceFile {
		t.Fatalf("expected %q, got %q", voiceFile, resolved)
	}
}

func TestVoiceManagerResolveUnknownID(t *testing.T) {
	tmp := t.TempDir()
	manifestPath := filepath.Join(tmp, "manifest.json")

	manifest := `{"voices": [{"id": "mimi", "path": "mimi.safetensors", "license": "CC-BY-4.0"}]}`

	err := os.WriteFile(manifestPath, []byte(manifest), 0o644)
	if err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	mgr, err := NewVoiceManager(manifestPath)
	if err != nil {
		t.Fatalf("new voice manager: %v", err)
	}

	_, err = mgr.ResolvePath("unknown")
	if err == nil {
		t.Fatal("expected error for unknown voice id")
	}
}
