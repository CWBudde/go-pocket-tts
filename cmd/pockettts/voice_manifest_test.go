package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// writeVoiceManifest writes a manifest declaring voice "alice" outside the
// default voices/ directory and returns the manifest path and the voice file.
func writeVoiceManifest(t *testing.T) (manifestPath, voiceFile string) {
	t.Helper()

	dir := filepath.Join(t.TempDir(), "v", "de")

	err := os.MkdirAll(dir, 0o755)
	if err != nil {
		t.Fatal(err)
	}

	voiceFile = filepath.Join(dir, "alice.safetensors")

	err = os.WriteFile(voiceFile, []byte("voice-data"), 0o644)
	if err != nil {
		t.Fatal(err)
	}

	manifestPath = filepath.Join(dir, "manifest.json")

	err = os.WriteFile(manifestPath,
		[]byte(`{"voices":[{"id":"alice","path":"alice.safetensors","license":"MIT"}]}`), 0o644)
	if err != nil {
		t.Fatal(err)
	}

	return manifestPath, voiceFile
}

func TestResolveVoiceForNative_ManifestPath(t *testing.T) {
	manifestPath, voiceFile := writeVoiceManifest(t)

	got, err := resolveVoiceForNative(manifestPath, "alice")
	if err != nil {
		t.Fatal(err)
	}

	if got != voiceFile {
		t.Errorf("resolveVoiceForNative = %q; want %q", got, voiceFile)
	}
}

func TestResolveVoiceOrPath_ManifestPath(t *testing.T) {
	manifestPath, voiceFile := writeVoiceManifest(t)

	got, err := resolveVoiceOrPath(manifestPath, "alice")
	if err != nil {
		t.Fatal(err)
	}

	if got != voiceFile {
		t.Errorf("resolveVoiceOrPath = %q; want %q", got, voiceFile)
	}
}

func TestCollectVoiceFiles_ManifestPath(t *testing.T) {
	manifestPath, voiceFile := writeVoiceManifest(t)

	files := collectVoiceFiles(manifestPath)
	if !slices.Equal(files, []string{voiceFile}) {
		t.Errorf("collectVoiceFiles = %v; want [%s]", files, voiceFile)
	}
}
