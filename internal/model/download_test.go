package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPinnedManifestDefaultRepo(t *testing.T) {
	m, err := PinnedManifest("kyutai/pocket-tts")
	if err != nil {
		t.Fatalf("manifest error: %v", err)
	}
	if len(m.Files) == 0 {
		t.Fatal("expected files in manifest")
	}
	if m.Files[0].Filename == "" || m.Files[0].Revision == "" {
		t.Fatal("expected filename and revision")
	}
}

func TestNormalizeETag(t *testing.T) {
	got := normalizeETag(`W/"58aa704a88faad35f22c34ea1cb55c4c5629de8b8e035c6e4936e2673dc07617"`)
	want := "58aa704a88faad35f22c34ea1cb55c4c5629de8b8e035c6e4936e2673dc07617"
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
	if !isSHA256Hex(got) {
		t.Fatalf("expected valid sha256")
	}
}

func TestExistingMatches(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "x.bin")
	if err := os.WriteFile(p, []byte("hello"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	ok, err := existingMatches(p, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
	if err != nil {
		t.Fatalf("existingMatches error: %v", err)
	}
	if !ok {
		t.Fatal("expected checksum match")
	}
}
