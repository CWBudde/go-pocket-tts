package onnx

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func resetSessionOnceForTest() {
	sessionMgrOnce = sync.Once{}
	sessionMgr = nil
	errSessionMgr = nil
}

func TestNewSessionManagerLoadsManifest(t *testing.T) {
	tmp := t.TempDir()

	for _, name := range []string{"text_conditioner.onnx", "flow_lm_main.onnx"} {
		err := os.WriteFile(filepath.Join(tmp, name), []byte("fake"), 0o644)
		if err != nil {
			t.Fatalf("write fake onnx file: %v", err)
		}
	}

	manifest := `{
  "graphs": [
    {
      "name": "text_conditioner",
      "filename": "text_conditioner.onnx",
      "inputs": [{"name":"tokens","dtype":"int64","shape":[1,"text_tokens"]}],
      "outputs": [{"name":"text_embeddings","dtype":"float","shape":[1,"text_tokens",1024]}]
    },
    {
      "name": "flow_lm_main",
      "filename": "flow_lm_main.onnx",
      "inputs": [{"name":"sequence","dtype":"float","shape":[1,"sequence_steps",32]}],
      "outputs": [{"name":"last_hidden","dtype":"float","shape":[1,1024]}]
    }
  ]
}`

	manifestPath := filepath.Join(tmp, "manifest.json")

	err := os.WriteFile(manifestPath, []byte(manifest), 0o644)
	if err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	sm, err := NewSessionManager(manifestPath)
	if err != nil {
		t.Fatalf("NewSessionManager failed: %v", err)
	}

	all := sm.Sessions()
	if len(all) != 2 {
		t.Fatalf("expected 2 sessions, got %d", len(all))
	}

	s, ok := sm.Session("text_conditioner")
	if !ok {
		t.Fatal("expected text_conditioner session")
	}

	if s.Path != filepath.Join(tmp, "text_conditioner.onnx") {
		t.Fatalf("unexpected session path: %s", s.Path)
	}

	if len(s.Inputs) != 1 || s.Inputs[0].Name != "tokens" {
		t.Fatalf("unexpected inputs: %+v", s.Inputs)
	}
}

func TestNewSessionManagerRejectsMissingFile(t *testing.T) {
	tmp := t.TempDir()
	manifest := `{
  "graphs": [
    {"name": "missing", "filename": "missing.onnx", "inputs": [], "outputs": []}
  ]
}`

	manifestPath := filepath.Join(tmp, "manifest.json")

	err := os.WriteFile(manifestPath, []byte(manifest), 0o644)
	if err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	_, err = NewSessionManager(manifestPath)
	if err == nil {
		t.Fatal("expected error for missing onnx file")
	}
}

func TestLoadSessionsOnceKeepsFirstManifest(t *testing.T) {
	resetSessionOnceForTest()

	tmp := t.TempDir()

	firstFile := filepath.Join(tmp, "a.onnx")
	secondFile := filepath.Join(tmp, "b.onnx")

	err := os.WriteFile(firstFile, []byte("a"), 0o644)
	if err != nil {
		t.Fatalf("write first file: %v", err)
	}

	err = os.WriteFile(secondFile, []byte("b"), 0o644)
	if err != nil {
		t.Fatalf("write second file: %v", err)
	}

	firstManifest := filepath.Join(tmp, "first.json")
	secondManifest := filepath.Join(tmp, "second.json")

	err = os.WriteFile(firstManifest, []byte(`{"graphs":[{"name":"a","filename":"a.onnx","inputs":[],"outputs":[]}]}`), 0o644)
	if err != nil {
		t.Fatalf("write first manifest: %v", err)
	}

	err = os.WriteFile(secondManifest, []byte(`{"graphs":[{"name":"b","filename":"b.onnx","inputs":[],"outputs":[]}]}`), 0o644)
	if err != nil {
		t.Fatalf("write second manifest: %v", err)
	}

	one, err := LoadSessionsOnce(firstManifest)
	if err != nil {
		t.Fatalf("load first once: %v", err)
	}

	two, err := LoadSessionsOnce(secondManifest)
	if err != nil {
		t.Fatalf("load second once: %v", err)
	}

	if one != two {
		t.Fatal("expected same session manager pointer from once loader")
	}

	if _, ok := two.Session("a"); !ok {
		t.Fatal("expected to keep first loaded session set")
	}

	if _, ok := two.Session("b"); ok {
		t.Fatal("did not expect second manifest to replace first in once loader")
	}
}
