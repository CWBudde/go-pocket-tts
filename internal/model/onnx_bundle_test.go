package model

import (
	"archive/zip"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestResolveBundleFromLock_ByVariant(t *testing.T) {
	tmp := t.TempDir()
	lockPath := filepath.Join(tmp, "lock.json")
	lock := ONNXBundleLock{
		Version: 1,
		Bundles: []ONNXBundle{{
			ID:      "b6369a24-cpu",
			Variant: "b6369a24",
			URL:     "https://example.invalid/bundle.zip",
			SHA256:  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		}},
	}

	data, err := json.Marshal(lock)
	if err != nil {
		t.Fatalf("marshal lock: %v", err)
	}

	if err := os.WriteFile(lockPath, data, 0o644); err != nil {
		t.Fatalf("write lock: %v", err)
	}

	b, err := resolveBundleFromLock(lockPath, "", "b6369a24")
	if err != nil {
		t.Fatalf("resolve bundle: %v", err)
	}

	if b.ID != "b6369a24-cpu" {
		t.Fatalf("unexpected id: %s", b.ID)
	}
}

func TestVerifyONNXManifestDir(t *testing.T) {
	tmp := t.TempDir()
	for _, fn := range []string{
		"text_conditioner.onnx",
		"flow_lm_main.onnx",
		"flow_lm_flow.onnx",
		"mimi_decoder.onnx",
	} {
		err := os.WriteFile(filepath.Join(tmp, fn), []byte("x"), 0o644)
		if err != nil {
			t.Fatalf("write fake graph: %v", err)
		}
	}

	manifest := map[string]any{
		"graphs": []map[string]any{
			{"name": "text_conditioner", "filename": "text_conditioner.onnx"},
			{"name": "flow_lm_main", "filename": "flow_lm_main.onnx"},
			{"name": "flow_lm_flow", "filename": "flow_lm_flow.onnx"},
			{"name": "mimi_decoder", "filename": "mimi_decoder.onnx"},
		},
	}

	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}

	err = os.WriteFile(filepath.Join(tmp, "manifest.json"), data, 0o644)
	if err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	err = verifyONNXManifestDir(tmp)
	if err != nil {
		t.Fatalf("verify manifest dir: %v", err)
	}
}

func TestExtractBundle_Zip(t *testing.T) {
	tmp := t.TempDir()
	zipPath := filepath.Join(tmp, "bundle.zip")
	outDir := filepath.Join(tmp, "out")

	fh, err := os.Create(zipPath)
	if err != nil {
		t.Fatalf("create zip: %v", err)
	}

	zw := zip.NewWriter(fh)

	w, err := zw.Create("manifest.json")
	if err != nil {
		t.Fatalf("create zip entry: %v", err)
	}

	_, _ = w.Write([]byte(`{"graphs":[]}`))

	if err := zw.Close(); err != nil {
		t.Fatalf("close zip writer: %v", err)
	}

	if err := fh.Close(); err != nil {
		t.Fatalf("close zip file: %v", err)
	}

	if err := extractZip(zipPath, outDir); err != nil {
		t.Fatalf("extract zip: %v", err)
	}

	if _, err := os.Stat(filepath.Join(outDir, "manifest.json")); err != nil {
		t.Fatalf("expected extracted file: %v", err)
	}
}
