package model

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveBundleFromLock_ByVariant(t *testing.T) {
	tmp := t.TempDir()
	lockPath := filepath.Join(tmp, "lock.json")
	writeLockFile(t, lockPath, ONNXBundleLock{
		Version: 1,
		Bundles: []ONNXBundle{{
			ID:      "b6369a24-cpu",
			Variant: "b6369a24",
			URL:     "https://example.invalid/bundle.zip",
			SHA256:  strings.Repeat("a", 64),
		}},
	})

	b, err := resolveBundleFromLock(lockPath, "", "b6369a24")
	if err != nil {
		t.Fatalf("resolve bundle: %v", err)
	}

	if b.ID != "b6369a24-cpu" {
		t.Fatalf("unexpected id: %s", b.ID)
	}
}

func TestResolveBundleFromLock_ByID(t *testing.T) {
	tmp := t.TempDir()
	lockPath := filepath.Join(tmp, "lock.json")
	writeLockFile(t, lockPath, ONNXBundleLock{
		Version: 1,
		Bundles: []ONNXBundle{
			{
				ID:      "cpu",
				Variant: "a",
				URL:     "https://example.invalid/a.zip",
				SHA256:  strings.Repeat("a", 64),
			},
			{
				ID:      "gpu",
				Variant: "b",
				URL:     "https://example.invalid/b.zip",
				SHA256:  strings.Repeat("b", 64),
			},
		},
	})

	b, err := resolveBundleFromLock(lockPath, "gpu", "ignored")
	if err != nil {
		t.Fatalf("resolve bundle: %v", err)
	}

	if b.URL != "https://example.invalid/b.zip" {
		t.Fatalf("unexpected URL: %s", b.URL)
	}
}

func TestResolveBundleFromLock_Errors(t *testing.T) {
	tmp := t.TempDir()
	lockPath := filepath.Join(tmp, "lock.json")

	writeLockFile(t, lockPath, ONNXBundleLock{Version: 1, Bundles: nil})

	_, err := resolveBundleFromLock(lockPath, "", "x")
	if err == nil || !strings.Contains(err.Error(), "has no bundles") {
		t.Fatalf("expected no-bundles error, got: %v", err)
	}

	writeLockFile(t, lockPath, ONNXBundleLock{
		Version: 1,
		Bundles: []ONNXBundle{{ID: "cpu", Variant: "a", URL: "x"}},
	})

	_, err = resolveBundleFromLock(lockPath, "missing", "a")
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("expected missing id error, got: %v", err)
	}

	_, err = resolveBundleFromLock(lockPath, "", "missing-variant")
	if err == nil || !strings.Contains(err.Error(), "no bundle found for variant") {
		t.Fatalf("expected missing variant error, got: %v", err)
	}

	err = os.WriteFile(lockPath, []byte("{not-json"), 0o644)
	if err != nil {
		t.Fatalf("write invalid lock: %v", err)
	}

	_, err = resolveBundleFromLock(lockPath, "", "x")
	if err == nil || !strings.Contains(err.Error(), "decode ONNX bundle lock file") {
		t.Fatalf("expected decode error, got: %v", err)
	}
}

func TestFetchBundleArchive_LocalAndFileURL(t *testing.T) {
	tmp := t.TempDir()
	src := filepath.Join(tmp, "bundle.zip")
	content := []byte("bundle-bytes")

	err := os.WriteFile(src, content, 0o644)
	if err != nil {
		t.Fatalf("write source file: %v", err)
	}

	want := sha256Hex(content)

	tmpPath, gotSHA, err := fetchBundleArchive(http.DefaultClient, src)
	if err != nil {
		t.Fatalf("fetch local bundle: %v", err)
	}

	defer func() { _ = os.Remove(tmpPath) }()

	gotContent, err := os.ReadFile(tmpPath)
	if err != nil {
		t.Fatalf("read fetched temp file: %v", err)
	}

	if !bytes.Equal(gotContent, content) {
		t.Fatalf("copied data mismatch")
	}

	if gotSHA != want {
		t.Fatalf("sha mismatch: got %s want %s", gotSHA, want)
	}

	tmpPath2, gotSHA2, err := fetchBundleArchive(http.DefaultClient, "file://"+src)
	if err != nil {
		t.Fatalf("fetch file:// bundle: %v", err)
	}

	defer func() { _ = os.Remove(tmpPath2) }()

	if gotSHA2 != want {
		t.Fatalf("sha mismatch for file://: got %s want %s", gotSHA2, want)
	}
}

func TestFetchBundleArchive_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()

	_, _, err := fetchBundleArchive(http.DefaultClient, srv.URL)
	if err == nil || !strings.Contains(err.Error(), "bundle download failed") {
		t.Fatalf("expected HTTP download error, got: %v", err)
	}
}

func TestExtractBundle_Zip(t *testing.T) {
	tmp := t.TempDir()
	zipPath := filepath.Join(tmp, "bundle.zip")
	outDir := filepath.Join(tmp, "out")

	writeZipArchive(t, zipPath, map[string][]byte{
		"manifest.json": []byte(`{"graphs":[]}`),
	})

	err := extractZip(zipPath, outDir)
	if err != nil {
		t.Fatalf("extract zip: %v", err)
	}

	_, err = os.Stat(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatalf("expected extracted file: %v", err)
	}
}

func TestExtractBundle_TarGz(t *testing.T) {
	tmp := t.TempDir()
	tarPath := filepath.Join(tmp, "bundle.tar.gz")
	outDir := filepath.Join(tmp, "out")

	writeTarGzArchive(t, tarPath, map[string][]byte{
		"manifest.json": []byte(`{"graphs":[]}`),
	})

	err := extractTarGz(tarPath, outDir)
	if err != nil {
		t.Fatalf("extract tar.gz: %v", err)
	}

	_, err = os.Stat(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatalf("expected extracted file: %v", err)
	}
}

func TestExtractBundle_NoExtensionFallsBack(t *testing.T) {
	tmp := t.TempDir()
	noExtPath := filepath.Join(tmp, "bundle.bin")
	outDir := filepath.Join(tmp, "out")

	writeZipArchive(t, noExtPath, map[string][]byte{
		"manifest.json": []byte(`{"graphs":[]}`),
	})

	err := extractBundle(noExtPath, outDir)
	if err != nil {
		t.Fatalf("extract fallback zip failed: %v", err)
	}

	_, err = os.Stat(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatalf("expected extracted file: %v", err)
	}
}

func TestExtractBundle_UnsafePathRejected(t *testing.T) {
	tmp := t.TempDir()
	zipPath := filepath.Join(tmp, "bundle.zip")
	outDir := filepath.Join(tmp, "out")

	writeZipArchive(t, zipPath, map[string][]byte{
		"../escape.txt": []byte("x"),
	})

	err := extractBundle(zipPath, outDir)
	if err == nil || !strings.Contains(err.Error(), "unsafe archive path traversal attempt") {
		t.Fatalf("expected traversal error, got: %v", err)
	}
}

func TestExtractBundle_UnsupportedFormat(t *testing.T) {
	tmp := t.TempDir()
	raw := filepath.Join(tmp, "bundle.raw")
	outDir := filepath.Join(tmp, "out")

	err := os.WriteFile(raw, []byte("not-an-archive"), 0o644)
	if err != nil {
		t.Fatalf("write raw file: %v", err)
	}

	err = extractBundle(raw, outDir)
	if err == nil || !strings.Contains(err.Error(), "unsupported bundle format") {
		t.Fatalf("expected unsupported format error, got: %v", err)
	}
}

func TestDownloadONNXBundle_FromLockAndFileURL(t *testing.T) {
	tmp := t.TempDir()
	outDir := filepath.Join(tmp, "out")
	archivePath := filepath.Join(tmp, "bundle.zip")
	lockPath := filepath.Join(tmp, "lock.json")

	files := validBundleFiles(t)
	writeZipArchive(t, archivePath, files)
	archiveSHA := sha256OfFile(t, archivePath)

	writeLockFile(t, lockPath, ONNXBundleLock{
		Version: 1,
		Bundles: []ONNXBundle{{
			ID:      "bundle-cpu",
			Variant: "b6369a24",
			URL:     "file://" + archivePath,
			SHA256:  archiveSHA,
		}},
	})

	var stdout bytes.Buffer

	err := DownloadONNXBundle(DownloadONNXBundleOptions{
		LockFile: lockPath,
		OutDir:   outDir,
		Stdout:   &stdout,
	})
	if err != nil {
		t.Fatalf("DownloadONNXBundle from lock: %v", err)
	}

	_, err = os.Stat(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatalf("missing extracted manifest: %v", err)
	}

	if !strings.Contains(stdout.String(), "resolved ONNX bundle from lock") {
		t.Fatalf("expected lock resolution log in stdout, got: %q", stdout.String())
	}

	if !strings.Contains(stdout.String(), "verified ONNX bundle manifest") {
		t.Fatalf("expected verify log in stdout, got: %q", stdout.String())
	}
}

func TestDownloadONNXBundle_HTTP(t *testing.T) {
	tmp := t.TempDir()
	outDir := filepath.Join(tmp, "out")
	archivePath := filepath.Join(tmp, "bundle.zip")
	files := validBundleFiles(t)
	writeZipArchive(t, archivePath, files)

	bundleBytes, err := os.ReadFile(archivePath)
	if err != nil {
		t.Fatalf("read bundle: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(bundleBytes)
	}))
	defer srv.Close()

	err = DownloadONNXBundle(DownloadONNXBundleOptions{
		OutDir:    outDir,
		BundleURL: srv.URL + "/bundle.zip",
		SHA256:    sha256Hex(bundleBytes),
	})
	if err != nil {
		t.Fatalf("DownloadONNXBundle HTTP: %v", err)
	}

	_, err = os.Stat(filepath.Join(outDir, "mimi_decoder.onnx"))
	if err != nil {
		t.Fatalf("missing extracted graph: %v", err)
	}
}

func TestDownloadONNXBundle_Errors(t *testing.T) {
	tmp := t.TempDir()
	archivePath := filepath.Join(tmp, "bundle.zip")
	writeZipArchive(t, archivePath, validBundleFiles(t))

	err := DownloadONNXBundle(DownloadONNXBundleOptions{})
	if err == nil || !strings.Contains(err.Error(), "out dir is required") {
		t.Fatalf("expected out dir error, got: %v", err)
	}

	err = DownloadONNXBundle(DownloadONNXBundleOptions{
		OutDir:    filepath.Join(tmp, "out-a"),
		BundleURL: "file://" + archivePath,
		SHA256:    "not-a-sha",
	})
	if err == nil || !strings.Contains(err.Error(), "invalid sha256 checksum") {
		t.Fatalf("expected invalid sha error, got: %v", err)
	}

	err = DownloadONNXBundle(DownloadONNXBundleOptions{
		OutDir:    filepath.Join(tmp, "out-b"),
		BundleURL: "file://" + archivePath,
		SHA256:    strings.Repeat("0", 64),
	})
	if err == nil || !strings.Contains(err.Error(), "bundle checksum mismatch") {
		t.Fatalf("expected checksum mismatch error, got: %v", err)
	}
}

func TestVerifyONNXManifestDir(t *testing.T) {
	tmp := t.TempDir()

	for path, content := range validBundleFiles(t) {
		target := filepath.Join(tmp, path)

		err := os.MkdirAll(filepath.Dir(target), 0o755)
		if err != nil {
			t.Fatalf("mkdir parent: %v", err)
		}

		err = os.WriteFile(target, content, 0o644)
		if err != nil {
			t.Fatalf("write file %s: %v", path, err)
		}
	}

	err := verifyONNXManifestDir(tmp)
	if err != nil {
		t.Fatalf("verify manifest dir: %v", err)
	}
}

func TestVerifyONNXManifestDir_Errors(t *testing.T) {
	tmp := t.TempDir()

	err := os.WriteFile(filepath.Join(tmp, "manifest.json"), []byte(`{"graphs":[]}`), 0o644)
	if err != nil {
		t.Fatalf("write empty manifest: %v", err)
	}

	err = verifyONNXManifestDir(tmp)
	if err == nil || !strings.Contains(err.Error(), "has no graphs") {
		t.Fatalf("expected no-graphs error, got: %v", err)
	}

	manifest := `{"graphs":[{"name":"text_conditioner","filename":"text_conditioner.onnx"}]}`

	err = os.WriteFile(filepath.Join(tmp, "manifest.json"), []byte(manifest), 0o644)
	if err != nil {
		t.Fatalf("write incomplete manifest: %v", err)
	}

	err = os.WriteFile(filepath.Join(tmp, "text_conditioner.onnx"), []byte("x"), 0o644)
	if err != nil {
		t.Fatalf("write graph file: %v", err)
	}

	err = verifyONNXManifestDir(tmp)
	if err == nil || !strings.Contains(err.Error(), `manifest missing required graph "flow_lm_main"`) {
		t.Fatalf("expected missing required graph error, got: %v", err)
	}
}

func writeLockFile(t *testing.T, path string, lock ONNXBundleLock) {
	t.Helper()

	data, err := json.Marshal(lock)
	if err != nil {
		t.Fatalf("marshal lock: %v", err)
	}

	err = os.WriteFile(path, data, 0o644)
	if err != nil {
		t.Fatalf("write lock: %v", err)
	}
}

func writeZipArchive(t *testing.T, path string, files map[string][]byte) {
	t.Helper()

	fh, err := os.Create(path)
	if err != nil {
		t.Fatalf("create zip %s: %v", path, err)
	}

	zw := zip.NewWriter(fh)
	for name, content := range files {
		w, err := zw.Create(name)
		if err != nil {
			_ = zw.Close()
			_ = fh.Close()

			t.Fatalf("create zip entry %s: %v", name, err)
		}

		_, err = w.Write(content)
		if err != nil {
			_ = zw.Close()
			_ = fh.Close()

			t.Fatalf("write zip entry %s: %v", name, err)
		}
	}

	err = zw.Close()
	if err != nil {
		_ = fh.Close()

		t.Fatalf("close zip writer: %v", err)
	}

	err = fh.Close()
	if err != nil {
		t.Fatalf("close zip file: %v", err)
	}
}

func writeTarGzArchive(t *testing.T, path string, files map[string][]byte) {
	t.Helper()

	fh, err := os.Create(path)
	if err != nil {
		t.Fatalf("create tar.gz %s: %v", path, err)
	}

	gw := gzip.NewWriter(fh)
	tw := tar.NewWriter(gw)

	for name, content := range files {
		h := &tar.Header{
			Name: name,
			Mode: 0o644,
			Size: int64(len(content)),
		}

		err = tw.WriteHeader(h)
		if err != nil {
			_ = tw.Close()
			_ = gw.Close()
			_ = fh.Close()

			t.Fatalf("write tar header %s: %v", name, err)
		}

		_, err = tw.Write(content)
		if err != nil {
			_ = tw.Close()
			_ = gw.Close()
			_ = fh.Close()

			t.Fatalf("write tar body %s: %v", name, err)
		}
	}

	err = tw.Close()
	if err != nil {
		_ = gw.Close()
		_ = fh.Close()

		t.Fatalf("close tar writer: %v", err)
	}

	err = gw.Close()
	if err != nil {
		_ = fh.Close()

		t.Fatalf("close gzip writer: %v", err)
	}

	err = fh.Close()
	if err != nil {
		t.Fatalf("close file: %v", err)
	}
}

func validBundleFiles(t *testing.T) map[string][]byte {
	t.Helper()

	manifest := map[string]any{
		"graphs": []map[string]string{
			{"name": "text_conditioner", "filename": "text_conditioner.onnx"},
			{"name": "flow_lm_main", "filename": "flow_lm_main.onnx"},
			{"name": "flow_lm_flow", "filename": "flow_lm_flow.onnx"},
			{"name": "mimi_decoder", "filename": "mimi_decoder.onnx"},
		},
	}

	manifestBytes, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}

	return map[string][]byte{
		"manifest.json":         manifestBytes,
		"text_conditioner.onnx": []byte("tc"),
		"flow_lm_main.onnx":     []byte("main"),
		"flow_lm_flow.onnx":     []byte("flow"),
		"mimi_decoder.onnx":     []byte("mimi"),
	}
}

func sha256OfFile(t *testing.T, path string) string {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file for sha: %v", err)
	}

	return sha256Hex(data)
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func TestExtractTarGz_UnsafePathRejected(t *testing.T) {
	tmp := t.TempDir()
	tarPath := filepath.Join(tmp, "bundle.tar.gz")
	outDir := filepath.Join(tmp, "out")

	var b bytes.Buffer
	gw := gzip.NewWriter(&b)
	tw := tar.NewWriter(gw)

	err := tw.WriteHeader(&tar.Header{Name: "../escape", Mode: 0o644, Size: int64(len("x"))})
	if err != nil {
		t.Fatalf("write tar header: %v", err)
	}

	_, err = io.WriteString(tw, "x")
	if err != nil {
		t.Fatalf("write tar body: %v", err)
	}

	err = tw.Close()
	if err != nil {
		t.Fatalf("close tar writer: %v", err)
	}

	err = gw.Close()
	if err != nil {
		t.Fatalf("close gzip writer: %v", err)
	}

	err = os.WriteFile(tarPath, b.Bytes(), 0o644)
	if err != nil {
		t.Fatalf("write tar.gz: %v", err)
	}

	err = extractTarGz(tarPath, outDir)
	if err == nil || !strings.Contains(err.Error(), "unsafe archive path traversal attempt") {
		t.Fatalf("expected traversal error, got: %v", err)
	}
}
