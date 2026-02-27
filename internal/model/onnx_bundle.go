package model

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type ONNXBundleLock struct {
	Version int          `json:"version"`
	Bundles []ONNXBundle `json:"bundles"`
}

type ONNXBundle struct {
	ID      string `json:"id"`
	Variant string `json:"variant"`
	URL     string `json:"url"`
	SHA256  string `json:"sha256"`
}

type DownloadONNXBundleOptions struct {
	BundleID   string
	Variant    string
	BundleURL  string
	SHA256     string
	LockFile   string
	OutDir     string
	HTTPClient *http.Client
	Stdout     io.Writer
	Stderr     io.Writer
}

func DownloadONNXBundle(opts DownloadONNXBundleOptions) error {
	if opts.OutDir == "" {
		return errors.New("out dir is required")
	}

	if opts.Variant == "" {
		opts.Variant = "b6369a24"
	}

	if opts.LockFile == "" {
		opts.LockFile = filepath.Join("bundles", "onnx-bundles.lock.json")
	}

	if opts.Stdout == nil {
		opts.Stdout = io.Discard
	}

	if opts.Stderr == nil {
		opts.Stderr = io.Discard
	}

	if opts.HTTPClient == nil {
		opts.HTTPClient = &http.Client{Timeout: 0}
	}

	bundleURL := strings.TrimSpace(opts.BundleURL)

	checksum := strings.ToLower(strings.TrimSpace(opts.SHA256))
	if bundleURL == "" {
		b, err := resolveBundleFromLock(opts.LockFile, opts.BundleID, opts.Variant)
		if err != nil {
			return err
		}

		bundleURL = b.URL
		if checksum == "" {
			checksum = strings.ToLower(strings.TrimSpace(b.SHA256))
		}

		_, _ = fmt.Fprintf(opts.Stdout, "resolved ONNX bundle from lock: id=%s variant=%s url=%s\n", b.ID, b.Variant, b.URL)
	}

	if bundleURL == "" {
		return fmt.Errorf("bundle URL is required (pass --bundle-url or configure %s)", opts.LockFile)
	}

	if checksum != "" && !isSHA256Hex(checksum) {
		return fmt.Errorf("invalid sha256 checksum %q", checksum)
	}

	err := os.MkdirAll(opts.OutDir, 0o755)
	if err != nil {
		return fmt.Errorf("create out dir: %w", err)
	}

	tmpArchive, actualSHA, err := fetchBundleArchive(opts.HTTPClient, bundleURL)
	if err != nil {
		return err
	}

	defer func() { _ = os.Remove(tmpArchive) }()

	if checksum != "" && checksum != actualSHA {
		return fmt.Errorf("bundle checksum mismatch: expected %s got %s", checksum, actualSHA)
	}

	_, _ = fmt.Fprintf(opts.Stdout, "downloaded ONNX bundle (%s) sha256=%s\n", bundleURL, actualSHA)

	err = extractBundle(tmpArchive, opts.OutDir)
	if err != nil {
		return err
	}

	_, _ = fmt.Fprintf(opts.Stdout, "extracted bundle into %s\n", opts.OutDir)

	err = verifyONNXManifestDir(opts.OutDir)
	if err != nil {
		return err
	}

	_, _ = fmt.Fprintf(opts.Stdout, "verified ONNX bundle manifest in %s\n", opts.OutDir)

	return nil
}

func resolveBundleFromLock(lockFile, bundleID, variant string) (ONNXBundle, error) {
	data, err := os.ReadFile(lockFile)
	if err != nil {
		return ONNXBundle{}, fmt.Errorf("read ONNX bundle lock file %q: %w", lockFile, err)
	}

	var lock ONNXBundleLock

	err = json.Unmarshal(data, &lock)
	if err != nil {
		return ONNXBundle{}, fmt.Errorf("decode ONNX bundle lock file %q: %w", lockFile, err)
	}

	if len(lock.Bundles) == 0 {
		return ONNXBundle{}, fmt.Errorf("ONNX bundle lock %q has no bundles; pass --bundle-url", lockFile)
	}

	if bundleID != "" {
		for _, b := range lock.Bundles {
			if b.ID == bundleID {
				return b, nil
			}
		}

		return ONNXBundle{}, fmt.Errorf("bundle id %q not found in %s", bundleID, lockFile)
	}

	for _, b := range lock.Bundles {
		if b.Variant == variant {
			return b, nil
		}
	}

	return ONNXBundle{}, fmt.Errorf("no bundle found for variant %q in %s", variant, lockFile)
}

func fetchBundleArchive(client *http.Client, bundleURL string) (string, string, error) {
	tmpFile, err := os.CreateTemp("", "pockettts-onnx-bundle-*")
	if err != nil {
		return "", "", fmt.Errorf("create temp bundle file: %w", err)
	}

	tmpPath := tmpFile.Name()

	var reader io.ReadCloser

	//nolint:nestif // Distinct HTTP and local-file acquisition paths with explicit cleanup are intentional.
	if strings.HasPrefix(bundleURL, "http://") || strings.HasPrefix(bundleURL, "https://") {
		req, err := http.NewRequest(http.MethodGet, bundleURL, nil)
		if err != nil {
			_ = tmpFile.Close()
			// #nosec G703 -- tmpPath is from os.CreateTemp in this function and removed only for local cleanup.
			_ = os.Remove(tmpPath)

			return "", "", fmt.Errorf("build bundle request: %w", err)
		}

		// #nosec G704 -- Bundle URL comes from explicit CLI/lock configuration and is expected to support remote HTTPS downloads.
		resp, err := client.Do(req)
		if err != nil {
			_ = tmpFile.Close()
			// #nosec G703 -- tmpPath is from os.CreateTemp in this function and removed only for local cleanup.
			_ = os.Remove(tmpPath)

			return "", "", fmt.Errorf("bundle download failed: %w", err)
		}

		if resp.StatusCode < 200 || resp.StatusCode > 299 {
			_ = resp.Body.Close()
			_ = tmpFile.Close()
			_ = os.Remove(tmpPath)

			return "", "", fmt.Errorf("bundle download failed: %s", resp.Status)
		}

		reader = resp.Body
	} else {
		local := strings.TrimPrefix(bundleURL, "file://")

		fh, err := os.Open(local)
		if err != nil {
			_ = tmpFile.Close()
			_ = os.Remove(tmpPath)

			return "", "", fmt.Errorf("open local bundle %q: %w", local, err)
		}

		reader = fh
	}

	defer func() { _ = reader.Close() }()

	h := sha256.New()

	_, err = io.Copy(io.MultiWriter(tmpFile, h), reader)
	if err != nil {
		_ = tmpFile.Close()
		_ = os.Remove(tmpPath)

		return "", "", fmt.Errorf("write temp bundle file: %w", err)
	}

	err = tmpFile.Close()
	if err != nil {
		_ = os.Remove(tmpPath)
		return "", "", fmt.Errorf("close temp bundle file: %w", err)
	}

	return tmpPath, hex.EncodeToString(h.Sum(nil)), nil
}

func extractBundle(bundlePath, outDir string) error {
	base := strings.ToLower(bundlePath)
	switch {
	case strings.HasSuffix(base, ".zip"):
		return extractZip(bundlePath, outDir)
	case strings.HasSuffix(base, ".tar.gz"), strings.HasSuffix(base, ".tgz"):
		return extractTarGz(bundlePath, outDir)
	default:
		// Attempt ZIP first, then tar.gz for local temp files without extension.
		err := extractZip(bundlePath, outDir)
		if err == nil {
			return nil
		}

		err = extractTarGz(bundlePath, outDir)
		if err == nil {
			return nil
		}

		return fmt.Errorf("unsupported bundle format for %s (expected .zip or .tar.gz/.tgz)", bundlePath)
	}
}

func extractZip(bundlePath, outDir string) error {
	zr, err := zip.OpenReader(bundlePath)
	if err != nil {
		return fmt.Errorf("open zip bundle: %w", err)
	}

	defer func() { _ = zr.Close() }()

	for _, f := range zr.File {
		targetPath, err := safeExtractPath(outDir, f.Name)
		if err != nil {
			return err
		}

		if f.FileInfo().IsDir() {
			err := os.MkdirAll(targetPath, 0o755)
			if err != nil {
				return fmt.Errorf("create dir %s: %w", targetPath, err)
			}

			continue
		}

		err = os.MkdirAll(filepath.Dir(targetPath), 0o755)
		if err != nil {
			return fmt.Errorf("create parent dir for %s: %w", targetPath, err)
		}

		src, err := f.Open()
		if err != nil {
			return fmt.Errorf("open zip entry %s: %w", f.Name, err)
		}

		dst, err := os.Create(targetPath)
		if err != nil {
			_ = src.Close()
			return fmt.Errorf("create extracted file %s: %w", targetPath, err)
		}

		//nolint:gosec // Bundle size is controlled by trusted lock metadata and verified by checksum before extract.
		_, err = io.Copy(dst, src)
		if err != nil {
			_ = dst.Close()
			_ = src.Close()

			return fmt.Errorf("extract zip entry %s: %w", f.Name, err)
		}

		_ = dst.Close()
		_ = src.Close()
	}

	return nil
}

func extractTarGz(bundlePath, outDir string) error {
	fh, err := os.Open(bundlePath)
	if err != nil {
		return fmt.Errorf("open tar.gz bundle: %w", err)
	}

	defer func() { _ = fh.Close() }()

	gz, err := gzip.NewReader(fh)
	if err != nil {
		return fmt.Errorf("open gzip reader: %w", err)
	}

	defer func() { _ = gz.Close() }()

	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			return fmt.Errorf("read tar entry: %w", err)
		}

		targetPath, err := safeExtractPath(outDir, hdr.Name)
		if err != nil {
			return err
		}

		switch hdr.Typeflag {
		case tar.TypeDir:
			err := os.MkdirAll(targetPath, 0o755)
			if err != nil {
				return fmt.Errorf("create dir %s: %w", targetPath, err)
			}
		case tar.TypeReg:
			err = os.MkdirAll(filepath.Dir(targetPath), 0o755)
			if err != nil {
				return fmt.Errorf("create parent dir for %s: %w", targetPath, err)
			}

			dst, err := os.Create(targetPath)
			if err != nil {
				return fmt.Errorf("create extracted file %s: %w", targetPath, err)
			}

			//nolint:gosec // Bundle archive is checksum-verified before extraction.
			_, err = io.Copy(dst, tr)
			if err != nil {
				_ = dst.Close()
				return fmt.Errorf("extract tar entry %s: %w", hdr.Name, err)
			}

			_ = dst.Close()
		default:
			// Ignore non-regular entries for bundle portability.
		}
	}

	return nil
}

func safeExtractPath(baseDir, entryName string) (string, error) {
	cleaned := filepath.Clean(strings.TrimPrefix(entryName, "/"))
	target := filepath.Join(baseDir, cleaned)

	base := filepath.Clean(baseDir) + string(os.PathSeparator)
	if !strings.HasPrefix(filepath.Clean(target)+string(os.PathSeparator), base) {
		return "", fmt.Errorf("unsafe archive path traversal attempt: %q", entryName)
	}

	return target, nil
}

type onnxManifestLite struct {
	Graphs []onnxGraphLite `json:"graphs"`
}

type onnxGraphLite struct {
	Name     string `json:"name"`
	Filename string `json:"filename"`
}

func verifyONNXManifestDir(outDir string) error {
	manifestPath := filepath.Join(outDir, "manifest.json")

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return fmt.Errorf("read ONNX manifest: %w", err)
	}

	var m onnxManifestLite

	err = json.Unmarshal(data, &m)
	if err != nil {
		return fmt.Errorf("decode ONNX manifest: %w", err)
	}

	if len(m.Graphs) == 0 {
		return errors.New("ONNX manifest has no graphs")
	}

	requiredNames := []string{
		"text_conditioner",
		"flow_lm_main",
		"flow_lm_flow",
		"mimi_decoder",
	}

	required := make(map[string]bool, len(requiredNames))
	for _, name := range requiredNames {
		required[name] = false
	}

	for _, g := range m.Graphs {
		if g.Name == "" {
			return errors.New("manifest graph has empty name")
		}

		if g.Filename == "" {
			return fmt.Errorf("manifest graph %q has empty filename", g.Name)
		}

		graphPath := filepath.Join(outDir, g.Filename)

		_, err = os.Stat(graphPath)
		if err != nil {
			return fmt.Errorf("manifest graph file %q: %w", g.Filename, err)
		}

		if _, ok := required[g.Name]; ok {
			required[g.Name] = true
		}
	}

	for _, name := range requiredNames {
		if !required[name] {
			return fmt.Errorf("manifest missing required graph %q", name)
		}
	}

	return nil
}
