package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

type DownloadOptions struct {
	Repo    string
	OutDir  string
	HFToken string
	Stdout  io.Writer
	Stderr  io.Writer
}

type ErrAccessDenied struct {
	Repo string
	Msg  string
}

func (e *ErrAccessDenied) Error() string {
	if e.Msg != "" {
		return e.Msg
	}
	return fmt.Sprintf("access denied for %s", e.Repo)
}

type lockManifest struct {
	Repo      string                `json:"repo"`
	Generated string                `json:"generated"`
	Files     map[string]lockRecord `json:"files"`
}

type lockRecord struct {
	Revision string `json:"revision"`
	SHA256   string `json:"sha256"`
}

var shaHexPattern = regexp.MustCompile(`(?i)^[a-f0-9]{64}$`)

func Download(opts DownloadOptions) error {
	if opts.Repo == "" {
		return fmt.Errorf("repo is required")
	}
	if opts.OutDir == "" {
		return fmt.Errorf("out dir is required")
	}
	if opts.Stdout == nil {
		opts.Stdout = io.Discard
	}
	if opts.Stderr == nil {
		opts.Stderr = io.Discard
	}

	manifest, err := PinnedManifest(opts.Repo)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(opts.OutDir, 0o755); err != nil {
		return fmt.Errorf("create out dir: %w", err)
	}

	lockPath := filepath.Join(opts.OutDir, "download-manifest.lock.json")
	lock := readLockManifest(lockPath)
	if lock.Files == nil {
		lock.Files = make(map[string]lockRecord)
	}
	lock.Repo = opts.Repo
	lock.Generated = time.Now().UTC().Format(time.RFC3339)

	client := &http.Client{Timeout: 0}

	for _, f := range manifest.Files {
		expected := strings.ToLower(f.SHA256)
		if expected == "" {
			if lr, ok := lock.Files[f.Filename]; ok && lr.Revision == f.Revision && isSHA256Hex(lr.SHA256) {
				expected = strings.ToLower(lr.SHA256)
			} else {
				expected, err = resolveChecksumFromMetadata(client, manifest.Repo, f, opts.HFToken)
				if err != nil {
					return err
				}
			}
		}

		localPath := filepath.Join(opts.OutDir, filepath.FromSlash(f.Filename))
		if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
			return fmt.Errorf("create local subdir: %w", err)
		}

		if ok, err := existingMatches(localPath, expected); err != nil {
			return err
		} else if ok {
			fmt.Fprintf(opts.Stdout, "skip %s (checksum match)\n", f.Filename)
			lock.Files[f.Filename] = lockRecord{Revision: f.Revision, SHA256: expected}
			continue
		}

		fmt.Fprintf(opts.Stdout, "download %s@%s -> %s\n", f.Filename, f.Revision, localPath)
		actual, err := downloadWithProgress(client, manifest.Repo, f, opts.HFToken, localPath, opts.Stdout)
		if err != nil {
			return err
		}
		if actual != expected {
			return fmt.Errorf("checksum mismatch for %s: expected %s got %s", f.Filename, expected, actual)
		}
		fmt.Fprintf(opts.Stdout, "verified %s (sha256=%s)\n", f.Filename, actual)
		lock.Files[f.Filename] = lockRecord{Revision: f.Revision, SHA256: expected}
	}

	if err := writeLockManifest(lockPath, lock); err != nil {
		return err
	}
	fmt.Fprintf(opts.Stdout, "wrote lock manifest: %s\n", lockPath)
	return nil
}

func existingMatches(path, expected string) (bool, error) {
	fi, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, fmt.Errorf("stat existing file: %w", err)
	}
	if fi.IsDir() {
		return false, fmt.Errorf("expected file at %s, found directory", path)
	}
	actual, err := fileSHA256(path)
	if err != nil {
		return false, err
	}
	return actual == expected, nil
}

func downloadWithProgress(client *http.Client, repo string, file ModelFile, token, outPath string, stdout io.Writer) (string, error) {
	url := resolveURL(repo, file)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	setAuth(req, token)

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("download request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return "", &ErrAccessDenied{
			Repo: repo,
			Msg:  fmt.Sprintf("access denied for %s; provide HF_TOKEN or --hf-token", repo),
		}
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return "", fmt.Errorf("download failed for %s: %s", file.Filename, resp.Status)
	}

	tmp := outPath + ".tmp"
	fh, err := os.Create(tmp)
	if err != nil {
		return "", fmt.Errorf("create temp file: %w", err)
	}

	h := sha256.New()
	mw := io.MultiWriter(fh, h)

	var written int64
	buf := make([]byte, 64*1024)
	total := resp.ContentLength
	lastPrint := time.Now()
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			wn, writeErr := mw.Write(buf[:n])
			if writeErr != nil {
				_ = fh.Close()
				_ = os.Remove(tmp)
				return "", fmt.Errorf("write temp file: %w", writeErr)
			}
			written += int64(wn)
			if time.Since(lastPrint) > 700*time.Millisecond {
				if total > 0 {
					pct := float64(written) * 100 / float64(total)
					fmt.Fprintf(stdout, "  progress: %.1f%% (%d/%d bytes)\n", pct, written, total)
				} else {
					fmt.Fprintf(stdout, "  progress: %d bytes\n", written)
				}
				lastPrint = time.Now()
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			_ = fh.Close()
			_ = os.Remove(tmp)
			return "", fmt.Errorf("download read failed: %w", readErr)
		}
	}

	if err := fh.Close(); err != nil {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("close temp file: %w", err)
	}
	if err := os.Rename(tmp, outPath); err != nil {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("move temp file into place: %w", err)
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}

func resolveChecksumFromMetadata(client *http.Client, repo string, f ModelFile, token string) (string, error) {
	req, err := http.NewRequest(http.MethodHead, resolveURL(repo, f), nil)
	if err != nil {
		return "", fmt.Errorf("build metadata request: %w", err)
	}
	setAuth(req, token)

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("metadata request failed for %s: %w", f.Filename, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return "", &ErrAccessDenied{
			Repo: repo,
			Msg:  fmt.Sprintf("access denied for %s; provide HF_TOKEN or --hf-token", repo),
		}
	}
	if resp.StatusCode < 200 || resp.StatusCode > 399 {
		return "", fmt.Errorf("metadata request failed for %s: %s", f.Filename, resp.Status)
	}

	for _, key := range []string{"X-Linked-Etag", "X-Repo-Commit", "Etag"} {
		if v := normalizeETag(resp.Header.Get(key)); isSHA256Hex(v) {
			return strings.ToLower(v), nil
		}
	}

	return "", fmt.Errorf("unable to resolve sha256 metadata for %s; provide pinned checksum", f.Filename)
}

func resolveURL(repo string, file ModelFile) string {
	return fmt.Sprintf("https://huggingface.co/%s/resolve/%s/%s", repo, file.Revision, file.Filename)
}

func setAuth(req *http.Request, token string) {
	if token == "" {
		return
	}
	req.Header.Set("Authorization", "Bearer "+token)
}

func normalizeETag(v string) string {
	v = strings.TrimSpace(v)
	v = strings.Trim(v, "\"")
	v = strings.TrimPrefix(v, "W/")
	v = strings.Trim(v, "\"")
	return v
}

func isSHA256Hex(v string) bool {
	return shaHexPattern.MatchString(v)
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open file for checksum: %w", err)
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("read file for checksum: %w", err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func readLockManifest(path string) lockManifest {
	b, err := os.ReadFile(path)
	if err != nil {
		return lockManifest{}
	}
	var out lockManifest
	if err := json.Unmarshal(b, &out); err != nil {
		return lockManifest{}
	}
	if out.Files == nil {
		out.Files = map[string]lockRecord{}
	}
	return out
}

func writeLockManifest(path string, lock lockManifest) error {
	b, err := json.MarshalIndent(lock, "", "  ")
	if err != nil {
		return fmt.Errorf("encode lock manifest: %w", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("write lock manifest: %w", err)
	}
	return nil
}
