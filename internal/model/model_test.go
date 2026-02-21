package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// ErrAccessDenied
// ---------------------------------------------------------------------------

func TestErrAccessDenied_WithMsg(t *testing.T) {
	err := &ErrAccessDenied{Repo: "org/repo", Msg: "custom error"}
	if err.Error() != "custom error" {
		t.Errorf("Error() = %q; want %q", err.Error(), "custom error")
	}
}

func TestErrAccessDenied_WithoutMsg(t *testing.T) {
	err := &ErrAccessDenied{Repo: "org/repo"}
	if !strings.Contains(err.Error(), "org/repo") {
		t.Errorf("Error() = %q; should mention repo", err.Error())
	}
}

// ---------------------------------------------------------------------------
// PinnedManifest
// ---------------------------------------------------------------------------

func TestPinnedManifest_KnownRepos(t *testing.T) {
	repos := []string{
		"kyutai/pocket-tts",
		"kyutai/pocket-tts-without-voice-cloning",
	}
	for _, repo := range repos {
		t.Run(repo, func(t *testing.T) {
			m, err := PinnedManifest(repo)
			if err != nil {
				t.Fatalf("PinnedManifest(%q) error = %v", repo, err)
			}
			if m.Repo != repo {
				t.Errorf("Repo = %q; want %q", m.Repo, repo)
			}
			if len(m.Files) == 0 {
				t.Error("Files is empty")
			}
			for _, f := range m.Files {
				if f.Filename == "" {
					t.Error("File has empty Filename")
				}
				if f.Revision == "" {
					t.Error("File has empty Revision")
				}
			}
		})
	}
}

func TestPinnedManifest_UnknownRepo(t *testing.T) {
	_, err := PinnedManifest("unknown/repo")
	if err == nil {
		t.Error("PinnedManifest(unknown) = nil; want error")
	}
}

func TestPinnedManifest_WithoutVoiceCloning_HasChecksums(t *testing.T) {
	m, err := PinnedManifest("kyutai/pocket-tts-without-voice-cloning")
	if err != nil {
		t.Fatalf("PinnedManifest error = %v", err)
	}
	for _, f := range m.Files {
		if f.SHA256 == "" {
			t.Errorf("file %q has empty SHA256; expected pinned checksum", f.Filename)
		}
		if !isSHA256Hex(f.SHA256) {
			t.Errorf("file %q SHA256 %q is not valid hex", f.Filename, f.SHA256)
		}
	}
}

// ---------------------------------------------------------------------------
// existingMatches
// ---------------------------------------------------------------------------

func TestExistingMatches_NoFile(t *testing.T) {
	ok, err := existingMatches("/nonexistent/path/file.bin", "abc")
	if err != nil {
		t.Fatalf("existingMatches(missing) error = %v", err)
	}
	if ok {
		t.Error("existingMatches(missing) = true; want false")
	}
}

func TestExistingMatches_Directory(t *testing.T) {
	dir := t.TempDir()
	_, err := existingMatches(dir, "abc")
	if err == nil {
		t.Error("existingMatches(directory) = nil; want error")
	}
}

func TestExistingMatches_ChecksumMismatch(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "f.bin")
	os.WriteFile(p, []byte("data"), 0o644)

	ok, err := existingMatches(p, strings.Repeat("a", 64))
	if err != nil {
		t.Fatalf("existingMatches error = %v", err)
	}
	if ok {
		t.Error("existingMatches(mismatch) = true; want false")
	}
}

func TestExistingMatches_ChecksumMatch(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "f.bin")
	content := []byte("hello world")
	os.WriteFile(p, content, 0o644)

	h := sha256.Sum256(content)
	checksum := hex.EncodeToString(h[:])

	ok, err := existingMatches(p, checksum)
	if err != nil {
		t.Fatalf("existingMatches error = %v", err)
	}
	if !ok {
		t.Error("existingMatches(match) = false; want true")
	}
}

// ---------------------------------------------------------------------------
// fileSHA256
// ---------------------------------------------------------------------------

func TestFileSHA256_KnownContent(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "f.bin")
	content := []byte("test content")
	os.WriteFile(p, content, 0o644)

	h := sha256.Sum256(content)
	want := hex.EncodeToString(h[:])

	got, err := fileSHA256(p)
	if err != nil {
		t.Fatalf("fileSHA256 error = %v", err)
	}
	if got != want {
		t.Errorf("fileSHA256 = %q; want %q", got, want)
	}
}

func TestFileSHA256_MissingFile(t *testing.T) {
	_, err := fileSHA256("/nonexistent/file.bin")
	if err == nil {
		t.Error("fileSHA256(missing) = nil; want error")
	}
}

func TestFileSHA256_EmptyFile(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "empty.bin")
	os.WriteFile(p, []byte{}, 0o644)

	h := sha256.Sum256([]byte{})
	want := hex.EncodeToString(h[:])

	got, err := fileSHA256(p)
	if err != nil {
		t.Fatalf("fileSHA256(empty) error = %v", err)
	}
	if got != want {
		t.Errorf("fileSHA256(empty) = %q; want %q", got, want)
	}
}

// ---------------------------------------------------------------------------
// readLockManifest / writeLockManifest
// ---------------------------------------------------------------------------

func TestReadLockManifest_MissingFile(t *testing.T) {
	// Missing file returns empty lockManifest without error.
	lock := readLockManifest("/nonexistent/lock.json")
	// Files may be nil on error path; caller is responsible for nil-checking.
	// Verify it does not panic.
	_ = lock.Repo
	_ = lock.Files
}

func TestReadLockManifest_InvalidJSON(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "lock.json")
	os.WriteFile(p, []byte("{bad"), 0o644)

	// Invalid JSON returns empty lockManifest without error.
	lock := readLockManifest(p)
	_ = lock.Repo
	_ = lock.Files
}

func TestReadLockManifest_ValidFile(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "lock.json")
	content := `{"repo":"org/repo","generated":"2026-01-01T00:00:00Z","files":{"a.bin":{"revision":"r1","sha256":"` + strings.Repeat("1", 64) + `"}}}`
	os.WriteFile(p, []byte(content), 0o644)

	lock := readLockManifest(p)
	if lock.Repo != "org/repo" {
		t.Errorf("Repo = %q; want org/repo", lock.Repo)
	}
	if lock.Files == nil {
		t.Fatal("Files is nil")
	}
	rec, ok := lock.Files["a.bin"]
	if !ok {
		t.Fatal("Files[a.bin] not found")
	}
	if rec.Revision != "r1" {
		t.Errorf("Revision = %q; want r1", rec.Revision)
	}
}

func TestWriteReadLockManifest_RoundTrip(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "lock.json")

	original := lockManifest{
		Repo:      "kyutai/pocket-tts",
		Generated: "2026-01-01T00:00:00Z",
		Files: map[string]lockRecord{
			"model.safetensors": {
				Revision: "abc123",
				SHA256:   strings.Repeat("a", 64),
			},
		},
	}

	if err := writeLockManifest(p, original); err != nil {
		t.Fatalf("writeLockManifest error = %v", err)
	}

	got := readLockManifest(p)
	if got.Repo != original.Repo {
		t.Errorf("Repo = %q; want %q", got.Repo, original.Repo)
	}
	if got.Generated != original.Generated {
		t.Errorf("Generated = %q; want %q", got.Generated, original.Generated)
	}
	rec, ok := got.Files["model.safetensors"]
	if !ok {
		t.Fatal("Files[model.safetensors] not found")
	}
	if rec.Revision != "abc123" {
		t.Errorf("Revision = %q; want abc123", rec.Revision)
	}
}

func TestWriteLockManifest_CreatesFile(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "subdir", "lock.json")

	// Parent dir does not exist — expect failure (writeLockManifest doesn't mkdir).
	err := writeLockManifest(p, lockManifest{Files: map[string]lockRecord{}})
	if err == nil {
		t.Error("writeLockManifest(missing parent) = nil; want error")
	}
}

// ---------------------------------------------------------------------------
// resolveURL
// ---------------------------------------------------------------------------

func TestResolveURL(t *testing.T) {
	f := ModelFile{Filename: "model.safetensors", Revision: "abc123"}
	got := resolveURL("org/repo", f)
	want := "https://huggingface.co/org/repo/resolve/abc123/model.safetensors"
	if got != want {
		t.Errorf("resolveURL = %q; want %q", got, want)
	}
}

// ---------------------------------------------------------------------------
// setAuth
// ---------------------------------------------------------------------------

func TestSetAuth_WithToken(t *testing.T) {
	req, _ := http.NewRequest(http.MethodGet, "http://example.com", nil)
	setAuth(req, "mytoken")
	got := req.Header.Get("Authorization")
	if got != "Bearer mytoken" {
		t.Errorf("Authorization = %q; want %q", got, "Bearer mytoken")
	}
}

func TestSetAuth_EmptyToken(t *testing.T) {
	req, _ := http.NewRequest(http.MethodGet, "http://example.com", nil)
	setAuth(req, "")
	got := req.Header.Get("Authorization")
	if got != "" {
		t.Errorf("Authorization = %q; want empty for empty token", got)
	}
}

// ---------------------------------------------------------------------------
// normalizeETag / isSHA256Hex
// ---------------------------------------------------------------------------

func TestNormalizeETag_Variants(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{`"abc"`, "abc"},
		{`W/"abc"`, "abc"},
		{`  abc  `, "abc"},
		{`W/"` + strings.Repeat("a", 64) + `"`, strings.Repeat("a", 64)},
		{"", ""},
	}
	for _, tt := range tests {
		got := normalizeETag(tt.input)
		if got != tt.want {
			t.Errorf("normalizeETag(%q) = %q; want %q", tt.input, got, tt.want)
		}
	}
}

func TestIsSHA256Hex(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{strings.Repeat("a", 64), true},
		{strings.Repeat("A", 64), true},
		{"58aa704a88faad35f22c34ea1cb55c4c5629de8b8e035c6e4936e2673dc07617", true},
		{strings.Repeat("a", 63), false},  // too short
		{strings.Repeat("a", 65), false},  // too long
		{"", false},
		{strings.Repeat("g", 64), false}, // invalid hex char
	}
	for _, tt := range tests {
		got := isSHA256Hex(tt.input)
		if got != tt.want {
			t.Errorf("isSHA256Hex(%q) = %v; want %v", tt.input, got, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// Download — validation path (no network)
// ---------------------------------------------------------------------------

func TestDownload_EmptyRepo(t *testing.T) {
	err := Download(DownloadOptions{OutDir: "/tmp"})
	if err == nil {
		t.Error("Download(empty repo) = nil; want error")
	}
}

func TestDownload_EmptyOutDir(t *testing.T) {
	err := Download(DownloadOptions{Repo: "kyutai/pocket-tts"})
	if err == nil {
		t.Error("Download(empty outDir) = nil; want error")
	}
}

func TestDownload_UnknownRepo(t *testing.T) {
	err := Download(DownloadOptions{Repo: "not/a/known/repo", OutDir: t.TempDir()})
	if err == nil {
		t.Error("Download(unknown repo) = nil; want error")
	}
}

// ---------------------------------------------------------------------------
// Download — HTTP interactions via httptest
// ---------------------------------------------------------------------------

// sha256hex returns the lowercase hex SHA256 of data.
func sha256hex(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

// buildTestManifest creates a Manifest pointing to a single file with a known
// SHA256 checksum and returns the manifest and file content.
func buildTestManifest(repo, filename string, content []byte) (Manifest, string) {
	checksum := sha256hex(content)
	return Manifest{
		Repo: repo,
		Files: []ModelFile{
			{Filename: filename, Revision: "rev1", SHA256: checksum},
		},
	}, checksum
}

func TestDownloadWithProgress_Success(t *testing.T) {
	content := []byte("fake model weights")
	expectedSum := sha256hex(content)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write(content)
	}))
	defer srv.Close()

	tmp := t.TempDir()
	outPath := filepath.Join(tmp, "model.bin")
	file := ModelFile{Filename: "model.bin", Revision: "rev1"}

	got, err := downloadWithProgress(newHFClient(srv.URL), "org/repo", file, "", outPath, &strings.Builder{})
	if err != nil {
		t.Fatalf("downloadWithProgress error = %v", err)
	}
	if got != expectedSum {
		t.Errorf("checksum = %q; want %q", got, expectedSum)
	}

	data, readErr := os.ReadFile(outPath)
	if readErr != nil {
		t.Fatalf("ReadFile: %v", readErr)
	}
	if string(data) != string(content) {
		t.Errorf("file content = %q; want %q", data, content)
	}
}

func TestDownloadWithProgress_AccessDenied(t *testing.T) {
	for _, code := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		t.Run(fmt.Sprintf("HTTP%d", code), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(code)
			}))
			defer srv.Close()

			_, err := downloadWithProgress(newHFClient(srv.URL), "org/repo",
				ModelFile{Filename: "f.bin", Revision: "r1"}, "", filepath.Join(t.TempDir(), "f.bin"), &strings.Builder{})
			if err == nil {
				t.Errorf("HTTP %d should return error", code)
			}
			var denied *ErrAccessDenied
			if !isAccessDenied(err, &denied) {
				t.Errorf("expected ErrAccessDenied, got %T: %v", err, err)
			}
		})
	}
}

func TestDownloadWithProgress_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	_, err := downloadWithProgress(newHFClient(srv.URL), "org/repo",
		ModelFile{Filename: "f.bin", Revision: "r1"}, "", filepath.Join(t.TempDir(), "f.bin"), &strings.Builder{})
	if err == nil {
		t.Error("HTTP 500 should return error")
	}
}

// ---------------------------------------------------------------------------
// resolveChecksumFromMetadata — via httptest
// ---------------------------------------------------------------------------

func TestResolveChecksumFromMetadata_LinkedEtag(t *testing.T) {
	checksum := strings.Repeat("a", 64)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Linked-Etag", `"`+checksum+`"`)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	got, err := resolveChecksumFromMetadata(newHFClient(srv.URL), "org/repo",
		ModelFile{Filename: "f.bin", Revision: "r1"}, "")
	if err != nil {
		t.Fatalf("resolveChecksumFromMetadata error = %v", err)
	}
	if got != checksum {
		t.Errorf("checksum = %q; want %q", got, checksum)
	}
}

func TestResolveChecksumFromMetadata_EtagFallback(t *testing.T) {
	checksum := strings.Repeat("b", 64)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Etag", `"`+checksum+`"`)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	got, err := resolveChecksumFromMetadata(newHFClient(srv.URL), "org/repo",
		ModelFile{Filename: "f.bin", Revision: "r1"}, "")
	if err != nil {
		t.Fatalf("resolveChecksumFromMetadata error = %v", err)
	}
	if got != checksum {
		t.Errorf("checksum = %q; want %q", got, checksum)
	}
}

func TestResolveChecksumFromMetadata_NoUsableHeader(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	_, err := resolveChecksumFromMetadata(newHFClient(srv.URL), "org/repo",
		ModelFile{Filename: "f.bin", Revision: "r1"}, "")
	if err == nil {
		t.Error("no usable header should return error")
	}
}

func TestResolveChecksumFromMetadata_AccessDenied(t *testing.T) {
	for _, code := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		t.Run(fmt.Sprintf("HTTP%d", code), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(code)
			}))
			defer srv.Close()

			_, err := resolveChecksumFromMetadata(newHFClient(srv.URL), "org/repo",
				ModelFile{Filename: "f.bin", Revision: "r1"}, "")
			var denied *ErrAccessDenied
			if err == nil || !isAccessDenied(err, &denied) {
				t.Errorf("expected ErrAccessDenied for HTTP %d, got %v", code, err)
			}
		})
	}
}

func TestResolveChecksumFromMetadata_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	_, err := resolveChecksumFromMetadata(newHFClient(srv.URL), "org/repo",
		ModelFile{Filename: "f.bin", Revision: "r1"}, "")
	if err == nil {
		t.Error("HTTP 500 should return error")
	}
}

func TestResolveChecksumFromMetadata_WithToken(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("X-Linked-Etag", strings.Repeat("c", 64))
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	_, _ = resolveChecksumFromMetadata(newHFClient(srv.URL), "org/repo",
		ModelFile{Filename: "f.bin", Revision: "r1"}, "my-token")

	if gotAuth != "Bearer my-token" {
		t.Errorf("Authorization = %q; want %q", gotAuth, "Bearer my-token")
	}
}

// ---------------------------------------------------------------------------
// ExportONNX — validation-only paths
// ---------------------------------------------------------------------------

func TestExportONNX_EmptyModelsDir(t *testing.T) {
	err := ExportONNX(ExportOptions{OutDir: t.TempDir()})
	if err == nil {
		t.Error("ExportONNX(empty ModelsDir) = nil; want error")
	}
}

func TestExportONNX_EmptyOutDir(t *testing.T) {
	err := ExportONNX(ExportOptions{ModelsDir: t.TempDir()})
	if err == nil {
		t.Error("ExportONNX(empty OutDir) = nil; want error")
	}
}

func TestExportONNX_PythonBinNotFound(t *testing.T) {
	err := ExportONNX(ExportOptions{
		ModelsDir: t.TempDir(),
		OutDir:    t.TempDir(),
		PythonBin: "/nonexistent/python999",
	})
	if err == nil {
		t.Error("ExportONNX(bad python) = nil; want error")
	}
}

// ---------------------------------------------------------------------------
// validateExportTooling
// ---------------------------------------------------------------------------

func TestValidateExportTooling_MissingBin(t *testing.T) {
	err := validateExportTooling("/definitely/not/real/python")
	if err == nil {
		t.Error("validateExportTooling(missing) = nil; want error")
	}
}

// ---------------------------------------------------------------------------
// resolveScriptPath
// ---------------------------------------------------------------------------

func TestResolveScriptPath_EmptyPath(t *testing.T) {
	_, err := resolveScriptPath("")
	if err == nil {
		t.Error("resolveScriptPath(\"\") = nil; want error")
	}
}

func TestResolveScriptPath_NonExistent(t *testing.T) {
	_, err := resolveScriptPath("definitely_not_a_real_script.py")
	if err == nil {
		t.Error("resolveScriptPath(nonexistent) = nil; want error")
	}
}

func TestResolveScriptPath_ExistsInCwd(t *testing.T) {
	// Create a script file in the current working directory.
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	scriptName := "test_resolve_script_tmp.py"
	scriptPath := filepath.Join(cwd, scriptName)
	if err := os.WriteFile(scriptPath, []byte("# test"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	t.Cleanup(func() { os.Remove(scriptPath) })

	got, err := resolveScriptPath(scriptName)
	if err != nil {
		t.Fatalf("resolveScriptPath error = %v", err)
	}
	if got != scriptPath {
		t.Errorf("resolveScriptPath = %q; want %q", got, scriptPath)
	}
}

// ---------------------------------------------------------------------------
// detectPocketTTSPython
// ---------------------------------------------------------------------------

func TestDetectPocketTTSPython_NoBinary(t *testing.T) {
	// When pocket-tts is not on PATH, must return "python3".
	// We can't control PATH easily, so just verify it returns a non-empty string.
	result := detectPocketTTSPython()
	if result == "" {
		t.Error("detectPocketTTSPython() returned empty string")
	}
}

// ---------------------------------------------------------------------------
// Helpers used in tests
// ---------------------------------------------------------------------------

// hfTransport is a test RoundTripper that rewrites huggingface.co requests
// to a local test server, enabling tests of the production HTTP code paths.
type hfTransport struct {
	target string // e.g. "http://127.0.0.1:PORT"
}

func (t *hfTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.URL.Scheme = "http"
	clone.URL.Host = strings.TrimPrefix(t.target, "http://")
	return http.DefaultTransport.RoundTrip(clone)
}

// newHFClient returns an *http.Client whose transport redirects
// all requests (including those to huggingface.co) to the given server.
func newHFClient(serverURL string) *http.Client {
	return &http.Client{Transport: &hfTransport{target: serverURL}}
}

// isAccessDenied checks whether err message contains "access denied".
func isAccessDenied(err error, target **ErrAccessDenied) bool {
	if err == nil {
		return false
	}
	if strings.Contains(err.Error(), "access denied") {
		e := &ErrAccessDenied{}
		*target = e
		return true
	}
	return false
}

// writeLockManifest_valid serialises to JSON and verifies round-trip.
func TestWriteLockManifest_ValidContent(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "lock.json")

	lock := lockManifest{
		Repo:      "test/repo",
		Generated: "2026-01-01T00:00:00Z",
		Files: map[string]lockRecord{
			"a.bin": {Revision: "rev1", SHA256: strings.Repeat("1", 64)},
		},
	}
	if err := writeLockManifest(p, lock); err != nil {
		t.Fatalf("writeLockManifest error = %v", err)
	}

	raw, _ := os.ReadFile(p)
	var got lockManifest
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if got.Repo != lock.Repo {
		t.Errorf("Repo = %q; want %q", got.Repo, lock.Repo)
	}
	if got.Files["a.bin"].Revision != "rev1" {
		t.Errorf("Revision = %q; want rev1", got.Files["a.bin"].Revision)
	}
}
