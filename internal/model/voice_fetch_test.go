package model

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestVoiceURL(t *testing.T) {
	for ref, want := range map[string]struct {
		url    string
		isHF   bool
		errSub string
	}{
		"hf://kyutai/pocket-tts-without-voice-cloning/languages/german/embeddings/anna.safetensors@abc123": {
			url:  "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/abc123/languages/german/embeddings/anna.safetensors",
			isHF: true,
		},
		"https://example.com/voices/anna.safetensors?download=1": {url: "https://example.com/voices/anna.safetensors?download=1"},
		"hf://kyutai/pocket-tts/embeddings/anna.safetensors":     {errSub: "no pinned revision"},
		"http://example.com/anna.safetensors":                    {errSub: "https"},
		"ftp://example.com/anna.safetensors":                     {errSub: "https"},
		"https:///anna.safetensors":                              {errSub: "host"},
	} {
		url, isHF, err := voiceURL(ref)

		if want.errSub != "" {
			if err == nil || !strings.Contains(err.Error(), want.errSub) {
				t.Errorf("voiceURL(%q) error = %v; want one containing %q", ref, err, want.errSub)
			}

			continue
		}

		if err != nil || url != want.url || isHF != want.isHF {
			t.Errorf("voiceURL(%q) = %q, %v, %v; want %q, %v", ref, url, isHF, err, want.url, want.isHF)
		}
	}
}

func TestIsRemoteVoice(t *testing.T) {
	for ref, want := range map[string]bool{
		"https://example.com/a.safetensors": true,
		"hf://org/repo/a.safetensors@rev":   true,
		"http://example.com/a.safetensors":  true, // remote, rejected later by FetchVoice
		"juergen":                           false,
		"voices/juergen.safetensors":        false,
	} {
		if got := IsRemoteVoice(ref); got != want {
			t.Errorf("IsRemoteVoice(%q) = %v; want %v", ref, got, want)
		}
	}
}

func TestFetchVoice_DownloadsOnceThenUsesCache(t *testing.T) {
	payload := []byte("voice-bytes")

	var hits atomic.Int32

	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)

		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("https:// voice request sent Authorization %q; the token is for hf:// only", got)
		}

		_, _ = w.Write(payload)
	}))
	defer srv.Close()

	opts := FetchVoiceOptions{
		Client:   srv.Client(),
		CacheDir: filepath.Join(t.TempDir(), "voices"),
		Token:    "hf_secret",
		Stdout:   io.Discard,
	}
	ref := srv.URL + "/voices/anna.safetensors"

	first, err := FetchVoice(ref, opts)
	if err != nil {
		t.Fatalf("FetchVoice: %v", err)
	}

	if filepath.Dir(first) != opts.CacheDir || !strings.HasSuffix(first, "-anna.safetensors") {
		t.Errorf("cache path = %q; want <cache>/<hash>-anna.safetensors", first)
	}

	got, err := os.ReadFile(first)
	if err != nil || string(got) != string(payload) {
		t.Fatalf("cached file = %q, %v; want %q", got, err, payload)
	}

	second, err := FetchVoice(ref, opts)
	if err != nil || second != first {
		t.Fatalf("second FetchVoice = %q, %v; want %q", second, err, first)
	}

	if n := hits.Load(); n != 1 {
		t.Errorf("server hits = %d; want 1 (second call must use the cache)", n)
	}

	// Another URL with the same base name gets its own cache file.
	other, err := FetchVoice(srv.URL+"/other/anna.safetensors", opts)
	if err != nil || other == first {
		t.Errorf("other URL cache path = %q, %v; want a path other than %q", other, err, first)
	}
}

func TestFetchVoice_HTTPErrorLeavesNoFile(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusNotFound)
	}))
	defer srv.Close()

	cacheDir := t.TempDir()

	_, err := FetchVoice(srv.URL+"/anna.safetensors", FetchVoiceOptions{
		Client: srv.Client(), CacheDir: cacheDir, Stdout: io.Discard,
	})
	if err == nil || !strings.Contains(err.Error(), "404") {
		t.Fatalf("FetchVoice error = %v; want a 404 error", err)
	}

	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		t.Fatal(err)
	}

	if len(entries) != 0 {
		t.Errorf("cache dir has %d entries after a failed download; want none", len(entries))
	}
}
