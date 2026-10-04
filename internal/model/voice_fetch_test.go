package model

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
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

func TestFetchVoice_ConcurrentMissesBothSucceed(t *testing.T) {
	payload := []byte("voice-bytes")

	// Hold both requests until each has arrived, so the two downloads
	// overlap; a shared temp file makes one rename fail.
	var arrived sync.WaitGroup

	arrived.Add(2)

	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		arrived.Done()
		arrived.Wait()

		_, _ = w.Write(payload)
	}))
	defer srv.Close()

	opts := FetchVoiceOptions{Client: srv.Client(), CacheDir: t.TempDir(), Stdout: io.Discard}
	ref := srv.URL + "/anna.safetensors"

	var (
		wg   sync.WaitGroup
		errs [2]error
	)

	for i := range errs {
		wg.Go(func() { _, errs[i] = FetchVoice(ref, opts) })
	}

	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("FetchVoice #%d: %v", i, err)
		}
	}

	path, err := FetchVoice(ref, opts)
	if err != nil {
		t.Fatal(err)
	}

	got, err := os.ReadFile(path)
	if err != nil || string(got) != string(payload) {
		t.Errorf("cached file = %q, %v; want %q", got, err, payload)
	}

	entries, _ := os.ReadDir(opts.CacheDir)
	if len(entries) != 1 {
		t.Errorf("cache dir has %d entries; want only the voice (no temp files left)", len(entries))
	}
}

func TestFetchVoice_RejectsRedirectToNonHTTPS(t *testing.T) {
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("voice-bytes"))
	}))
	defer plain.Close()

	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, plain.URL+"/anna.safetensors", http.StatusFound)
	}))
	defer srv.Close()

	cacheDir := t.TempDir()

	_, err := FetchVoice(srv.URL+"/anna.safetensors", FetchVoiceOptions{
		Client: srv.Client(), CacheDir: cacheDir, Stdout: io.Discard,
	})
	if err == nil || !strings.Contains(err.Error(), "https") {
		t.Fatalf("FetchVoice error = %v; want a redirect-to-http error", err)
	}

	entries, _ := os.ReadDir(cacheDir)
	if len(entries) != 0 {
		t.Errorf("cache dir has %d entries after a rejected redirect; want none", len(entries))
	}
}

func TestFetchVoice_AccessDeniedHint(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "no", http.StatusForbidden)
	}))
	defer srv.Close()

	_, err := FetchVoice(srv.URL+"/anna.safetensors", FetchVoiceOptions{
		Client: srv.Client(), CacheDir: t.TempDir(), Stdout: io.Discard,
	})

	var denied *AccessDeniedError
	if !errors.As(err, &denied) {
		t.Fatalf("FetchVoice error = %v; want an AccessDeniedError", err)
	}

	if strings.Contains(err.Error(), "HF_TOKEN") {
		t.Errorf("https:// voice error %q suggests HF_TOKEN, which is never sent for https:// voices", err)
	}
}

func TestVoiceRefExt(t *testing.T) {
	for ref, want := range map[string]string{
		"voices/anna.safetensors":                   ".safetensors",
		"prompt.WAV":                                ".wav",
		"anna":                                      "",
		"hf://org/repo/prompts/anna.wav@abc123":     ".wav",
		"hf://org/repo/anna.safetensors@abc123":     ".safetensors",
		"https://example.com/anna.wav?download=1":   ".wav",
		"https://example.com/anna.safetensors#frag": ".safetensors",
		"https://example.com/voices/":               "",
		// Upstream test_is_safetensors_source_handles_revisions_and_query_strings.
		"voice.safetensors": ".safetensors",
		"hf://owner/repo/voices/voice.safetensors@abcdef":  ".safetensors",
		"https://example.com/voice.safetensors?download=1": ".safetensors",
		"https://example.com/voice.wav?format=safetensors": ".wav",
	} {
		if got := VoiceRefExt(ref); got != want {
			t.Errorf("VoiceRefExt(%q) = %q; want %q", ref, got, want)
		}
	}
}

// roundTripFunc serves requests in-process, so hf:// references (whose host
// is fixed to huggingface.co) can be tested without the network.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// A WAV prompt keeps its .wav name in the cache, so the voice-prompt reader
// decodes it as WAV rather than as raw PCM; voice files still get .safetensors.
func TestFetchVoice_CachesWAVPromptAsWAV(t *testing.T) {
	for ref, want := range map[string]struct {
		url, suffix, auth string
	}{
		"hf://kyutai/tts-voices/alba-mackenna/casual.wav@abc123": {
			url:    "https://huggingface.co/kyutai/tts-voices/resolve/abc123/alba-mackenna/casual.wav",
			suffix: "-casual.wav", auth: "Bearer hf_secret",
		},
		"https://example.com/prompts/Casual.WAV?download=1": {
			url:    "https://example.com/prompts/Casual.WAV?download=1",
			suffix: "-Casual.WAV",
		},
		"https://example.com/voices/anna": {
			url:    "https://example.com/voices/anna",
			suffix: "-anna.safetensors",
		},
	} {
		var gotURL, gotAuth string

		client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			gotURL, gotAuth = r.URL.String(), r.Header.Get("Authorization")

			return &http.Response{
				StatusCode: http.StatusOK, Status: "200 OK",
				Body: io.NopCloser(strings.NewReader("RIFF")), ContentLength: 4, Request: r,
			}, nil
		})}

		path, err := FetchVoice(ref, FetchVoiceOptions{
			Client: client, CacheDir: t.TempDir(), Token: "hf_secret", Stdout: io.Discard,
		})
		if err != nil {
			t.Fatalf("FetchVoice(%q): %v", ref, err)
		}

		if gotURL != want.url || gotAuth != want.auth {
			t.Errorf("FetchVoice(%q) requested %q (auth %q); want %q (auth %q)",
				ref, gotURL, gotAuth, want.url, want.auth)
		}

		if !strings.HasSuffix(path, want.suffix) {
			t.Errorf("FetchVoice(%q) cache path = %q; want suffix %q", ref, path, want.suffix)
		}
	}
}
