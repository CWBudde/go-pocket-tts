package main

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveServeDefaultVoice(t *testing.T) {
	errFetch := errors.New("fetch must not be called")
	noFetch := func(string) (string, error) { return "", errFetch }

	for _, ref := range []string{"", "juergen", "voices/german/anna.safetensors"} {
		got, err := resolveServeDefaultVoice(ref, noFetch)
		if err != nil || got != ref {
			t.Errorf("resolveServeDefaultVoice(%q) = %q, %v; want it unchanged", ref, got, err)
		}
	}

	for _, ref := range []string{
		"prompt.wav", "/x/Prompt.WAV",
		"hf://org/repo/prompt.wav@rev", "https://example.com/prompt.wav?download=1",
	} {
		_, err := resolveServeDefaultVoice(ref, noFetch)
		if err == nil || !strings.Contains(err.Error(), "WAV") {
			t.Errorf("resolveServeDefaultVoice(%q) error = %v; want a WAV error", ref, err)
		}
	}

	var fetched string

	got, err := resolveServeDefaultVoice("hf://org/repo/anna.safetensors@rev", func(ref string) (string, error) {
		fetched = ref

		return "/cache/anna.safetensors", nil
	})
	if err != nil || got != "/cache/anna.safetensors" || fetched != "hf://org/repo/anna.safetensors@rev" {
		t.Errorf("remote voice = %q, %v (fetched %q); want the fetched cache path", got, err, fetched)
	}
}

// URL voices land in <user cache dir>/pockettts/voices.
func TestUserCacheVoiceFetcher_UsesUserCacheDir(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, ".cache"))
	t.Setenv("LocalAppData", filepath.Join(home, "AppData", "Local"))

	cacheDir, err := os.UserCacheDir()
	if err != nil {
		t.Skipf("no user cache dir: %v", err)
	}

	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("voice-bytes"))
	}))
	defer srv.Close()

	got, err := userCacheVoiceFetcher(srv.Client(), io.Discard)(srv.URL + "/anna.safetensors")
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}

	if want := filepath.Join(cacheDir, "pockettts", "voices"); filepath.Dir(got) != want {
		t.Errorf("cached voice = %q; want it in %q", got, want)
	}
}
