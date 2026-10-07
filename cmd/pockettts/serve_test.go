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

func TestResolveVoiceRef(t *testing.T) {
	errFetch := errors.New("fetch must not be called")
	noFetch := func(string) (string, error) { return "", errFetch }
	errEncode := errors.New("encode must not be called")
	noEncode := func(string) (string, func(), error) { return "", nil, errEncode }

	for _, ref := range []string{"", "juergen", "voices/german/anna.safetensors"} {
		got, cleanup, err := resolveVoiceRef("--default-voice", ref, noFetch, noEncode)
		if err != nil || got != ref {
			t.Errorf("resolveVoiceRef(%q) = %q, %v; want it unchanged", ref, got, err)
		}

		cleanup()
	}

	// A local WAV prompt is encoded once at startup; its voice file goes
	// away when the server stops.
	for _, ref := range []string{"prompt.wav", "/x/Prompt.WAV"} {
		var encoded string

		removed := false

		got, cleanup, err := resolveVoiceRef("--default-voice", ref, noFetch, func(wav string) (string, func(), error) {
			encoded = wav

			return "/tmp/voice.safetensors", func() { removed = true }, nil
		})
		if err != nil || got != "/tmp/voice.safetensors" || encoded != ref {
			t.Errorf("resolveVoiceRef(%q) = %q, %v (encoded %q); want the encoded voice", ref, got, err, encoded)
		}

		cleanup()

		if !removed {
			t.Errorf("resolveVoiceRef(%q): cleanup did not remove the encoded voice", ref)
		}
	}

	_, _, err := resolveVoiceRef("--default-voice", "prompt.wav", noFetch, func(string) (string, func(), error) {
		return "", nil, errors.New("weights are zeroed")
	})
	if err == nil || !strings.Contains(err.Error(), "zeroed") {
		t.Errorf("encoder failure: err = %v; want it passed on", err)
	}

	var fetched string

	got, cleanup, err := resolveVoiceRef("--default-voice", "hf://org/repo/anna.safetensors@rev", func(ref string) (string, error) {
		fetched = ref

		return "/cache/anna.safetensors", nil
	}, noEncode)
	if err != nil || got != "/cache/anna.safetensors" || fetched != "hf://org/repo/anna.safetensors@rev" {
		t.Errorf("remote voice = %q, %v (fetched %q); want the fetched cache path", got, err, fetched)
	}

	cleanup()

	t.Run("remote WAV", testResolveVoiceRefRemoteWAV)
	t.Run("remote WAV failures", testResolveVoiceRefRemoteWAVFailures)
}

// A remote WAV prompt is downloaded into the voice cache, then cloned from
// the cached file like a local one (upstream get_state_for_audio_prompt runs
// download_if_necessary first). Stopping the server removes the cloned voice
// but keeps the download for the next start.
func testResolveVoiceRefRemoteWAV(t *testing.T) {
	for _, ref := range []string{
		"https://example.com/prompts/casual.wav?download=1",
		"hf://kyutai/tts-voices/alba-mackenna/casual.wav@abc123",
	} {
		t.Run(ref, func(t *testing.T) {
			cacheDir, tmpDir := t.TempDir(), t.TempDir()
			cached := filepath.Join(cacheDir, "0123-casual.wav")
			encodedPath := filepath.Join(tmpDir, "voice.safetensors")

			var fetched, encoded []string

			fetch := func(r string) (string, error) {
				fetched = append(fetched, r)

				return cached, os.WriteFile(cached, []byte("RIFF"), 0o600)
			}
			encode := func(wav string) (string, func(), error) {
				encoded = append(encoded, wav)

				err := os.WriteFile(encodedPath, []byte("voice"), 0o600)

				return encodedPath, func() { _ = os.Remove(encodedPath) }, err
			}

			got, cleanup, err := resolveVoiceRef("--default-voice", ref, fetch, encode)
			if err != nil {
				t.Fatalf("resolveVoiceRef: %v", err)
			}

			if len(fetched) != 1 || fetched[0] != ref {
				t.Errorf("fetched %q; want exactly [%q]", fetched, ref)
			}

			if len(encoded) != 1 || encoded[0] != cached {
				t.Errorf("encoded %q; want exactly the downloaded file [%q]", encoded, cached)
			}

			if got != encodedPath {
				t.Errorf("voice = %q; want the encoded voice %q", got, encodedPath)
			}

			cleanup()

			_, err = os.Stat(encodedPath)
			if !errors.Is(err, os.ErrNotExist) {
				t.Errorf("encoded voice still exists after cleanup (stat err %v)", err)
			}

			_, err = os.Stat(cached)
			if err != nil {
				t.Errorf("cached download missing after cleanup: %v", err)
			}
		})
	}
}

// A remote WAV that cannot be downloaded or encoded stops serve at startup,
// and no encoded voice is left behind.
func testResolveVoiceRefRemoteWAVFailures(t *testing.T) {
	const ref = "hf://kyutai/tts-voices/alba-mackenna/casual.wav@abc123"

	t.Run("fetch fails", func(t *testing.T) {
		encodeCalled := false

		got, cleanup, err := resolveVoiceRef("--default-voice", ref,
			func(string) (string, error) { return "", errors.New("download failed: 404 Not Found") },
			func(string) (string, func(), error) {
				encodeCalled = true

				return "", nil, errors.New("encode must not be called")
			})
		if err == nil || !strings.Contains(err.Error(), "404") || !strings.Contains(err.Error(), "--default-voice") {
			t.Errorf("err = %v; want the --default-voice download error", err)
		}

		if got != "" || cleanup != nil || encodeCalled {
			t.Errorf("got %q, cleanup %v, encode called %v; want nothing after a failed download",
				got, cleanup != nil, encodeCalled)
		}
	})

	t.Run("encode fails", func(t *testing.T) {
		var encoded string

		got, cleanup, err := resolveVoiceRef("--default-voice", ref,
			func(string) (string, error) { return "/cache/0123-casual.wav", nil },
			func(wav string) (string, func(), error) {
				// Like encodeWAVVoice: no voice file is left after a failure.
				encoded = wav

				return "", nil, errors.New("gated weights are zeroed")
			})
		// The error names the remote ref, not only the cache file.
		if err == nil || !strings.Contains(err.Error(), "zeroed") || !strings.Contains(err.Error(), ref) {
			t.Errorf("err = %v; want the encode error naming %q", err, ref)
		}

		if encoded != "/cache/0123-casual.wav" {
			t.Errorf("encoded %q; want the downloaded file", encoded)
		}

		if got != "" || cleanup != nil {
			t.Errorf("got %q, cleanup %v; want nothing after a failed encode", got, cleanup != nil)
		}
	})
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
