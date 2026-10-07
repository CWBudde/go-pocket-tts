package main

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/audio"
	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/onnx"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
)

// noVoiceFetch is the fetcher for voices that must not be downloaded.
func noVoiceFetch(t *testing.T) func(string) (string, error) {
	t.Helper()

	return func(ref string) (string, error) {
		t.Errorf("fetched %q; want no download", ref)

		return "", errors.New("fetch must not be called")
	}
}

// writeTestVoice writes a one-frame voice embedding that tts.CheckVoiceFile
// accepts.
func writeTestVoice(t *testing.T, path string) {
	t.Helper()

	err := writeVoiceSafetensors(path, make([]float32, onnx.VoiceEmbeddingDim), []int64{1, 1, onnx.VoiceEmbeddingDim})
	if err != nil {
		t.Fatalf("write test voice: %v", err)
	}
}

// TestResolveSynthVoice_Remote: synth downloads URL voices like serve
// --default-voice, before any model loads.
func TestResolveSynthVoice_Remote(t *testing.T) {
	const (
		voiceRef = "hf://kyutai/pocket-tts-without-voice-cloning/languages/german/voices/anna.safetensors@1e08e6a"
		wavRef   = "https://example.com/prompts/casual.wav"
	)

	cfg := wavVoiceConfig()

	// A remote voice file is downloaded and used as is, on both native
	// backends.
	for _, backend := range []string{config.BackendNative, config.BackendNativeONNX} {
		cached := filepath.Join(t.TempDir(), "0123-anna.safetensors")
		writeTestVoice(t, cached)

		var fetched []string

		path, remove, err := resolveSynthVoice(cfg, backend, voiceRef, func(ref string) (string, error) {
			fetched = append(fetched, ref)

			return cached, nil
		})
		if err != nil || path != cached || len(fetched) != 1 || fetched[0] != voiceRef {
			t.Errorf("%s: remote voice = %q, %v (fetched %q); want the cached file", backend, path, err, fetched)
		}

		remove()

		_, err = os.Stat(cached)
		if err != nil {
			t.Errorf("%s: cached voice gone after remove: %v", backend, err)
		}
	}

	// A broken download fails here, before the model weights load.
	corrupt := filepath.Join(t.TempDir(), "0123-anna.safetensors")

	err := os.WriteFile(corrupt, []byte("<html>not a voice</html>"), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = resolveSynthVoice(cfg, config.BackendNative, voiceRef,
		func(string) (string, error) { return corrupt, nil })
	if err == nil || !strings.Contains(err.Error(), "--voice") || !strings.Contains(err.Error(), voiceRef) {
		t.Errorf("corrupt download: err = %v; want a --voice error naming %q", err, voiceRef)
	}

	// FetchVoice reuses a cached file without a request, so a broken one must
	// go, or every later run fails without trying the server again.
	_, err = os.Stat(corrupt)
	if !errors.Is(err, os.ErrNotExist) {
		t.Errorf("corrupt download still cached after the failed check: %v", err)
	}

	_, _, err = resolveSynthVoice(cfg, config.BackendNative, voiceRef,
		func(string) (string, error) { return "", errors.New("download failed: 404 Not Found") })
	if err == nil || !strings.Contains(err.Error(), "404") || !strings.Contains(err.Error(), "--voice") {
		t.Errorf("failed download: err = %v; want the --voice download error", err)
	}

	t.Run("WAV", func(t *testing.T) {
		enc, _ := fakeWAVEncoder(t, 2, nil)
		cached := filepath.Join(t.TempDir(), "0123-casual.wav")

		path, remove, err := resolveSynthVoice(cfg, config.BackendNative, wavRef, func(string) (string, error) {
			return cached, os.WriteFile(cached, []byte("RIFF"), 0o600)
		})
		if err != nil {
			t.Fatalf("remote WAV: %v", err)
		}

		if enc.input != cached || tts.CheckVoiceFile(path) != nil {
			t.Errorf("remote WAV: encoded %q into %q; want the downloaded %q as a loadable voice", enc.input, path, cached)
		}

		remove()

		_, err = os.Stat(path)
		if !errors.Is(err, os.ErrNotExist) {
			t.Errorf("cloned voice %s still there after remove: %v", path, err)
		}

		_, err = os.Stat(cached)
		if err != nil {
			t.Errorf("cached download gone after remove: %v", err)
		}
	})

	t.Run("WAV on native-onnx", func(t *testing.T) {
		_, _, err := resolveSynthVoice(cfg, config.BackendNativeONNX, wavRef, noVoiceFetch(t))
		if err == nil || !strings.Contains(err.Error(), "export-voice") {
			t.Errorf("err = %v; want the export-voice hint before any download", err)
		}
	})
}

// TestRunSynthCommand_RemoteVoiceToStdout: with --out -, the WAV goes to
// stdout, so the voice download reports on stderr.
func TestRunSynthCommand_RemoteVoiceToStdout(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, ".cache"))
	t.Setenv("LocalAppData", filepath.Join(home, "AppData", "Local"))

	voice := filepath.Join(t.TempDir(), "anna.safetensors")
	writeTestVoice(t, voice)

	voiceBytes, err := os.ReadFile(voice)
	if err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(voiceBytes)
	}))
	defer srv.Close()

	origClient, origSynth := synthVoiceClient, runNativeSynthesis

	t.Cleanup(func() { synthVoiceClient, runNativeSynthesis = origClient, origSynth })

	synthVoiceClient = srv.Client()

	wav, err := audio.EncodeWAV([]float32{0.1, 0.2})
	if err != nil {
		t.Fatal(err)
	}

	var synthVoice string

	runNativeSynthesis = func(_ context.Context, _ config.Config, _ []string, voicePath string) ([]byte, error) {
		synthVoice = voicePath

		return wav, nil
	}

	cfg := config.DefaultConfig()
	cfg.TTS.Backend = config.BackendNative

	var stdout, stderr bytes.Buffer

	ref := srv.URL + "/voices/anna.safetensors"

	err = runSynthCommand(context.Background(), cfg, synthRunOptions{Text: "hello", Out: "-", Voice: ref},
		nil, &stdout, &stderr)
	if err != nil {
		t.Fatalf("runSynthCommand: %v", err)
	}

	// Native output gets trailing silence, so compare the format, not the bytes.
	_, err = audio.DecodeWAV(stdout.Bytes())
	if !bytes.HasPrefix(stdout.Bytes(), []byte("RIFF")) || err != nil {
		t.Errorf("stdout is not just the WAV (%d bytes, decode: %v): %q", stdout.Len(), err, stdout.String()[:min(40, stdout.Len())])
	}

	if !strings.Contains(stderr.String(), "download voice "+ref) {
		t.Errorf("stderr = %q; want the voice download reported there", stderr.String())
	}

	if filepath.Base(filepath.Dir(synthVoice)) != "voices" || tts.CheckVoiceFile(synthVoice) != nil {
		t.Errorf("synthesized with voice %q; want the downloaded file in the voice cache", synthVoice)
	}
}
