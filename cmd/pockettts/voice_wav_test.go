package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	nativemodel "github.com/cwbudde/go-pocket-tts/internal/native"
	"github.com/cwbudde/go-pocket-tts/internal/onnx"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
)

// fakeWAVEncoder swaps buildVoiceEncoder for a fake that returns frames
// conditioning frames, and records the weights it was asked for.
func fakeWAVEncoder(t *testing.T, frames int, runErr error) (*fakeVoiceEncoder, *string) {
	t.Helper()

	// Cloned voices go to os.TempDir; keep them out of the system one.
	t.Setenv("TMPDIR", t.TempDir())

	orig := buildVoiceEncoder

	t.Cleanup(func() { buildVoiceEncoder = orig })

	enc := &fakeVoiceEncoder{output: make([]float32, frames*onnx.VoiceEmbeddingDim), runErr: runErr}
	for i := range enc.output {
		enc.output[i] = float32(i%7) / 7
	}

	var weights string

	buildVoiceEncoder = func(cfg config.Config, modelWeightsPath string) (voiceEncoder, error) {
		weights = modelWeightsPath
		encoderBackend = cfg.TTS.Backend

		return enc, nil
	}

	return enc, &weights
}

// encoderBackend is the backend of the config fakeWAVEncoder's builder saw.
var encoderBackend string

func wavVoiceConfig() config.Config {
	cfg := config.DefaultConfig()
	cfg.TTS.Backend = config.BackendNative
	cfg.Paths.ModelPath = "models/gated/german/model.safetensors"
	cfg.Paths.VoiceManifest = filepath.Join("does", "not", "exist.json")

	return cfg
}

func TestEncodeWAVVoice_WritesTempVoiceFile(t *testing.T) {
	enc, weights := fakeWAVEncoder(t, 2, nil)

	path, remove, err := encodeWAVVoice(wavVoiceConfig(), "speaker.wav")
	if err != nil {
		t.Fatalf("encodeWAVVoice: %v", err)
	}

	if enc.input != "speaker.wav" || !enc.closed {
		t.Errorf("encoder input %q, closed %v; want speaker.wav, closed", enc.input, enc.closed)
	}

	// The prompt is encoded with the checkpoint synthesis loads.
	if *weights != "models/gated/german/model.safetensors" {
		t.Errorf("encoder weights %q, want the --paths-model-path checkpoint", *weights)
	}

	err = tts.CheckVoiceFile(path)
	if err != nil {
		t.Fatalf("CheckVoiceFile(%s): %v", path, err)
	}

	_, shape, err := safetensors.LoadVoiceEmbedding(path)
	if err != nil || len(shape) != 3 || shape[1] != 2 || shape[2] != onnx.VoiceEmbeddingDim {
		t.Errorf("voice embedding shape %v, %v; want [1 2 %d]", shape, err, onnx.VoiceEmbeddingDim)
	}

	remove()

	_, err = os.Stat(filepath.Dir(path))
	if !os.IsNotExist(err) {
		t.Errorf("temp voice dir %s still there after remove: %v", filepath.Dir(path), err)
	}
}

func TestEncodeWAVVoice_Errors(t *testing.T) {
	encErr := errors.New("the checkpoint's Mimi encoder weights are zeroed")
	fakeWAVEncoder(t, 0, encErr)

	for _, ref := range []string{"hf://org/repo/prompt.wav@rev", "https://example.com/prompt.wav"} {
		_, _, err := encodeWAVVoice(wavVoiceConfig(), ref)
		if err == nil || !strings.Contains(err.Error(), "local") {
			t.Errorf("encodeWAVVoice(%q) error = %v; want a local-file error", ref, err)
		}
	}

	_, _, err := encodeWAVVoice(wavVoiceConfig(), "speaker.wav")
	if !errors.Is(err, encErr) || !strings.Contains(err.Error(), "speaker.wav") {
		t.Errorf("encoder failure: err = %v; want it wrapped with the prompt name", err)
	}
}

func TestResolveSynthVoice_WAV(t *testing.T) {
	enc, _ := fakeWAVEncoder(t, 1, nil)
	cfg := wavVoiceConfig()

	// No slash: still a WAV prompt, not a manifest ID (the manifest is missing).
	path, remove, err := resolveSynthVoice(cfg, config.BackendNative, "speaker.wav", noVoiceFetch(t))
	if err != nil {
		t.Fatalf("native .wav: %v", err)
	}

	if enc.input != "speaker.wav" || tts.CheckVoiceFile(path) != nil {
		t.Errorf("native .wav: encoded %q into %q; want speaker.wav as a loadable voice", enc.input, path)
	}

	remove()

	_, err = os.Stat(path)
	if !os.IsNotExist(err) {
		t.Errorf("voice %s still there after remove: %v", path, err)
	}

	enc.input = ""

	_, _, err = resolveSynthVoice(cfg, config.BackendNativeONNX, "speaker.wav", noVoiceFetch(t))
	if err == nil || !strings.Contains(err.Error(), "export-voice") || enc.input != "" {
		t.Errorf("native-onnx .wav: err = %v, encoded %q; want an export-voice hint and no encoding", err, enc.input)
	}

	// Other voices resolve as before.
	path, remove, err = resolveSynthVoice(cfg, config.BackendNative, "voices/anna.safetensors", noVoiceFetch(t))
	if err != nil || path != "voices/anna.safetensors" || enc.input != "" {
		t.Errorf("safetensors voice = %q, %v (encoded %q); want it unchanged", path, err, enc.input)
	}

	remove()
}

// TestResolveSynthVoice_RealCheckpoints clones the parity prompt with the
// gated german checkpoint (27 frames, like upstream) and expects the ungated
// one to be refused with the gated-weights hint.
func TestResolveSynthVoice_RealCheckpoints(t *testing.T) {
	gated := os.Getenv("POCKETTTS_GATED_MODELS")
	if gated == "" {
		gated = filepath.Join("..", "..", "models", "gated")
	}

	prompt := filepath.Join("..", "..", "internal", "native", "testdata", "python_parity", "voice_prompt.wav")
	cfg := wavVoiceConfig()

	t.Setenv("TMPDIR", t.TempDir())

	t.Run("gated", func(t *testing.T) {
		cfg.Paths.ModelPath = filepath.Join(gated, "german", "model.safetensors")

		_, err := os.Stat(cfg.Paths.ModelPath)
		if err != nil {
			t.Skipf("gated german checkpoint not found: %v", err)
		}

		path, remove, err := resolveSynthVoice(cfg, config.BackendNative, prompt, noVoiceFetch(t))
		if err != nil {
			t.Fatalf("resolveSynthVoice: %v", err)
		}
		defer remove()

		err = tts.CheckVoiceFile(path)
		if err != nil {
			t.Fatalf("CheckVoiceFile: %v", err)
		}

		_, shape, err := safetensors.LoadVoiceEmbedding(path)
		if err != nil || len(shape) != 3 || shape[1] != 27 {
			t.Errorf("cloned voice shape %v, %v; want [1 27 1024]", shape, err)
		}
	})

	t.Run("ungated", func(t *testing.T) {
		cfg.Paths.ModelPath = filepath.Join("..", "..", "models", "german", "model.safetensors")

		_, err := os.Stat(cfg.Paths.ModelPath)
		if err != nil {
			t.Skipf("ungated german checkpoint not found: %v", err)
		}

		_, _, err = resolveSynthVoice(cfg, config.BackendNative, prompt, noVoiceFetch(t))
		if !errors.Is(err, nativemodel.ErrMimiEncoderWeightsZeroed) {
			t.Errorf("ungated checkpoint: err = %v; want %v", err, nativemodel.ErrMimiEncoderWeightsZeroed)
		}
	})
}

// TestResolveSynthVoice_BackendOverride: `synth --backend native` with
// another backend in the config must still pick the native encoder.
func TestResolveSynthVoice_BackendOverride(t *testing.T) {
	fakeWAVEncoder(t, 1, nil)

	cfg := wavVoiceConfig()
	cfg.TTS.Backend = config.BackendNativeONNX

	_, remove, err := resolveSynthVoice(cfg, config.BackendNative, "speaker.wav", noVoiceFetch(t))
	if err != nil {
		t.Fatalf("resolveSynthVoice: %v", err)
	}

	remove()

	if encoderBackend != config.BackendNative {
		t.Errorf("encoder built for backend %q, want the selected %q", encoderBackend, config.BackendNative)
	}
}
