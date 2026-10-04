package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/model"
	"github.com/cwbudde/go-pocket-tts/internal/onnx"
)

// isWAVVoice reports whether a --voice / --default-voice is an audio prompt
// to clone rather than a voice ID or voice file.
func isWAVVoice(ref string) bool {
	return model.VoiceRefExt(ref) == ".wav"
}

// encodeWAVVoice clones a voice from a local WAV prompt like export-voice:
// it encodes the prompt with the Mimi encoder of the checkpoint synthesis
// loads (it must be the gated one) and writes the conditioning to a
// temporary voice file. remove deletes that file.
func encodeWAVVoice(cfg config.Config, wavPath string) (string, func(), error) {
	if model.IsRemoteVoice(wavPath) {
		return "", nil, fmt.Errorf("WAV voice %q: only local WAV files can be cloned; download it first", wavPath)
	}

	encoder, err := buildVoiceEncoder(cfg, resolveExportVoiceModelPath(cfg, ""))
	if err != nil {
		return "", nil, fmt.Errorf("WAV voice %q: %w", wavPath, err)
	}
	defer encoder.Close()

	embedding, err := encoder.EncodeVoice(wavPath)
	if err != nil {
		return "", nil, fmt.Errorf("WAV voice %q: %w", wavPath, err)
	}

	if len(embedding) == 0 || len(embedding)%onnx.VoiceEmbeddingDim != 0 {
		return "", nil, fmt.Errorf("WAV voice %q: encoded length %d is not a whole number of %d-wide frames",
			wavPath, len(embedding), onnx.VoiceEmbeddingDim)
	}

	dir, err := os.MkdirTemp("", "pockettts-voice-")
	if err != nil {
		return "", nil, fmt.Errorf("WAV voice %q: %w", wavPath, err)
	}

	remove := func() { _ = os.RemoveAll(dir) }
	path := filepath.Join(dir, "voice.safetensors")
	shape := []int64{1, int64(len(embedding) / onnx.VoiceEmbeddingDim), onnx.VoiceEmbeddingDim}

	err = writeVoiceSafetensors(path, embedding, shape)
	if err != nil {
		remove()

		return "", nil, fmt.Errorf("WAV voice %q: write voice file: %w", wavPath, err)
	}

	return path, remove, nil
}

// resolveSynthVoice resolves synth's --voice for the native backends. A WAV
// prompt is cloned into a temporary voice file (native only; native-onnx
// needs export-voice first); everything else goes to resolveNativeVoice.
// remove deletes a cloned voice and is a no-op otherwise.
func resolveSynthVoice(cfg config.Config, backend, voice string) (string, func(), error) {
	if !isWAVVoice(voice) {
		path, err := resolveNativeVoice(cfg, backend, voice)

		return path, func() {}, err
	}

	if backend != config.BackendNative {
		return "", nil, errors.New("--voice " + voice + ": WAV voices are cloned by the native backend only; " +
			"run 'pockettts export-voice' and pass the .safetensors instead")
	}

	return encodeWAVVoice(cfg, voice)
}
