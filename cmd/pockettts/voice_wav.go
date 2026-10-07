package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/model"
	"github.com/cwbudde/go-pocket-tts/internal/onnx"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
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
// needs export-voice first), and an https:// or hf:// voice or prompt is
// downloaded by fetch first, like serve --default-voice; everything else goes
// to resolveNativeVoice. remove deletes a cloned voice and is a no-op
// otherwise.
func resolveSynthVoice(
	cfg config.Config,
	backend, voice string,
	fetch func(string) (string, error),
) (string, func(), error) {
	if isWAVVoice(voice) && backend != config.BackendNative {
		return "", nil, errors.New("--voice " + voice + ": WAV voices are cloned by the native backend only; " +
			"run 'pockettts export-voice' and pass the .safetensors instead")
	}

	if !isWAVVoice(voice) && !model.IsRemoteVoice(voice) {
		path, err := resolveNativeVoice(cfg, backend, voice)

		return path, func() {}, err
	}

	// --backend may override the configured backend, which picks the encoder.
	cfg.TTS.Backend = backend

	path, remove, err := resolveVoiceRef("--voice", voice, fetch,
		func(wav string) (string, func(), error) { return encodeWAVVoice(cfg, wav) })
	if err != nil {
		return "", nil, err
	}

	// A downloaded voice file has no pinned checksum: load it now, so a broken
	// one fails before the model weights do. FetchVoice reuses a cached file
	// without a request, so drop the broken one for the next run to download
	// again.
	if !isWAVVoice(voice) {
		err = tts.CheckVoiceFile(path)
		if err != nil {
			_ = os.Remove(path)

			return "", nil, fmt.Errorf("--voice %q: %w", voice, err)
		}
	}

	return path, remove, nil
}

// resolveVoiceRef resolves a voice reference given by flag (--voice,
// --default-voice): a voice ID or local .safetensors path passes through, an
// https:// or hf:// voice is downloaded by fetch, and a WAV prompt (local, or
// downloaded by fetch first like upstream's download_if_necessary) is cloned
// by encode. cleanup removes a cloned voice, never the cached download, and
// is a no-op otherwise.
func resolveVoiceRef(
	flag, ref string,
	fetch func(string) (string, error),
	encode func(string) (string, func(), error),
) (string, func(), error) {
	ref = strings.TrimSpace(ref)
	noCleanup := func() {}

	if isWAVVoice(ref) {
		return cloneVoiceRef(flag, ref, fetch, encode)
	}

	if !model.IsRemoteVoice(ref) {
		return ref, noCleanup, nil
	}

	path, err := fetch(ref)
	if err != nil {
		return "", nil, fmt.Errorf("%s: %w", flag, err)
	}

	return path, noCleanup, nil
}

// cloneVoiceRef encodes the WAV prompt ref, downloading a remote one with
// fetch first. cleanup removes only the encoded voice.
func cloneVoiceRef(
	flag, ref string,
	fetch func(string) (string, error),
	encode func(string) (string, func(), error),
) (string, func(), error) {
	if !model.IsRemoteVoice(ref) {
		path, cleanup, err := encode(ref)
		if err != nil {
			return "", nil, fmt.Errorf("%s: %w", flag, err)
		}

		return path, cleanup, nil
	}

	wav, err := fetch(ref)
	if err != nil {
		return "", nil, fmt.Errorf("%s: %w", flag, err)
	}

	path, cleanup, err := encode(wav)
	if err != nil {
		// The encoder names the cache file; name the voice asked for too.
		return "", nil, fmt.Errorf("%s %q: %w", flag, ref, err)
	}

	return path, cleanup, nil
}
