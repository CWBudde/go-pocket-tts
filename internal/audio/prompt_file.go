package audio

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ReadVoicePrompt reads a voice prompt file as mono samples and their sample
// rate: a .wav of any rate and channel count (mixed down, see
// DecodePromptWAV), else raw ExpectedSampleRate mono PCM16LE.
func ReadVoicePrompt(path string) ([]float32, int, error) {
	if strings.TrimSpace(path) == "" {
		return nil, 0, errors.New("voice prompt: path must not be empty")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, fmt.Errorf("voice prompt: read %q: %w", path, err)
	}

	if len(data) == 0 {
		return nil, 0, fmt.Errorf("voice prompt: %q is empty", path)
	}

	if strings.EqualFold(filepath.Ext(path), ".wav") {
		samples, sampleRate, err := DecodePromptWAV(data)
		if err != nil {
			return nil, 0, fmt.Errorf("voice prompt: decode WAV %q: %w", path, err)
		}

		return samples, sampleRate, nil
	}

	samples, err := decodePCM16LE(data)
	if err != nil {
		return nil, 0, fmt.Errorf("voice prompt: decode raw PCM16 %q: %w", path, err)
	}

	return samples, ExpectedSampleRate, nil
}

func decodePCM16LE(data []byte) ([]float32, error) {
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("byte length %d is not a multiple of 2", len(data))
	}

	out := make([]float32, len(data)/2)
	for i := range out {
		pcm := int16(uint16(data[i*2]) | uint16(data[i*2+1])<<8)
		out[i] = float32(pcm) / 32768.0
	}

	return out, nil
}
