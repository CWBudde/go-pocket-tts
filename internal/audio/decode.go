package audio

import (
	"bytes"
	"errors"
	"fmt"

	"github.com/cwbudde/wav"
)

// Expected WAV format for Pocket TTS output.
const (
	ExpectedSampleRate = 24000
	ExpectedChannels   = 1
	ExpectedBitDepth   = 16
)

// ErrFormatMismatch is returned when a decoded WAV does not match the expected format.
var ErrFormatMismatch = errors.New("WAV format mismatch")

// DecodeWAV decodes WAV bytes and returns float32 PCM samples.
// It validates that the format is 24000 Hz, mono, 16-bit PCM.
func DecodeWAV(data []byte) ([]float32, error) {
	if len(data) == 0 {
		return nil, errors.New("empty WAV input")
	}

	r := bytes.NewReader(data)
	dec := wav.NewDecoder(r)
	if !dec.IsValidFile() {
		return nil, errors.New("invalid WAV file")
	}

	if dec.SampleRate != ExpectedSampleRate {
		return nil, fmt.Errorf("%w: sample rate %d, want %d", ErrFormatMismatch, dec.SampleRate, ExpectedSampleRate)
	}
	if dec.NumChans != ExpectedChannels {
		return nil, fmt.Errorf("%w: channels %d, want %d", ErrFormatMismatch, dec.NumChans, ExpectedChannels)
	}
	if dec.BitDepth != ExpectedBitDepth {
		return nil, fmt.Errorf("%w: bit depth %d, want %d", ErrFormatMismatch, dec.BitDepth, ExpectedBitDepth)
	}

	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, fmt.Errorf("reading PCM data: %w", err)
	}

	return buf.Data, nil
}
