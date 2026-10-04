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

// WAV format tags accepted by DecodeWAV (WAVE_FORMAT_EXTENSIBLE files report
// their sub-format).
const (
	wavFormatPCM       = 1
	wavFormatIEEEFloat = 3
)

// ErrFormatMismatch is returned when a decoded WAV does not match the expected format.
var ErrFormatMismatch = errors.New("WAV format mismatch")

// DecodeWAV decodes WAV bytes and returns float32 PCM samples.
// It validates that the format is 24000 Hz mono, as 16/24/32-bit integer PCM
// or 32/64-bit IEEE float. Float samples are clamped to [-1, 1].
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

	if !supportedSampleFormat(dec.WavAudioFormat, dec.BitDepth) {
		return nil, fmt.Errorf("%w: format tag %d with bit depth %d, want 16/24/32-bit PCM or 32/64-bit float",
			ErrFormatMismatch, dec.WavAudioFormat, dec.BitDepth)
	}

	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, fmt.Errorf("reading PCM data: %w", err)
	}

	return buf.Data, nil
}

// supportedSampleFormat reports whether DecodeWAV accepts the WAV format tag
// and bit depth.
func supportedSampleFormat(format, bitDepth uint16) bool {
	switch format {
	case wavFormatPCM:
		return bitDepth == 16 || bitDepth == 24 || bitDepth == 32
	case wavFormatIEEEFloat:
		return bitDepth == 32 || bitDepth == 64
	default:
		return false
	}
}
