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

// DecodePromptWAV decodes a voice-prompt WAV of any sample rate and channel
// count, like upstream audio_read: channels are mixed down to mono by their
// mean and the source rate is returned for resampling. On top of DecodeWAV's
// formats it reads unsigned 8-bit PCM, scaled like libsndfile as (b−128)/128.
// Rates above MaxPromptSampleRate are rejected before the PCM data is read.
func DecodePromptWAV(data []byte) ([]float32, int, error) {
	if len(data) == 0 {
		return nil, 0, errors.New("empty WAV input")
	}

	dec := wav.NewDecoder(bytes.NewReader(data))
	if !dec.IsValidFile() {
		return nil, 0, errors.New("invalid WAV file")
	}

	if dec.SampleRate == 0 || dec.NumChans == 0 {
		return nil, 0, fmt.Errorf("%w: sample rate %d, channels %d", ErrFormatMismatch, dec.SampleRate, dec.NumChans)
	}

	if dec.SampleRate > MaxPromptSampleRate {
		return nil, 0, fmt.Errorf("%w: sample rate %d exceeds %d", ErrFormatMismatch, dec.SampleRate, MaxPromptSampleRate)
	}

	is8Bit := dec.WavAudioFormat == wavFormatPCM && dec.BitDepth == 8
	if !is8Bit && !supportedSampleFormat(dec.WavAudioFormat, dec.BitDepth) {
		return nil, 0, fmt.Errorf("%w: format tag %d with bit depth %d, want 8/16/24/32-bit PCM or 32/64-bit float",
			ErrFormatMismatch, dec.WavAudioFormat, dec.BitDepth)
	}

	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, 0, fmt.Errorf("reading PCM data: %w", err)
	}

	// cwbudde/wav (v0.1.4+) already scales unsigned 8-bit PCM like
	// libsndfile, and so upstream, as (b−128)/128.
	return downmixMean(buf.Data, int(dec.NumChans)), int(dec.SampleRate), nil
}

// downmixMean averages interleaved frames of channels samples into mono, like
// numpy's mean(axis=1) on float32 data; mono input is returned as is.
func downmixMean(interleaved []float32, channels int) []float32 {
	if channels == 1 {
		return interleaved
	}

	out := make([]float32, len(interleaved)/channels)
	for i := range out {
		var sum float32
		for _, v := range interleaved[i*channels : (i+1)*channels] {
			sum += v
		}

		out[i] = sum / float32(channels)
	}

	return out
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
