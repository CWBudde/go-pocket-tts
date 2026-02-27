package testutil

import (
	"encoding/binary"
	"errors"
	"testing"
)

// AssertValidWAV checks that data is a valid PCM WAV file with the expected
// Pocket TTS format: RIFF header, 24000 Hz sample rate, mono, 16-bit depth,
// and at least one non-zero sample count.
func AssertValidWAV(tb testing.TB, data []byte) {
	tb.Helper()

	if len(data) < 44 {
		tb.Fatalf("WAV data too short: %d bytes", len(data))
	}

	if string(data[0:4]) != "RIFF" {
		tb.Fatalf("WAV: missing RIFF header (got %q)", string(data[0:4]))
	}

	if string(data[8:12]) != "WAVE" {
		tb.Fatalf("WAV: missing WAVE marker (got %q)", string(data[8:12]))
	}

	if string(data[12:16]) != "fmt " {
		tb.Fatalf("WAV: missing fmt chunk (got %q)", string(data[12:16]))
	}

	// fmt chunk fields (little-endian).
	audioFmt := binary.LittleEndian.Uint16(data[20:22])
	if audioFmt != 1 {
		tb.Fatalf("WAV: expected PCM format (1), got %d", audioFmt)
	}

	channels := binary.LittleEndian.Uint16(data[22:24])
	if channels != 1 {
		tb.Fatalf("WAV: expected mono (1 channel), got %d", channels)
	}

	sampleRate := binary.LittleEndian.Uint32(data[24:28])
	if sampleRate != 24000 {
		tb.Fatalf("WAV: expected sample rate 24000, got %d", sampleRate)
	}

	bitDepth := binary.LittleEndian.Uint16(data[34:36])
	if bitDepth != 16 {
		tb.Fatalf("WAV: expected 16-bit depth, got %d", bitDepth)
	}

	// Locate data chunk and verify non-zero sample count.
	dataSize, err := findDataChunkSize(data)
	if err != nil {
		tb.Fatalf("WAV: %v", err)
	}

	sampleCount := dataSize / 2 // 16-bit = 2 bytes per sample
	if sampleCount == 0 {
		tb.Fatal("WAV: data chunk contains zero samples")
	}
}

// AssertWAVDurationApprox asserts that the WAV audio duration falls within
// [minSec, maxSec]. It reads the sample count from the data chunk and uses
// a 24000 Hz sample rate.
func AssertWAVDurationApprox(tb testing.TB, data []byte, minSec, maxSec float64) {
	tb.Helper()

	dataSize, err := findDataChunkSize(data)
	if err != nil {
		tb.Fatalf("WAV duration check: %v", err)
	}
	const sampleRate = 24000
	sampleCount := dataSize / 2 // 16-bit mono

	durationSec := float64(sampleCount) / float64(sampleRate)
	if durationSec < minSec || durationSec > maxSec {
		tb.Fatalf("WAV duration %.3fs out of expected range [%.3fs, %.3fs]", durationSec, minSec, maxSec)
	}
}

// findDataChunkSize walks the WAV chunk list to locate the "data" sub-chunk
// and returns its size in bytes.
func findDataChunkSize(data []byte) (uint32, error) {
	// Start after the 12-byte RIFF/WAVE header.
	offset := 12
	for offset+8 <= len(data) {
		id := string(data[offset : offset+4])

		size := binary.LittleEndian.Uint32(data[offset+4 : offset+8])
		if id == "data" {
			return size, nil
		}

		offset += 8 + int(size)
		// Pad to even boundary.
		if size%2 != 0 {
			offset++
		}
	}

	return 0, errors.New("data chunk not found in WAV")
}
