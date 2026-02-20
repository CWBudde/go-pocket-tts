package audio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// makeWAV builds a minimal valid WAV file from parameters for testing.
func makeWAV(sampleRate uint32, numChannels uint16, bitDepth uint16, numSamples int) []byte {
	blockAlign := numChannels * bitDepth / 8
	byteRate := sampleRate * uint32(blockAlign)
	dataSize := uint32(numSamples) * uint32(blockAlign)
	riffSize := 4 + (8 + 16) + (8 + dataSize)

	buf := &bytes.Buffer{}
	buf.WriteString("RIFF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(riffSize))
	buf.WriteString("WAVE")

	// fmt chunk
	buf.WriteString("fmt ")
	_ = binary.Write(buf, binary.LittleEndian, uint32(16)) // chunk size
	_ = binary.Write(buf, binary.LittleEndian, uint16(1))  // PCM
	_ = binary.Write(buf, binary.LittleEndian, numChannels)
	_ = binary.Write(buf, binary.LittleEndian, sampleRate)
	_ = binary.Write(buf, binary.LittleEndian, byteRate)
	_ = binary.Write(buf, binary.LittleEndian, blockAlign)
	_ = binary.Write(buf, binary.LittleEndian, bitDepth)

	// data chunk
	buf.WriteString("data")
	_ = binary.Write(buf, binary.LittleEndian, dataSize)
	for range numSamples {
		_ = binary.Write(buf, binary.LittleEndian, int16(0))
	}

	return buf.Bytes()
}

func TestDecodeWAV(t *testing.T) {
	t.Run("decodes valid 24kHz mono 16-bit WAV", func(t *testing.T) {
		wav := makeWAV(24000, 1, 16, 100)
		samples, err := DecodeWAV(wav)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(samples) != 100 {
			t.Errorf("got %d samples, want 100", len(samples))
		}
	})

	t.Run("rejects wrong sample rate", func(t *testing.T) {
		wav := makeWAV(44100, 1, 16, 10)
		_, err := DecodeWAV(wav)
		if err == nil {
			t.Fatal("expected error for wrong sample rate")
		}
		if !errors.Is(err, ErrFormatMismatch) {
			t.Errorf("expected ErrFormatMismatch, got %v", err)
		}
	})

	t.Run("rejects stereo", func(t *testing.T) {
		wav := makeWAV(24000, 2, 16, 10)
		_, err := DecodeWAV(wav)
		if err == nil {
			t.Fatal("expected error for stereo")
		}
		if !errors.Is(err, ErrFormatMismatch) {
			t.Errorf("expected ErrFormatMismatch, got %v", err)
		}
	})

	t.Run("rejects invalid WAV data", func(t *testing.T) {
		_, err := DecodeWAV([]byte("not a wav file"))
		if err == nil {
			t.Fatal("expected error for invalid WAV")
		}
	})

	t.Run("rejects empty input", func(t *testing.T) {
		_, err := DecodeWAV(nil)
		if err == nil {
			t.Fatal("expected error for nil input")
		}
	})
}

func TestEncodeWAV(t *testing.T) {
	t.Run("produces valid WAV with RIFF header", func(t *testing.T) {
		samples := make([]float32, 100)
		data, err := EncodeWAV(samples)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(data) < 44 {
			t.Fatalf("WAV too short: %d bytes", len(data))
		}
		if string(data[:4]) != "RIFF" {
			t.Errorf("missing RIFF header")
		}
		if string(data[8:12]) != "WAVE" {
			t.Errorf("missing WAVE identifier")
		}
	})

	t.Run("encodes correct sample rate and channels", func(t *testing.T) {
		samples := make([]float32, 50)
		data, err := EncodeWAV(samples)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Parse fmt chunk: sample rate at byte 24, channels at byte 22.
		sampleRate := binary.LittleEndian.Uint32(data[24:28])
		numChans := binary.LittleEndian.Uint16(data[22:24])
		bitDepth := binary.LittleEndian.Uint16(data[34:36])

		if sampleRate != ExpectedSampleRate {
			t.Errorf("sample rate = %d, want %d", sampleRate, ExpectedSampleRate)
		}
		if numChans != ExpectedChannels {
			t.Errorf("channels = %d, want %d", numChans, ExpectedChannels)
		}
		if bitDepth != ExpectedBitDepth {
			t.Errorf("bit depth = %d, want %d", bitDepth, ExpectedBitDepth)
		}
	})
}

func TestDecodeEncodeRoundtrip(t *testing.T) {
	// Create samples with known values.
	original := []float32{0.0, 0.5, -0.5, 1.0, -1.0}
	encoded, err := EncodeWAV(original)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	decoded, err := DecodeWAV(encoded)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	if len(decoded) != len(original) {
		t.Fatalf("roundtrip: got %d samples, want %d", len(decoded), len(original))
	}

	// 16-bit quantization introduces error up to ~1/32768.
	const tolerance = 1.0 / 32768.0 * 2
	for i, want := range original {
		got := decoded[i]
		if math.Abs(float64(got-want)) > tolerance {
			t.Errorf("sample[%d] = %f, want %f (tolerance %f)", i, got, want, tolerance)
		}
	}
}
