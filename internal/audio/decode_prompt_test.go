package audio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"slices"
	"testing"
)

// rawWAV builds a WAV with a 16-byte fmt chunk and the given data bytes.
func rawWAV(format uint16, sampleRate uint32, channels, bitDepth uint16, data []byte) []byte {
	blockAlign := channels * bitDepth / 8

	buf := &bytes.Buffer{}
	buf.WriteString("RIFF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(4+8+16+8+len(data)))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(buf, binary.LittleEndian, format)
	_ = binary.Write(buf, binary.LittleEndian, channels)
	_ = binary.Write(buf, binary.LittleEndian, sampleRate)
	_ = binary.Write(buf, binary.LittleEndian, sampleRate*uint32(blockAlign))
	_ = binary.Write(buf, binary.LittleEndian, blockAlign)
	_ = binary.Write(buf, binary.LittleEndian, bitDepth)
	buf.WriteString("data")
	_ = binary.Write(buf, binary.LittleEndian, uint32(len(data)))
	buf.Write(data)

	return buf.Bytes()
}

func int16Bytes(values ...int16) []byte {
	out := make([]byte, 0, 2*len(values))
	for _, v := range values {
		out = binary.LittleEndian.AppendUint16(out, uint16(v))
	}

	return out
}

// Port of upstream tests/test_audio.py::test_audio_read_uses_soundfile_for_8_bit_wav.
func TestDecodePromptWAV_8BitSilence(t *testing.T) {
	data := rawWAV(wavFormatPCM, 8000, 1, 8, bytes.Repeat([]byte{128}, 16))

	samples, rate, err := DecodePromptWAV(data)
	if err != nil {
		t.Fatalf("DecodePromptWAV: %v", err)
	}

	if rate != 8000 {
		t.Errorf("rate = %d, want 8000", rate)
	}

	if len(samples) != 16 {
		t.Fatalf("len = %d, want 16", len(samples))
	}

	for i, v := range samples {
		if v != 0 {
			t.Fatalf("sample %d = %v, want 0", i, v)
		}
	}
}

func TestDecodePromptWAV_8BitScale(t *testing.T) {
	data := rawWAV(wavFormatPCM, 8000, 1, 8, []byte{0, 64, 128, 192, 255})

	samples, _, err := DecodePromptWAV(data)
	if err != nil {
		t.Fatalf("DecodePromptWAV: %v", err)
	}

	// libsndfile: (b - 128) / 128.
	want := []float32{-1, -0.5, 0, 0.5, 127.0 / 128}
	if !slices.Equal(samples, want) {
		t.Errorf("samples = %v, want %v", samples, want)
	}
}

func TestDecodePromptWAV_DownmixesByMean(t *testing.T) {
	tests := []struct {
		name     string
		channels uint16
		frames   []int16
		want     []float32
	}{
		{
			name:     "stereo",
			channels: 2,
			frames:   []int16{16384, -8192, 32767, 32767, -32768, 0},
			want:     []float32{(0.5 - 0.25) / 2, float32(32767) / 32768, -0.5},
		},
		{
			name:     "three channels",
			channels: 3,
			frames:   []int16{16384, 8192, 0, -16384, -16384, -16384},
			want:     []float32{(0.5 + 0.25) / 3, -0.5},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data := rawWAV(wavFormatPCM, 44100, tc.channels, 16, int16Bytes(tc.frames...))

			samples, rate, err := DecodePromptWAV(data)
			if err != nil {
				t.Fatalf("DecodePromptWAV: %v", err)
			}

			if rate != 44100 {
				t.Errorf("rate = %d, want 44100", rate)
			}

			if !slices.Equal(samples, tc.want) {
				t.Errorf("samples = %v, want %v", samples, tc.want)
			}
		})
	}
}

func TestDecodePromptWAV_MatchesDecodeWAVForModelFormat(t *testing.T) {
	in := []float32{0, 0.25, -0.5, 0.75, -1}

	for _, tc := range []struct {
		name             string
		bitDepth, format int
	}{
		{"int16", 16, wavFormatPCM},
		{"int24", 24, wavFormatPCM},
		{"float32", 32, wavFormatIEEEFloat},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data := encodeTestWAV(t, in, tc.bitDepth, tc.format)

			want, err := DecodeWAV(data)
			if err != nil {
				t.Fatalf("DecodeWAV: %v", err)
			}

			got, rate, err := DecodePromptWAV(data)
			if err != nil {
				t.Fatalf("DecodePromptWAV: %v", err)
			}

			if rate != ExpectedSampleRate || !slices.Equal(got, want) {
				t.Errorf("got %v at %d Hz, want %v at %d Hz", got, rate, want, ExpectedSampleRate)
			}
		})
	}
}

func TestDecodePromptWAV_Rejects(t *testing.T) {
	const wavFormatALaw = 6

	alaw := rawWAV(wavFormatALaw, 8000, 1, 8, []byte{0xd5, 0xd5})

	_, _, err := DecodePromptWAV(alaw)
	if !errors.Is(err, ErrFormatMismatch) {
		t.Errorf("A-law: err = %v, want ErrFormatMismatch", err)
	}

	_, _, err = DecodePromptWAV(nil)
	if err == nil {
		t.Error("empty input: want error")
	}

	_, _, err = DecodePromptWAV([]byte("not a wav file at all, just bytes"))
	if err == nil {
		t.Error("garbage input: want error")
	}
}
