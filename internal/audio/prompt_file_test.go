package audio

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestReadVoicePrompt_RawPCM16(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "prompt.pcm")

	// 0x4000 = 16384, 0x8000 = -32768, 0xffff = -1 (little-endian).
	err := os.WriteFile(path, []byte{0x00, 0x40, 0x00, 0x80, 0xff, 0xff}, 0o600)
	if err != nil {
		t.Fatal(err)
	}

	samples, rate, err := ReadVoicePrompt(path)
	if err != nil {
		t.Fatalf("ReadVoicePrompt: %v", err)
	}

	want := []float32{0.5, -1, -1.0 / 32768}
	if rate != ExpectedSampleRate || len(samples) != len(want) {
		t.Fatalf("got %d samples at %d Hz, want %d at %d Hz", len(samples), rate, len(want), ExpectedSampleRate)
	}

	for i := range want {
		if samples[i] != want[i] {
			t.Fatalf("samples = %v, want %v", samples, want)
		}
	}
}

func TestReadVoicePrompt_WAVKeepsItsRate(t *testing.T) {
	t.Parallel()

	wav, err := EncodeWAVPCM16([]float32{0.25, -0.25, 0}, 16000)
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "prompt.WAV")

	err = os.WriteFile(path, wav, 0o600)
	if err != nil {
		t.Fatal(err)
	}

	samples, rate, err := ReadVoicePrompt(path)
	if err != nil {
		t.Fatalf("ReadVoicePrompt: %v", err)
	}

	// EncodeWAVPCM16 scales by 32767, decoding divides by 32768.
	if rate != 16000 || len(samples) != 3 || math.Abs(float64(samples[0])-0.25) > 1e-4 {
		t.Fatalf("got %v at %d Hz, want [0.25 -0.25 0] at 16000 Hz", samples, rate)
	}
}

func TestReadVoicePrompt_Rejects(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	odd := filepath.Join(dir, "odd.pcm")
	empty := filepath.Join(dir, "empty.pcm")

	err := os.WriteFile(odd, []byte{1, 2, 3}, 0o600)
	if err != nil {
		t.Fatal(err)
	}

	err = os.WriteFile(empty, nil, 0o600)
	if err != nil {
		t.Fatal(err)
	}

	for _, path := range []string{"", " ", odd, empty, filepath.Join(dir, "missing.wav")} {
		_, _, err := ReadVoicePrompt(path)
		if err == nil {
			t.Errorf("ReadVoicePrompt(%q) succeeded", path)
		}
	}
}
