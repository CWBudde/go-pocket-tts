package bench_test

import (
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/bench"
)

func TestWAVDuration_TooShort(t *testing.T) {
	_, err := bench.WAVDuration(make([]byte, 10))
	if err == nil || !strings.Contains(err.Error(), "too short") {
		t.Fatalf("expected too-short error, got: %v", err)
	}
}

func TestWAVDuration_NotRIFF(t *testing.T) {
	data := make([]byte, 44)
	copy(data[0:4], "JUNK")
	copy(data[8:12], "WAVE")

	_, err := bench.WAVDuration(data)
	if err == nil || !strings.Contains(err.Error(), "RIFF") {
		t.Fatalf("expected RIFF error, got: %v", err)
	}
}

func TestWAVDuration_NotWAVE(t *testing.T) {
	data := make([]byte, 44)
	copy(data[0:4], "RIFF")
	copy(data[8:12], "JUNK")

	_, err := bench.WAVDuration(data)
	if err == nil || !strings.Contains(err.Error(), "RIFF") {
		t.Fatalf("expected RIFF/WAVE error, got: %v", err)
	}
}

func TestWAVDuration_NoFmtChunk(t *testing.T) {
	// Valid RIFF/WAVE header but no fmt chunk — only a data chunk.
	data := make([]byte, 44)
	copy(data[0:4], "RIFF")
	copy(data[8:12], "WAVE")
	copy(data[12:16], "data")
	data[16] = 20 // chunkSize=20

	_, err := bench.WAVDuration(data)
	if err == nil || !strings.Contains(err.Error(), "fmt") {
		t.Fatalf("expected fmt-not-found error, got: %v", err)
	}
}
