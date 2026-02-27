package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// --- ApplyHooks ---

func TestApplyHooks_NoHooks(t *testing.T) {
	samples := []float32{0.1, 0.2, 0.3}

	got := ApplyHooks(samples)
	if len(got) != len(samples) {
		t.Fatalf("ApplyHooks() len = %d; want %d", len(got), len(samples))
	}

	for i, v := range samples {
		if got[i] != v {
			t.Errorf("ApplyHooks()[%d] = %v; want %v", i, got[i], v)
		}
	}
}

func TestApplyHooks_SingleHook(t *testing.T) {
	scale := func(s []float32) []float32 {
		out := make([]float32, len(s))
		for i, v := range s {
			out[i] = v * 2
		}

		return out
	}

	samples := []float32{0.1, 0.5, 1.0}
	got := ApplyHooks(samples, scale)

	want := []float32{0.2, 1.0, 2.0}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Errorf("ApplyHooks()[%d] = %v; want %v", i, got[i], want[i])
		}
	}
}

func TestApplyHooks_MultipleHooks_AppliedInOrder(t *testing.T) {
	var order []int
	h1 := func(s []float32) []float32 { order = append(order, 1); return s }
	h2 := func(s []float32) []float32 { order = append(order, 2); return s }
	h3 := func(s []float32) []float32 { order = append(order, 3); return s }

	ApplyHooks([]float32{0}, h1, h2, h3)

	if len(order) != 3 || order[0] != 1 || order[1] != 2 || order[2] != 3 {
		t.Errorf("hooks applied in wrong order: %v", order)
	}
}

func TestApplyHooks_EmptySamples(t *testing.T) {
	got := ApplyHooks([]float32{})
	if len(got) != 0 {
		t.Errorf("ApplyHooks(empty) = %v; want empty", got)
	}
}

// --- EncodeWAVPCM16 ---

func TestEncodeWAVPCM16_InvalidSampleRate(t *testing.T) {
	tests := []struct {
		name       string
		sampleRate int
	}{
		{"zero", 0},
		{"negative", -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := EncodeWAVPCM16([]float32{0.1}, tt.sampleRate)
			if err == nil {
				t.Errorf("EncodeWAVPCM16(rate=%d) = nil; want error", tt.sampleRate)
			}
		})
	}
}

func TestEncodeWAVPCM16_ValidOutput(t *testing.T) {
	samples := []float32{0.0, 0.5, -0.5, 1.0, -1.0}
	sampleRate := 22050

	data, err := EncodeWAVPCM16(samples, sampleRate)
	if err != nil {
		t.Fatalf("EncodeWAVPCM16 error = %v", err)
	}

	// Minimum WAV size: 44 bytes header + 2 bytes per sample.
	minSize := 44 + len(samples)*2
	if len(data) < minSize {
		t.Errorf("output length = %d; want at least %d", len(data), minSize)
	}

	// Verify RIFF header.
	if !bytes.HasPrefix(data, []byte("RIFF")) {
		t.Error("output does not start with RIFF")
	}

	if !bytes.Contains(data[:12], []byte("WAVE")) {
		t.Error("output does not contain WAVE marker")
	}
}

func TestEncodeWAVPCM16_SampleRateInHeader(t *testing.T) {
	sampleRate := 16000

	data, err := EncodeWAVPCM16([]float32{0}, sampleRate)
	if err != nil {
		t.Fatalf("EncodeWAVPCM16 error = %v", err)
	}

	// Sample rate is at offset 24 in a standard WAV header (little-endian uint32).
	got := binary.LittleEndian.Uint32(data[24:28])
	if int(got) != sampleRate {
		t.Errorf("sample rate in header = %d; want %d", got, sampleRate)
	}
}

func TestEncodeWAVPCM16_Clamping(t *testing.T) {
	// Values > 1.0 and < -1.0 must be clamped to int16 range.
	samples := []float32{2.0, -2.0}

	data, err := EncodeWAVPCM16(samples, 44100)
	if err != nil {
		t.Fatalf("EncodeWAVPCM16 error = %v", err)
	}

	// PCM data starts after 44-byte header.
	v1 := int16(binary.LittleEndian.Uint16(data[44:46]))
	v2 := int16(binary.LittleEndian.Uint16(data[46:48]))

	if v1 != 32767 {
		t.Errorf("clamped +2.0 = %d; want 32767", v1)
	}

	if v2 != -32767 {
		t.Errorf("clamped -2.0 = %d; want -32767", v2)
	}
}

func TestEncodeWAVPCM16_EmptySamples(t *testing.T) {
	data, err := EncodeWAVPCM16([]float32{}, 44100)
	if err != nil {
		t.Fatalf("EncodeWAVPCM16(empty) error = %v", err)
	}
	// Should still produce a valid (minimal) WAV header.
	if len(data) < 44 {
		t.Errorf("empty WAV length = %d; want at least 44", len(data))
	}
}
