package audio

import (
	"math"
	"testing"
)

func TestPeakNormalize(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		wantPeak float32
	}{
		{
			name:     "scales half-amplitude signal to 1.0",
			input:    []float32{0.0, 0.5, -0.25, 0.5},
			wantPeak: 1.0,
		},
		{
			name:     "scales quiet signal",
			input:    []float32{0.1, -0.1, 0.05},
			wantPeak: 1.0,
		},
		{
			name:     "already normalized signal unchanged",
			input:    []float32{0.0, 1.0, -0.5},
			wantPeak: 1.0,
		},
		{
			name:     "silence remains silence",
			input:    []float32{0.0, 0.0, 0.0},
			wantPeak: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Copy input to avoid mutation affecting test data.
			in := make([]float32, len(tt.input))
			copy(in, tt.input)

			got := PeakNormalize(in)
			peak := peakOf(got)

			if tt.wantPeak == 0.0 {
				if peak != 0.0 {
					t.Errorf("expected silence, got peak %f", peak)
				}

				return
			}

			if math.Abs(float64(peak-tt.wantPeak)) > 1e-6 {
				t.Errorf("peak = %f, want %f", peak, tt.wantPeak)
			}
		})
	}
}

func TestPeakNormalize_preservesRelativeAmplitudes(t *testing.T) {
	input := []float32{0.0, 0.25, 0.5}
	got := PeakNormalize(input)
	// After normalization: 0.5→1.0, 0.25→0.5, 0.0→0.0
	if math.Abs(float64(got[1]/got[2])-0.5) > 1e-6 {
		t.Errorf("relative amplitude not preserved: got[1]/got[2] = %f, want 0.5", got[1]/got[2])
	}
}

func TestDCBlock(t *testing.T) {
	const sr = 24000
	const n = sr // 1 second of audio

	t.Run("removes DC offset", func(t *testing.T) {
		// Create signal with DC offset of 0.5.
		input := make([]float32, n)
		for i := range input {
			input[i] = 0.5
		}

		got := DCBlock(input, sr)

		// After DC blocking, the mean should be near zero.
		mean := meanOf(got)
		if math.Abs(float64(mean)) > 0.01 {
			t.Errorf("mean after DC block = %f, want near 0", mean)
		}
	})

	t.Run("preserves AC content", func(t *testing.T) {
		// 1 kHz sine wave (well above DC block cutoff).
		input := make([]float32, n)
		for i := range input {
			input[i] = float32(math.Sin(2 * math.Pi * 1000 * float64(i) / float64(sr)))
		}

		inputRMS := rmsOf(input)

		got := DCBlock(input, sr)
		gotRMS := rmsOf(got)

		// RMS should be preserved within 1%.
		ratio := float64(gotRMS / inputRMS)
		if math.Abs(ratio-1.0) > 0.01 {
			t.Errorf("RMS ratio = %f, want ~1.0", ratio)
		}
	})
}

func TestFadeIn(t *testing.T) {
	const sr = 24000

	t.Run("first sample is zero", func(t *testing.T) {
		input := make([]float32, sr)
		for i := range input {
			input[i] = 1.0
		}

		got := FadeIn(input, sr, 10) // 10ms fade
		if got[0] != 0.0 {
			t.Errorf("first sample = %f, want 0.0", got[0])
		}
	})

	t.Run("sample after fade is unmodified", func(t *testing.T) {
		input := make([]float32, sr)
		for i := range input {
			input[i] = 1.0
		}

		got := FadeIn(input, sr, 10)

		fadeSamples := int(10.0 / 1000.0 * float64(sr)) // 240 samples
		if got[fadeSamples] != 1.0 {
			t.Errorf("sample at fade end = %f, want 1.0", got[fadeSamples])
		}
	})

	t.Run("ramp is monotonically increasing", func(t *testing.T) {
		input := make([]float32, sr)
		for i := range input {
			input[i] = 1.0
		}

		got := FadeIn(input, sr, 50) // 50ms

		fadeSamples := int(50.0 / 1000.0 * float64(sr))
		for i := 1; i < fadeSamples; i++ {
			if got[i] < got[i-1] {
				t.Fatalf("not monotonic at sample %d: %f < %f", i, got[i], got[i-1])
			}
		}
	})
}

func TestFadeOut(t *testing.T) {
	const sr = 24000

	t.Run("last sample is zero", func(t *testing.T) {
		input := make([]float32, sr)
		for i := range input {
			input[i] = 1.0
		}

		got := FadeOut(input, sr, 10)
		if got[len(got)-1] != 0.0 {
			t.Errorf("last sample = %f, want 0.0", got[len(got)-1])
		}
	})

	t.Run("sample before fade is unmodified", func(t *testing.T) {
		input := make([]float32, sr)
		for i := range input {
			input[i] = 1.0
		}

		got := FadeOut(input, sr, 10)
		fadeSamples := int(10.0 / 1000.0 * float64(sr))

		idx := len(got) - fadeSamples - 1
		if got[idx] != 1.0 {
			t.Errorf("sample before fade = %f, want 1.0", got[idx])
		}
	})

	t.Run("ramp is monotonically decreasing", func(t *testing.T) {
		input := make([]float32, sr)
		for i := range input {
			input[i] = 1.0
		}

		got := FadeOut(input, sr, 50)
		fadeSamples := int(50.0 / 1000.0 * float64(sr))

		start := len(got) - fadeSamples
		for i := start + 1; i < len(got); i++ {
			if got[i] > got[i-1] {
				t.Fatalf("not monotonic at sample %d: %f > %f", i, got[i], got[i-1])
			}
		}
	})
}

// Test helpers

func peakOf(s []float32) float32 {
	var peak float32
	for _, v := range s {
		if a := float32(math.Abs(float64(v))); a > peak {
			peak = a
		}
	}

	return peak
}

func meanOf(s []float32) float32 {
	var sum float64
	for _, v := range s {
		sum += float64(v)
	}

	return float32(sum / float64(len(s)))
}

func rmsOf(s []float32) float32 {
	var sum float64
	for _, v := range s {
		sum += float64(v) * float64(v)
	}

	return float32(math.Sqrt(sum / float64(len(s))))
}
