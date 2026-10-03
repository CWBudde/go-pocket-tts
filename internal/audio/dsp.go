package audio

import (
	"math"

	"github.com/cwbudde/algo-dsp/dsp/filter/biquad"
	"github.com/cwbudde/algo-dsp/dsp/filter/design"
)

// PeakNormalize scales samples so the peak amplitude reaches 1.0.
// If all samples are zero, the input is returned unchanged.
func PeakNormalize(samples []float32) []float32 {
	var peak float32
	for _, v := range samples {
		if a := float32(math.Abs(float64(v))); a > peak {
			peak = a
		}
	}

	if peak == 0 {
		return samples
	}

	gain := 1.0 / peak

	out := make([]float32, len(samples))
	for i, v := range samples {
		out[i] = v * gain
	}

	return out
}

// DCBlock removes DC offset from samples using a high-pass biquad filter
// from the algo-dsp library. The cutoff is set at 20 Hz.
func DCBlock(samples []float32, sampleRate int) []float32 {
	coeffs := design.Highpass(20.0, 0.707, float64(sampleRate))
	section := biquad.NewSection(coeffs)

	out := make([]float32, len(samples))
	for i, v := range samples {
		out[i] = float32(section.ProcessSample(float64(v)))
	}

	return out
}

// linspaceGain returns element i of torch.linspace(0, 1, n): 0 for the first
// element and exactly 1 for the last.
func linspaceGain(i, n int) float32 {
	if n <= 1 {
		return 0
	}

	return float32(i) / float32(n-1)
}

// LinearRamp multiplies the first n samples in place by linspace(0, 1, n);
// n is clamped to len(samples).
func LinearRamp(samples []float32, n int) {
	n = min(n, len(samples))
	for i := range n {
		samples[i] *= linspaceGain(i, n)
	}
}

// ChunkFadeIn applies upstream's per-chunk fade-in in place: a fresh Mimi
// decoder state puts a small step in its first samples, heard as a click, so
// the first 5 ms (sampleRate/200 samples) are ramped by linspace(0, 1, n).
func ChunkFadeIn(samples []float32, sampleRate int) {
	LinearRamp(samples, sampleRate/200)
}

// FadeIn applies a linear fade-in ramp over the given duration in milliseconds.
// The ramp runs from 0 on the first sample to 1 on the last faded sample.
func FadeIn(samples []float32, sampleRate int, ms float64) []float32 {
	out := make([]float32, len(samples))
	copy(out, samples)

	LinearRamp(out, int(ms/1000.0*float64(sampleRate)))

	return out
}

// FadeOut applies a linear fade-out ramp over the given duration in milliseconds.
// The ramp runs from 1 on the first faded sample to 0 on the last sample.
func FadeOut(samples []float32, sampleRate int, ms float64) []float32 {
	fadeSamples := min(int(ms/1000.0*float64(sampleRate)), len(samples))

	out := make([]float32, len(samples))
	copy(out, samples)

	start := len(samples) - fadeSamples
	for i := range fadeSamples {
		out[start+i] *= linspaceGain(fadeSamples-1-i, fadeSamples)
	}

	return out
}
