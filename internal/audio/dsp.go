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

// FadeIn applies a linear fade-in ramp over the given duration in milliseconds.
func FadeIn(samples []float32, sampleRate int, ms float64) []float32 {
	fadeSamples := min(int(ms/1000.0*float64(sampleRate)), len(samples))

	out := make([]float32, len(samples))
	copy(out, samples)

	for i := range fadeSamples {
		gain := float32(i) / float32(fadeSamples)
		out[i] = samples[i] * gain
	}

	return out
}

// FadeOut applies a linear fade-out ramp over the given duration in milliseconds.
func FadeOut(samples []float32, sampleRate int, ms float64) []float32 {
	fadeSamples := min(int(ms/1000.0*float64(sampleRate)), len(samples))

	out := make([]float32, len(samples))
	copy(out, samples)

	start := len(samples) - fadeSamples
	for i := start; i < len(samples); i++ {
		remaining := len(samples) - 1 - i
		gain := float32(remaining) / float32(fadeSamples)
		out[i] = samples[i] * gain
	}

	return out
}
