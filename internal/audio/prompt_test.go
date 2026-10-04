package audio

import (
	"math"
	"slices"
	"testing"
)

// TestPrepareVoicePrompt_TruncatesAtSourceRateThenResamples feeds a 31 s
// prompt at 48 kHz with speech at 0–1 s, 20–21 s and 29.5–31 s. Upstream cuts
// it to 30 s at 48 kHz, resamples, then ends it on a pause, so the output ends
// just after 30 s: the 20 s burst survives, the cut-off tail does not.
func TestPrepareVoicePrompt_TruncatesAtSourceRateThenResamples(t *testing.T) {
	const rate = 48000

	in := make([]float32, 31*rate)
	for _, burst := range [][2]float64{{0, 1}, {20, 21}, {29.5, 31}} {
		for i := int(burst[0] * rate); i < int(burst[1]*rate); i++ {
			in[i] = float32(0.5 * math.Sin(float64(i)/7))
		}
	}

	orig := slices.Clone(in)

	got := mustPrepareVoicePrompt(t, in, rate)

	want := EndOnPause(mustResamplePoly(t, in[:VoicePromptMaxSec*rate], ExpectedSampleRate, rate), ExpectedSampleRate)
	if !slices.Equal(got, want) {
		t.Errorf("output (%d samples) differs from truncate → resample → EndOnPause (%d samples)", len(got), len(want))
	}

	const pause = ExpectedSampleRate * 8 / 100
	if lo, hi := 29*ExpectedSampleRate, VoicePromptMaxSec*ExpectedSampleRate+pause; len(got) < lo || len(got) > hi {
		t.Errorf("len = %d (%.2f s at 24 kHz), want within [%d, %d]",
			len(got), float64(len(got))/ExpectedSampleRate, lo, hi)
	}

	if !slices.Equal(in, orig) {
		t.Error("input slice was modified")
	}
}

func TestPrepareVoicePrompt_ModelRateIsEndOnPauseOnly(t *testing.T) {
	in := append(tone(ExpectedSampleRate/2), make([]float32, ExpectedSampleRate/4)...)

	if got, want := mustPrepareVoicePrompt(t, in, ExpectedSampleRate), EndOnPause(in, ExpectedSampleRate); !slices.Equal(got, want) {
		t.Errorf("24 kHz prompt: got %d samples, want EndOnPause's %d", len(got), len(want))
	}
}

func TestPrepareVoicePrompt_CommonRates(t *testing.T) {
	for _, rate := range []int{8000, 16000, 44100, 48000, MaxPromptSampleRate} {
		in := append(tone(rate/2), make([]float32, rate/4)...)

		got := mustPrepareVoicePrompt(t, in, rate)

		// 0.5 s of tone at 24 kHz plus the 80 ms pause, give or take the
		// resampling filter's ringing into the silence.
		const pause = ExpectedSampleRate * 8 / 100
		if want := ExpectedSampleRate/2 + pause; len(got) < want || len(got) > want+ExpectedSampleRate/50 {
			t.Errorf("rate %d: len = %d, want about %d", rate, len(got), want)
		}
	}
}

func TestPrepareVoicePrompt_RejectsImpracticalRate(t *testing.T) {
	for _, rate := range []int{0, -8000, MaxPromptSampleRate + 1, maxWAVRate} {
		_, err := PrepareVoicePrompt(make([]float32, 16), rate)
		if err == nil {
			t.Errorf("rate %d: want error", rate)
		}
	}
}

func mustPrepareVoicePrompt(t *testing.T, samples []float32, rate int) []float32 {
	t.Helper()

	out, err := PrepareVoicePrompt(samples, rate)
	if err != nil {
		t.Fatalf("PrepareVoicePrompt(rate %d): %v", rate, err)
	}

	return out
}
