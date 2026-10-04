package audio

import (
	"math"
	"slices"
	"testing"
)

const endOnPauseSR = 24000

// tone returns n samples of sin(i/5), the upstream test signal.
func tone(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(float64(i) / 5))
	}

	return out
}

// Port of upstream tests/test_end_on_pause.py::test_end_on_pause_trims_fades_and_pads.
func TestEndOnPause_TrimsFadesAndPads(t *testing.T) {
	speech := tone(endOnPauseSR) // 1 s of tone
	fade := int(0.02 * endOnPauseSR)

	// A prompt stopping on speech, and one with 0.5 s of silence.
	for _, tail := range []int{0, endOnPauseSR / 2} {
		wav := append(slices.Clone(speech), make([]float32, tail)...)

		out := endOnPause(wav, endOnPauseSR, 0.2, defaultEndOnPauseFadeSec, defaultEndOnPauseFloorDB)

		if want := endOnPauseSR + int(0.2*endOnPauseSR); len(out) != want {
			t.Fatalf("tail %d: len = %d, want %d", tail, len(out), want)
		}

		if !slices.Equal(out[:endOnPauseSR-fade], speech[:endOnPauseSR-fade]) {
			t.Errorf("tail %d: samples before the fade changed", tail)
		}

		if v := math.Abs(float64(out[endOnPauseSR-1])); v >= 1e-6 {
			t.Errorf("tail %d: last speech sample = %v, want faded to 0", tail, v)
		}

		for i, v := range out[endOnPauseSR:] {
			if v != 0 {
				t.Fatalf("tail %d: pause sample %d = %v, want 0", tail, i, v)
			}
		}
	}
}

// Port of upstream tests/test_end_on_pause.py::test_end_on_pause_silent_input_is_returned_whole.
func TestEndOnPause_SilentInputIsReturnedWhole(t *testing.T) {
	if got := EndOnPause(make([]float32, 100), endOnPauseSR); len(got) < 100 {
		t.Fatalf("len = %d, want >= 100", len(got))
	}
}

func TestEndOnPause_DefaultsAndInputUntouched(t *testing.T) {
	const frame = endOnPauseSR / 50 // 20 ms

	speech := tone(10 * frame)

	// A trailing frame 40 dB below the tone is more than 35 dB down: cut.
	quiet := tone(frame)
	for i := range quiet {
		quiet[i] *= 0.01
	}

	wav := append(slices.Clone(speech), quiet...)
	orig := slices.Clone(wav)

	out := EndOnPause(wav, endOnPauseSR)

	if want := len(speech) + endOnPauseSR*8/100; len(out) != want {
		t.Fatalf("len = %d, want %d (speech + 80 ms pause)", len(out), want)
	}

	if !slices.Equal(wav, orig) {
		t.Error("EndOnPause modified its input")
	}

	// The 20 ms fade is torch.linspace(1, 0, fade): the first faded sample is
	// untouched, the last is zero.
	fadeStart := len(speech) - frame
	if out[fadeStart] != speech[fadeStart] {
		t.Errorf("first faded sample = %v, want %v", out[fadeStart], speech[fadeStart])
	}

	if out[len(speech)-1] != 0 {
		t.Errorf("last faded sample = %v, want 0", out[len(speech)-1])
	}

	// All-silent input longer than a frame keeps its full frames (upstream:
	// every frame is within 35 dB of the loudest) and gets the pause.
	silent := EndOnPause(make([]float32, 3*frame+7), endOnPauseSR)
	if want := 3*frame + endOnPauseSR*8/100; len(silent) != want {
		t.Errorf("silent len = %d, want %d", len(silent), want)
	}
}

func TestLinspaceDown(t *testing.T) {
	got := linspaceDown(5)
	want := []float32{1, 0.75, 0.5, 0.25, 0}

	if !slices.Equal(got, want) {
		t.Fatalf("linspaceDown(5) = %v, want %v", got, want)
	}

	if got := linspaceDown(1); !slices.Equal(got, []float32{1}) {
		t.Fatalf("linspaceDown(1) = %v, want [1]", got)
	}
}
