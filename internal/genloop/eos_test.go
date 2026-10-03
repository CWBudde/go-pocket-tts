package genloop

import "testing"

// framesKept runs the stop rule over eos (one entry per step, true where the
// model flags EOS) and returns how many frames the loop keeps.
func framesKept(framesAfter int, eos []bool) int {
	stop := EOSStop{FramesAfter: framesAfter}

	for step, isEOS := range eos {
		if stop.Stop(step, isEOS) {
			return step
		}
	}

	return len(eos)
}

// eosAt returns n steps with EOS flagged at the given steps.
func eosAt(n int, steps ...int) []bool {
	eos := make([]bool, n)
	for _, s := range steps {
		eos[s] = true
	}

	return eos
}

func TestEOSStop(t *testing.T) {
	for _, tc := range []struct {
		name        string
		framesAfter int
		eos         []bool
		want        int
	}{
		// Upstream keeps eos_step + frames_after_eos frames.
		{"eos at 10, 3 after", 3, eosAt(40, 10), 13},
		{"eos at 10, 1 after", 1, eosAt(40, 10), 11},
		{"eos at 10, 0 after drops the eos frame", 0, eosAt(40, 10), 10},
		{"eos at the first allowed step", 2, eosAt(40, MinFramesBeforeEOS), MinFramesBeforeEOS + 2},
		{"eos before the minimum is ignored", 2, eosAt(40, 2), 40},
		{"early eos ignored, later eos counts", 2, eosAt(40, 2, 9), 11},
		{"later eos does not restart the countdown", 3, eosAt(40, 10, 12), 13},
		{"no eos runs to max steps", 3, eosAt(40), 40},
		{"countdown past max steps", 5, eosAt(12, 10), 12},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := framesKept(tc.framesAfter, tc.eos); got != tc.want {
				t.Errorf("frames kept = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestEOSStop_EOSStep(t *testing.T) {
	stop := EOSStop{FramesAfter: 2}

	if _, ok := stop.EOSStep(); ok {
		t.Fatal("EOSStep reported before any EOS")
	}

	stop.Stop(3, true)

	if _, ok := stop.EOSStep(); ok {
		t.Fatal("EOS before the minimum frames was recorded")
	}

	stop.Stop(7, true)

	if step, ok := stop.EOSStep(); !ok || step != 7 {
		t.Fatalf("EOSStep = %d, %v; want 7, true", step, ok)
	}
}
