// Package genloop holds the rules shared by the autoregressive generation
// loops of the native and ONNX backends.
package genloop

// MinFramesBeforeEOS is the first generation step at which EOS is accepted
// (upstream _MIN_FRAMES_BEFORE_EOS). Before speech starts, the EOS logit of
// some voices can cross the threshold, and a short text would then end before
// the word is spoken.
const MinFramesBeforeEOS = 6

// EOSStop decides when an autoregressive loop ends after EOS, matching
// upstream TTSModel._autoregressive_generation: the loop keeps
// eos_step + FramesAfter frames, so FramesAfter == 0 drops the EOS frame.
// The zero value with FramesAfter set is ready to use.
type EOSStop struct {
	FramesAfter int

	eosStep int
	seen    bool
}

// Stop records whether step flagged EOS and reports whether the loop must
// stop before keeping this step's frame.
func (s *EOSStop) Stop(step int, isEOS bool) bool {
	if isEOS && !s.seen && step >= MinFramesBeforeEOS {
		s.eosStep = step
		s.seen = true
	}

	return s.seen && step >= s.eosStep+s.FramesAfter
}

// EOSStep returns the step at which EOS was accepted, if any.
func (s *EOSStop) EOSStep() (int, bool) {
	return s.eosStep, s.seen
}
