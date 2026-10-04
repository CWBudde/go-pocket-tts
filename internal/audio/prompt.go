package audio

import "fmt"

// VoicePromptMaxSec is the prompt length upstream keeps when its CLI
// (export-voice, serve) calls get_state_for_audio_prompt with truncate=True.
const VoicePromptMaxSec = 30

// MaxPromptSampleRate is the highest voice-prompt sample rate accepted. A WAV
// header may claim any uint32 rate, and resampling an arbitrary one to 24 kHz
// can need a filter of billions of taps; upstream has no such bound and would
// run out of memory. 384 kHz covers every common recording rate.
const MaxPromptSampleRate = 384000

// PrepareVoicePrompt turns mono prompt samples at sampleRate into the Mimi
// encoder input, in upstream get_state_for_audio_prompt's order: truncate to
// VoicePromptMaxSec at the source rate, resample to ExpectedSampleRate
// (convert_audio), then EndOnPause. sampleRate must be in
// (0, MaxPromptSampleRate]. The input slice is not modified.
func PrepareVoicePrompt(samples []float32, sampleRate int) ([]float32, error) {
	if sampleRate <= 0 || sampleRate > MaxPromptSampleRate {
		return nil, fmt.Errorf("voice prompt: sample rate %d outside (0, %d]", sampleRate, MaxPromptSampleRate)
	}

	if limit := VoicePromptMaxSec * sampleRate; len(samples) > limit {
		samples = samples[:limit]
	}

	samples, err := ResamplePoly(samples, ExpectedSampleRate, sampleRate)
	if err != nil {
		return nil, fmt.Errorf("voice prompt: %w", err)
	}

	return EndOnPause(samples, ExpectedSampleRate), nil
}
