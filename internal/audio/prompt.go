package audio

// VoicePromptMaxSec is the prompt length upstream keeps when its CLI
// (export-voice, serve) calls get_state_for_audio_prompt with truncate=True.
const VoicePromptMaxSec = 30

// PrepareVoicePrompt turns mono prompt samples at sampleRate into the Mimi
// encoder input, in upstream get_state_for_audio_prompt's order: truncate to
// VoicePromptMaxSec at the source rate, resample to ExpectedSampleRate
// (convert_audio), then EndOnPause. The input slice is not modified.
func PrepareVoicePrompt(samples []float32, sampleRate int) []float32 {
	if limit := VoicePromptMaxSec * sampleRate; len(samples) > limit {
		samples = samples[:limit]
	}

	samples = ResamplePoly(samples, ExpectedSampleRate, sampleRate)

	return EndOnPause(samples, ExpectedSampleRate)
}
