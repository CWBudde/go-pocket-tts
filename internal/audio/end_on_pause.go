package audio

import "math"

// Defaults of upstream pocket_tts/data/audio_utils.py end_on_pause.
const (
	defaultEndOnPausePauseSec = 0.08
	defaultEndOnPauseFadeSec  = 0.02
	defaultEndOnPauseFloorDB  = 35.0
)

// EndOnPause ends a voice prompt on exactly 80 ms of silence, a port of
// upstream end_on_pause (#334).
//
// Training prompts end inside the pause between two words. A prompt that
// stops on speech makes the model continue that utterance, and one that ends
// on a long silence delays the onset. The trailing silence (20 ms frames more
// than 35 dB below the loudest one) is cut, the last 20 ms of what remains is
// faded out, and 80 ms of zeros is appended. Input shorter than one frame is
// returned as is. samples is never modified.
func EndOnPause(samples []float32, sampleRate int) []float32 {
	return endOnPause(samples, sampleRate,
		defaultEndOnPausePauseSec, defaultEndOnPauseFadeSec, defaultEndOnPauseFloorDB)
}

func endOnPause(samples []float32, sampleRate int, pauseSec, fadeSec, floorDB float64) []float32 {
	frame := max(1, int(0.02*float64(sampleRate)))

	n := len(samples) / frame
	if n == 0 {
		return samples
	}

	db := make([]float64, n)
	maxDB := math.Inf(-1)

	for i := range db {
		var sum float64
		for _, v := range samples[i*frame : (i+1)*frame] {
			sum += float64(v) * float64(v)
		}

		db[i] = 20 * math.Log10(math.Sqrt(sum/float64(frame))+1e-12)
		maxDB = max(maxDB, db[i])
	}

	last := 0

	for i, v := range db {
		if v > maxDB-floorDB {
			last = i
		}
	}

	end := (last + 1) * frame
	pause := int(pauseSec * float64(sampleRate))

	out := make([]float32, end+pause)
	copy(out, samples[:end])

	fade := min(int(fadeSec*float64(sampleRate)), end)
	for i, g := range linspaceDown(fade) {
		out[end-fade+i] *= g
	}

	return out
}

// linspaceDown returns torch.linspace(1, 0, n) in float32: steps from the start
// for the first half and from the end for the second, as torch computes them.
func linspaceDown(n int) []float32 {
	out := make([]float32, n)
	if n == 1 {
		out[0] = 1
	}

	if n <= 1 {
		return out
	}

	const start, stop float32 = 1, 0

	step := (stop - start) / float32(n-1)
	half := n / 2

	for i := range out {
		if i < half {
			out[i] = start + step*float32(i)
		} else {
			out[i] = stop - step*float32(n-1-i)
		}
	}

	return out
}
