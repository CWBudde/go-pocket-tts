// Package bench provides benchmarking primitives for the pockettts bench command.
package bench

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Run result and stats
// ---------------------------------------------------------------------------

// RunResult holds the timing and audio metadata for a single synthesis run.
type RunResult struct {
	Index       int
	Cold        bool // true for the first run (cold-start)
	Duration    time.Duration
	WAVDuration time.Duration
	RTF         float64
}

// Stats holds aggregate timing statistics across all runs.
type Stats struct {
	Min  time.Duration
	Max  time.Duration
	Mean time.Duration
}

// ComputeStats calculates min, max and mean over a slice of durations.
// The slice must be non-empty.
func ComputeStats(durations []time.Duration) Stats {
	if len(durations) == 0 {
		return Stats{}
	}
	mn, mx := durations[0], durations[0]
	var sum time.Duration
	for _, d := range durations {
		if d < mn {
			mn = d
		}
		if d > mx {
			mx = d
		}
		sum += d
	}
	return Stats{
		Min:  mn,
		Max:  mx,
		Mean: sum / time.Duration(len(durations)),
	}
}

// ---------------------------------------------------------------------------
// RTF helpers
// ---------------------------------------------------------------------------

// CalcRTF returns synthesis_duration / audio_duration.
// Returns 0 if audioDur is zero to avoid division by zero.
func CalcRTF(synthDur, audioDur time.Duration) float64 {
	if audioDur <= 0 {
		return 0
	}
	return float64(synthDur) / float64(audioDur)
}

// WAVDuration returns the playback duration of a WAV file from its RIFF header.
// It supports 16-bit PCM WAV files only.
func WAVDuration(wav []byte) (time.Duration, error) {
	// Minimal RIFF/WAV header is 44 bytes.
	if len(wav) < 44 {
		return 0, fmt.Errorf("wav too short (%d bytes)", len(wav))
	}
	if string(wav[0:4]) != "RIFF" || string(wav[8:12]) != "WAVE" {
		return 0, fmt.Errorf("not a RIFF/WAVE file")
	}

	// Walk chunks to find "fmt " — it may not always be at offset 12.
	pos := 12
	for pos+8 <= len(wav) {
		chunkID := string(wav[pos : pos+4])
		chunkSize := int(binary.LittleEndian.Uint32(wav[pos+4 : pos+8]))
		if chunkID == "fmt " {
			if pos+8+16 > len(wav) {
				return 0, fmt.Errorf("fmt chunk too short")
			}
			sampleRate := int64(binary.LittleEndian.Uint32(wav[pos+8+4 : pos+8+8]))
			blockAlign := int64(binary.LittleEndian.Uint16(wav[pos+8+12 : pos+8+14]))
			if sampleRate == 0 || blockAlign == 0 {
				return 0, fmt.Errorf("invalid fmt chunk: sampleRate=%d blockAlign=%d", sampleRate, blockAlign)
			}

			// Find data chunk size.
			dataSize, err := findDataChunkSize(wav)
			if err != nil {
				return 0, err
			}

			numSamples := dataSize / blockAlign
			nanos := numSamples * int64(time.Second) / sampleRate
			return time.Duration(nanos), nil
		}
		pos += 8 + chunkSize
		if chunkSize%2 != 0 {
			pos++ // RIFF pad byte
		}
	}
	return 0, fmt.Errorf("fmt chunk not found")
}

func findDataChunkSize(wav []byte) (int64, error) {
	pos := 12
	for pos+8 <= len(wav) {
		chunkID := string(wav[pos : pos+4])
		chunkSize := int64(binary.LittleEndian.Uint32(wav[pos+4 : pos+8]))
		if chunkID == "data" {
			return chunkSize, nil
		}
		pos += 8 + int(chunkSize)
		if chunkSize%2 != 0 {
			pos++
		}
	}
	return 0, fmt.Errorf("data chunk not found")
}

// ---------------------------------------------------------------------------
// RTF threshold gate
// ---------------------------------------------------------------------------

// CheckRTFThreshold returns an error if meanRTF > threshold.
// A threshold of 0 disables the gate.
func CheckRTFThreshold(meanRTF, threshold float64) error {
	if threshold <= 0 {
		return nil
	}
	if meanRTF > threshold {
		return fmt.Errorf("mean RTF %.3f exceeds threshold %.3f", meanRTF, threshold)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Output formatters
// ---------------------------------------------------------------------------

// FormatTable writes a human-readable ASCII table of bench results to w.
func FormatTable(runs []RunResult, stats Stats, w io.Writer) {
	sb := &strings.Builder{}

	fmt.Fprintf(sb, "%-5s  %-5s  %10s  %12s  %8s\n", "Run", "Cold", "MS", "Audio(ms)", "RTF")
	fmt.Fprintln(sb, strings.Repeat("-", 48))

	for _, r := range runs {
		cold := ""
		if r.Cold {
			cold = "yes"
		}
		fmt.Fprintf(sb, "%-5d  %-5s  %10.1f  %12.1f  %8.3f\n",
			r.Index+1,
			cold,
			float64(r.Duration.Milliseconds()),
			float64(r.WAVDuration.Milliseconds()),
			r.RTF,
		)
	}

	fmt.Fprintln(sb, strings.Repeat("-", 48))
	fmt.Fprintf(sb, "%-5s  %-5s  %10.1f  %12s  %8s  (min)\n", "", "", float64(stats.Min.Milliseconds()), "", "")
	fmt.Fprintf(sb, "%-5s  %-5s  %10.1f  %12s  %8s  (mean)\n", "", "", float64(stats.Mean.Milliseconds()), "", "")
	fmt.Fprintf(sb, "%-5s  %-5s  %10.1f  %12s  %8s  (max)\n", "", "", float64(stats.Max.Milliseconds()), "", "")

	fmt.Fprint(w, sb.String())
}

// jsonReport is the top-level JSON structure emitted by FormatJSON.
type jsonReport struct {
	Runs  []jsonRun `json:"runs"`
	Stats jsonStats `json:"stats"`
}

type jsonRun struct {
	Index      int     `json:"index"`
	Cold       bool    `json:"cold"`
	DurationMS float64 `json:"duration_ms"`
	AudioMS    float64 `json:"audio_ms"`
	RTF        float64 `json:"rtf"`
}

type jsonStats struct {
	MinMS  float64 `json:"min_ms"`
	MeanMS float64 `json:"mean_ms"`
	MaxMS  float64 `json:"max_ms"`
}

// FormatJSON writes a JSON report of bench results to w.
func FormatJSON(runs []RunResult, stats Stats, w io.Writer) {
	jr := jsonReport{
		Runs: make([]jsonRun, len(runs)),
		Stats: jsonStats{
			MinMS:  float64(stats.Min.Milliseconds()),
			MeanMS: float64(stats.Mean.Milliseconds()),
			MaxMS:  float64(stats.Max.Milliseconds()),
		},
	}
	for i, r := range runs {
		jr.Runs[i] = jsonRun{
			Index:      r.Index,
			Cold:       r.Cold,
			DurationMS: float64(r.Duration.Milliseconds()),
			AudioMS:    float64(r.WAVDuration.Milliseconds()),
			RTF:        r.RTF,
		}
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(jr)
}
