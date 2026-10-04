package audio

import (
	"encoding/json"
	"math"
	"os"
	"runtime"
	"slices"
	"testing"
)

// resampleGolden is testdata/resample_poly.json, written by
// scripts/dump_resample_golden.py from upstream convert_audio.
type resampleGolden struct {
	Source string `json:"source"`
	Cases  []struct {
		Name     string    `json:"name"`
		FromRate int       `json:"from_rate"`
		ToRate   int       `json:"to_rate"`
		NIn      int       `json:"n_in"`
		Output   []float32 `json:"output"`
	} `json:"cases"`
}

// promptTestSignal mirrors signal() in scripts/dump_resample_golden.py: a
// chirp, a 3.1 kHz tone and deterministic pseudo-noise, rounded to float32.
func promptTestSignal(rate, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		t := float64(i) / float64(rate)
		chirp := 0.5 * math.Sin(2*math.Pi*200*t+math.Pi*40000*t*t)
		tone := 0.25 * math.Sin(2*math.Pi*3100*t)
		noise := 0.1 * (float64(i*7919%1000)/500 - 1)
		out[i] = float32(chirp + tone + noise)
	}

	return out
}

func TestResamplePoly_MatchesScipy(t *testing.T) {
	data, err := os.ReadFile("testdata/resample_poly.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	var golden resampleGolden

	err = json.Unmarshal(data, &golden)
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}

	if len(golden.Cases) == 0 {
		t.Fatal("fixture has no cases")
	}

	// scipy accumulates in float32; this port in float64.
	const tol = 1e-6

	for _, c := range golden.Cases {
		t.Run(c.Name, func(t *testing.T) {
			in := promptTestSignal(c.FromRate, c.NIn)
			orig := slices.Clone(in)

			got := mustResamplePoly(t, in, c.ToRate, c.FromRate)

			if len(got) != len(c.Output) {
				t.Fatalf("len = %d, want %d", len(got), len(c.Output))
			}

			var worst float64

			worstAt := 0

			for i := range got {
				if d := math.Abs(float64(got[i]) - float64(c.Output[i])); d > worst {
					worst, worstAt = d, i
				}
			}

			if worst > tol {
				t.Errorf("max abs diff %.3g at %d (got %v, want %v), tolerance %g",
					worst, worstAt, got[worstAt], c.Output[worstAt], tol)
			}

			if !slices.Equal(in, orig) {
				t.Error("input slice was modified")
			}

			t.Logf("max abs diff %.3g (%s)", worst, golden.Source)
		})
	}
}

func TestResamplePoly_ReducesRatio(t *testing.T) {
	in := promptTestSignal(44100, 441)

	got, want := mustResamplePoly(t, in, 24000, 44100), mustResamplePoly(t, in, 80, 147)
	if !slices.Equal(got, want) {
		t.Error("24000/44100 and 80/147 differ")
	}
}

// TestResamplePoly_RejectsImpracticalRatio: 24000/0xffffffff reduces to
// 1600/286331153, whose filter would need 5.7e9 taps (~43 GiB). It must fail
// fast, before allocating the filter.
// maxWAVRate is the largest rate a WAV header can claim (0xffffffff), capped
// to int on 32-bit targets.
var maxWAVRate = int(min(uint64(math.MaxUint32), uint64(math.MaxInt)))

func TestResamplePoly_RejectsImpracticalRatio(t *testing.T) {
	in := promptTestSignal(48000, 64)

	cases := []struct {
		name     string
		up, down int
	}{
		{"huge down", ExpectedSampleRate, maxWAVRate},
		{"huge up", maxWAVRate, ExpectedSampleRate},
		{"just past the limit", 1, resamplePolyMaxFactor + 1},
		{"zero down", ExpectedSampleRate, 0},
		{"negative up", -1, ExpectedSampleRate},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var before, after runtime.MemStats

			runtime.ReadMemStats(&before)

			out, err := ResamplePoly(in, c.up, c.down)

			runtime.ReadMemStats(&after)

			if err == nil {
				t.Fatalf("ResamplePoly(%d, %d) = %d samples, want error", c.up, c.down, len(out))
			}

			if alloc := after.TotalAlloc - before.TotalAlloc; alloc > 1<<20 {
				t.Errorf("allocated %d bytes before failing, want < 1 MiB", alloc)
			}
		})
	}
}

func TestResamplePoly_LimitFactorWorks(t *testing.T) {
	out, err := ResamplePoly(promptTestSignal(48000, 64), 1, resamplePolyMaxFactor)
	if err != nil {
		t.Fatalf("ResamplePoly(1, %d): %v", resamplePolyMaxFactor, err)
	}

	if len(out) != 1 {
		t.Errorf("len = %d, want 1", len(out))
	}
}

// TestResamplePoly_LargeProductsFitInt resamples a 30 s prompt at 24001 Hz:
// len·up and the zero-stuffed positions reach about 1.7e10, past int32, so
// this fails on 32-bit targets unless they are computed in 64 bits.
func TestResamplePoly_LargeProductsFitInt(t *testing.T) {
	const rate = ExpectedSampleRate + 1

	out, err := ResamplePoly(promptTestSignal(rate, VoicePromptMaxSec*rate), ExpectedSampleRate, rate)
	if err != nil {
		t.Fatalf("ResamplePoly: %v", err)
	}

	// ceil(720030 · 24000 / 24001) = 720000.
	if want := VoicePromptMaxSec * ExpectedSampleRate; len(out) != want {
		t.Fatalf("len = %d, want %d", len(out), want)
	}

	for i, v := range out {
		if math.IsNaN(float64(v)) || math.Abs(float64(v)) > 2 {
			t.Fatalf("out[%d] = %g", i, v)
		}
	}
}

func TestResamplePoly_EmptyInput(t *testing.T) {
	out, err := ResamplePoly(nil, ExpectedSampleRate, 44100)
	if err != nil || len(out) != 0 {
		t.Errorf("ResamplePoly(nil) = %d samples, %v; want 0 samples, nil", len(out), err)
	}
}

func mustResamplePoly(t *testing.T, samples []float32, up, down int) []float32 {
	t.Helper()

	out, err := ResamplePoly(samples, up, down)
	if err != nil {
		t.Fatalf("ResamplePoly(%d, %d): %v", up, down, err)
	}

	return out
}
