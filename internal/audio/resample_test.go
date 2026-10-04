package audio

import (
	"encoding/json"
	"math"
	"os"
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

			got := ResamplePoly(in, c.ToRate, c.FromRate)

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

	if got, want := ResamplePoly(in, 24000, 44100), ResamplePoly(in, 80, 147); !slices.Equal(got, want) {
		t.Error("24000/44100 and 80/147 differ")
	}
}
