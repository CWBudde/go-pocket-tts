package bench_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/example/go-pocket-tts/internal/audio"
	"github.com/example/go-pocket-tts/internal/bench"
)

// ---------------------------------------------------------------------------
// Aggregation (Task 9.3 — min/max/mean)
// ---------------------------------------------------------------------------

func TestStats_MinMaxMean(t *testing.T) {
	durations := []time.Duration{
		100 * time.Millisecond,
		200 * time.Millisecond,
		300 * time.Millisecond,
	}
	s := bench.ComputeStats(durations)

	if s.Min != 100*time.Millisecond {
		t.Errorf("want min=100ms, got %v", s.Min)
	}

	if s.Max != 300*time.Millisecond {
		t.Errorf("want max=300ms, got %v", s.Max)
	}

	if s.Mean != 200*time.Millisecond {
		t.Errorf("want mean=200ms, got %v", s.Mean)
	}
}

func TestStats_SingleRun(t *testing.T) {
	s := bench.ComputeStats([]time.Duration{150 * time.Millisecond})
	if s.Min != s.Max || s.Min != s.Mean {
		t.Errorf("single run: min/max/mean should all be equal, got min=%v max=%v mean=%v", s.Min, s.Max, s.Mean)
	}
}

// ---------------------------------------------------------------------------
// RTF calculation (Task 9.2)
// ---------------------------------------------------------------------------

func TestRTF_Calculation(t *testing.T) {
	// 1 second of audio synthesised in 500ms → RTF = 0.5
	synthDur := 500 * time.Millisecond
	audioDur := 1 * time.Second

	rtf := bench.CalcRTF(synthDur, audioDur)
	if rtf < 0.499 || rtf > 0.501 {
		t.Errorf("want RTF≈0.5, got %.4f", rtf)
	}
}

func TestRTF_ZeroAudioDuration(t *testing.T) {
	rtf := bench.CalcRTF(500*time.Millisecond, 0)
	if rtf != 0 {
		t.Errorf("want RTF=0 for zero audio duration, got %.4f", rtf)
	}
}

func TestAudioDurationFromWAV(t *testing.T) {
	// 24000 samples at 24 kHz = exactly 1 second
	samples := make([]float32, 24000)

	wav, err := audio.EncodeWAV(samples)
	if err != nil {
		t.Fatalf("EncodeWAV: %v", err)
	}

	dur, err := bench.WAVDuration(wav)
	if err != nil {
		t.Fatalf("WAVDuration: %v", err)
	}
	const want = time.Second

	diff := dur - want
	if diff < 0 {
		diff = -diff
	}

	if diff > time.Millisecond {
		t.Errorf("want 1s audio duration, got %v", dur)
	}
}

// ---------------------------------------------------------------------------
// RTF threshold gate (Task 9.3)
// ---------------------------------------------------------------------------

func TestRTFThreshold_ExceedsThreshold(t *testing.T) {
	// Mean RTF = 1.5, threshold = 1.0 → should fail
	err := bench.CheckRTFThreshold(1.5, 1.0)
	if err == nil {
		t.Error("want error when mean RTF exceeds threshold")
	}
}

func TestRTFThreshold_BelowThreshold(t *testing.T) {
	err := bench.CheckRTFThreshold(0.8, 1.0)
	if err != nil {
		t.Errorf("want no error when RTF below threshold, got: %v", err)
	}
}

func TestRTFThreshold_ExactlyAtThreshold(t *testing.T) {
	err := bench.CheckRTFThreshold(1.0, 1.0)
	if err != nil {
		t.Errorf("want no error at exact threshold, got: %v", err)
	}
}

func TestRTFThreshold_DisabledWhenZero(t *testing.T) {
	err := bench.CheckRTFThreshold(9999, 0)
	if err != nil {
		t.Errorf("threshold=0 should disable gate, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

func TestFormatTable_ContainsHeaders(t *testing.T) {
	runs := []bench.RunResult{
		{Index: 0, Cold: true, Duration: 800 * time.Millisecond, RTF: 0.8, WAVDuration: time.Second},
		{Index: 1, Cold: false, Duration: 500 * time.Millisecond, RTF: 0.5, WAVDuration: time.Second},
	}
	stats := bench.ComputeStats([]time.Duration{800 * time.Millisecond, 500 * time.Millisecond})

	var buf strings.Builder
	bench.FormatTable(runs, stats, &buf)
	out := buf.String()

	for _, want := range []string{"run", "cold", "ms", "rtf"} {
		if !strings.Contains(strings.ToLower(out), want) {
			t.Errorf("table output missing %q:\n%s", want, out)
		}
	}
}

func TestFormatJSON_IsValidJSON(t *testing.T) {
	runs := []bench.RunResult{
		{Index: 0, Cold: true, Duration: 800 * time.Millisecond, RTF: 0.8, WAVDuration: time.Second},
	}
	stats := bench.ComputeStats([]time.Duration{800 * time.Millisecond})

	var buf bytes.Buffer
	bench.FormatJSON(runs, stats, &buf)

	var out any

	err := json.Unmarshal(buf.Bytes(), &out)
	if err != nil {
		t.Errorf("FormatJSON produced invalid JSON: %v\n%s", err, buf.String())
	}
}
