package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/example/go-pocket-tts/internal/audio"
)

// makeMinimalWAV returns a valid WAV byte slice with a few samples.
func makeMinimalWAV(t *testing.T) []byte {
	t.Helper()

	data, err := audio.EncodeWAV([]float32{0.1, 0.2, 0.3, 0.4})
	if err != nil {
		t.Fatalf("EncodeWAV: %v", err)
	}

	return data
}

// writeFakeTTSScript writes a shell script that outputs a minimal WAV to stdout.
// The script ignores all arguments, writes the WAV bytes, and exits 0.
func writeFakeTTSScript(t *testing.T, wavData []byte) string {
	t.Helper()
	tmp := t.TempDir()
	script := filepath.Join(tmp, "pocket-tts")

	// Write the WAV bytes as a hex dump that the script will decode via printf.
	// Simpler: write a Go binary that just writes the file would need build; use
	// a script that cat's a pre-written WAV file instead.
	wavFile := filepath.Join(tmp, "out.wav")

	err := os.WriteFile(wavFile, wavData, 0o644)
	if err != nil {
		t.Fatalf("WriteFile wav: %v", err)
	}

	scriptContent := "#!/bin/sh\ncat " + wavFile + "\n"

	err = os.WriteFile(script, []byte(scriptContent), 0o755)
	if err != nil {
		t.Fatalf("WriteFile script: %v", err)
	}

	return script
}

func TestRunBench_SingleRun(t *testing.T) {
	wavData := makeMinimalWAV(t)
	exe := writeFakeTTSScript(t, wavData)

	results, err := runBench(context.Background(), benchOptions{
		ExecutablePath: exe,
		Text:           "hello world",
		Runs:           1,
	})
	if err != nil {
		t.Fatalf("runBench: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	if !results[0].Cold {
		t.Error("first run should be marked Cold")
	}

	if results[0].Duration <= 0 {
		t.Error("expected positive duration")
	}
}

func TestRunBench_MultipleRuns(t *testing.T) {
	wavData := makeMinimalWAV(t)
	exe := writeFakeTTSScript(t, wavData)

	results, err := runBench(context.Background(), benchOptions{
		ExecutablePath: exe,
		Text:           "hello world",
		Runs:           3,
	})
	if err != nil {
		t.Fatalf("runBench: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	// Only the first run is cold.
	for i, r := range results {
		if r.Cold != (i == 0) {
			t.Errorf("run %d: Cold=%v, want %v", i, r.Cold, i == 0)
		}
	}
}

func TestRunBench_SynthesisFailure(t *testing.T) {
	tmp := t.TempDir()

	script := filepath.Join(tmp, "pocket-tts")

	writeErr := os.WriteFile(script, []byte("#!/bin/sh\nexit 1\n"), 0o755)
	if writeErr != nil {
		t.Fatalf("WriteFile: %v", writeErr)
	}

	_, err := runBench(context.Background(), benchOptions{
		ExecutablePath: script,
		Text:           "hello world",
		Runs:           1,
	})
	if err == nil {
		t.Fatal("expected error from failed synthesis")
	}
}

func TestRunBench_WAVDurationCalculated(t *testing.T) {
	// 24000 samples at 24kHz = 1 second of audio.
	samples := make([]float32, 24000)
	for i := range samples {
		samples[i] = 0.1
	}

	wavData, err := audio.EncodeWAV(samples)
	if err != nil {
		t.Fatalf("EncodeWAV: %v", err)
	}

	exe := writeFakeTTSScript(t, wavData)

	results, err := runBench(context.Background(), benchOptions{
		ExecutablePath: exe,
		Text:           "hello",
		Runs:           1,
	})
	if err != nil {
		t.Fatalf("runBench: %v", err)
	}

	if results[0].WAVDuration <= 0 {
		t.Errorf("expected positive WAVDuration, got %v", results[0].WAVDuration)
	}
}
