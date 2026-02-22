//go:build integration

package onnx

import (
	"context"
	"math"
	"testing"

	"github.com/example/go-pocket-tts/internal/tokenizer"
)

// TestGenerateAudioIntegration_ProducesPlausibleAudio runs the full generation
// pipeline against real ONNX models and verifies that the output is non-trivial
// audio of plausible length (~0.5–5 s at 24 kHz), not silence or a constant.
func TestGenerateAudioIntegration_ProducesPlausibleAudio(t *testing.T) {
	libPath := ortLibPath(t)
	tokPath := tokenizerModelPath(t)
	manifestPath := textConditionerManifestPath(t)

	tok, err := tokenizer.NewSentencePieceTokenizer(tokPath)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer: %v", err)
	}

	const inputText = "Hello."
	tokenIDs, err := tok.Encode(inputText)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(tokenIDs) == 0 {
		t.Fatal("tokenizer returned no token IDs")
	}
	t.Logf("tokenized %q → %d tokens: %v", inputText, len(tokenIDs), tokenIDs)

	engine, err := NewEngine(manifestPath, RunnerConfig{
		LibraryPath: libPath,
		APIVersion:  23,
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer engine.Close()

	// Verify all required graphs are present.
	for _, name := range []string{"text_conditioner", "flow_lm_main", "flow_lm_flow", "latent_to_mimi", "mimi_decoder"} {
		if _, ok := engine.Runner(name); !ok {
			t.Skipf("%s graph not present in manifest; skipping", name)
		}
	}

	cfg := GenerateConfig{
		Temperature:    0.7,
		EOSThreshold:   -4.0,
		MaxSteps:       256,
		LSDDecodeSteps: 1,
	}

	pcm, err := engine.GenerateAudio(context.Background(), tokenIDs, cfg)
	if err != nil {
		t.Fatalf("GenerateAudio: %v", err)
	}

	// Plausible length: 0.5–5 seconds at 24 kHz.
	const sampleRate = 24000
	minSamples := sampleRate / 2  // 0.5 s
	maxSamples := sampleRate * 5  // 5.0 s
	t.Logf("generated %d samples (%.2f s)", len(pcm), float64(len(pcm))/sampleRate)

	if len(pcm) < minSamples {
		t.Errorf("too few samples: %d (< %d for 0.5 s)", len(pcm), minSamples)
	}
	if len(pcm) > maxSamples {
		t.Errorf("too many samples: %d (> %d for 5.0 s)", len(pcm), maxSamples)
	}

	// Verify the output is not silence (all zeros) or a constant value.
	var sum, sumSq float64
	for _, s := range pcm {
		v := float64(s)
		sum += v
		sumSq += v * v
	}
	n := float64(len(pcm))
	mean := sum / n
	variance := sumSq/n - mean*mean
	rms := math.Sqrt(variance)
	t.Logf("mean=%.6f rms=%.6f", mean, rms)

	if rms < 1e-6 {
		t.Error("audio appears to be silence or a constant (RMS ≈ 0)")
	}
}
