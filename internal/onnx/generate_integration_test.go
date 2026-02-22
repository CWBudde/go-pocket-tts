//go:build integration

package onnx

import (
	"context"
	"encoding/binary"
	"math"
	"math/rand"
	"strconv"
	"os"
	"path/filepath"
	"testing"

	"github.com/example/go-pocket-tts/internal/safetensors"
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
	minSamples := sampleRate / 2 // 0.5 s
	maxSamples := sampleRate * 5 // 5.0 s
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

// ---------------------------------------------------------------------------
// Task 19.4: integration test for voice conditioning
// ---------------------------------------------------------------------------

// writeSyntheticVoiceSafetensors creates a temporary .safetensors file with a
// synthetic voice embedding of shape [T, 1024] filled with random values.
// This is used to exercise the voice conditioning path without real voice data.
func writeSyntheticVoiceSafetensors(t *testing.T, T int) string {
	t.Helper()

	const D = 1024
	vals := make([]float32, T*D)
	r := rand.New(rand.NewSource(42))
	for i := range vals {
		vals[i] = r.Float32()*2 - 1 // [-1, 1)
	}

	// Build safetensors binary: 8-byte LE header length → JSON header → raw data.
	rawData := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(rawData[i*4:], math.Float32bits(v))
	}

	// Inline JSON header for a 2D [T, 1024] tensor.
	headerJSON := []byte(`{"voice":{"dtype":"F32","shape":[` +
		strconv.Itoa(T) + `,1024],"data_offsets":[0,` + strconv.Itoa(len(rawData)) + `]}}`)

	lenBuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(lenBuf, uint64(len(headerJSON)))

	var buf []byte
	buf = append(buf, lenBuf...)
	buf = append(buf, headerJSON...)
	buf = append(buf, rawData...)

	path := filepath.Join(t.TempDir(), "voice.safetensors")
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write synthetic voice file: %v", err)
	}
	return path
}

// TestGenerateAudioIntegration_VoiceConditioningDiffersFromUnvoiced verifies
// that synthesizing the same text with and without a voice embedding produces
// different audio output. The voice embedding conditions the generation, so
// the distributions should differ.
func TestGenerateAudioIntegration_VoiceConditioningDiffersFromUnvoiced(t *testing.T) {
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

	baseCfg := GenerateConfig{
		Temperature:    0.0, // deterministic: same seed → same output without voice
		EOSThreshold:   -4.0,
		MaxSteps:       64,
		LSDDecodeSteps: 1,
	}

	// Generate without voice conditioning.
	pcmNoVoice, err := engine.GenerateAudio(context.Background(), tokenIDs, baseCfg)
	if err != nil {
		t.Fatalf("GenerateAudio (no voice): %v", err)
	}
	t.Logf("no-voice: %d samples", len(pcmNoVoice))

	// Build voice embedding from a synthetic safetensors file (5 voice frames).
	voicePath := writeSyntheticVoiceSafetensors(t, 5)
	voiceData, voiceShape, err := safetensors.LoadVoiceEmbedding(voicePath)
	if err != nil {
		t.Fatalf("LoadVoiceEmbedding: %v", err)
	}
	voiceEmb, err := NewTensor(voiceData, voiceShape)
	if err != nil {
		t.Fatalf("NewTensor voice: %v", err)
	}

	voiceCfg := baseCfg
	voiceCfg.VoiceEmbedding = voiceEmb

	// Generate with voice conditioning.
	pcmWithVoice, err := engine.GenerateAudio(context.Background(), tokenIDs, voiceCfg)
	if err != nil {
		t.Fatalf("GenerateAudio (with voice): %v", err)
	}
	t.Logf("with-voice: %d samples", len(pcmWithVoice))

	// Both outputs should be non-empty.
	if len(pcmNoVoice) == 0 {
		t.Fatal("no-voice output is empty")
	}
	if len(pcmWithVoice) == 0 {
		t.Fatal("with-voice output is empty")
	}

	// Outputs should differ — voice conditioning changes the generation.
	// Note: the synthetic voice embedding is out-of-distribution (random values,
	// not a real mimi_encoder output). This test validates that the code path is
	// exercised and the conditioning has *some* effect on generation, not that
	// the voice transfer is semantically meaningful.
	// Compare RMS difference over the overlap region.
	n := len(pcmNoVoice)
	if len(pcmWithVoice) < n {
		n = len(pcmWithVoice)
	}
	var diffSumSq float64
	for i := range n {
		d := float64(pcmNoVoice[i]) - float64(pcmWithVoice[i])
		diffSumSq += d * d
	}
	diffRMS := math.Sqrt(diffSumSq / float64(n))
	t.Logf("diff RMS between voiced and unvoiced: %.6f", diffRMS)

	if diffRMS < 1e-6 {
		t.Error("voice-conditioned and unconditioned outputs are identical; voice embedding had no effect")
	}
}
