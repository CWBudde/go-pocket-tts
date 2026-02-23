//go:build integration

package tts

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/safetensors"
)

func TestSynthesizeNativeSafetensors_ShortMediumChunked(t *testing.T) {
	modelPath, tokPath := requireNativeSafetensorsAssets(t)

	cfg := config.DefaultConfig()
	cfg.TTS.Backend = config.BackendNativeSafetensors
	cfg.Paths.ModelPath = modelPath
	cfg.Paths.TokenizerModel = tokPath
	cfg.TTS.MaxSteps = 24

	svc, err := NewService(cfg)
	if err != nil {
		t.Fatalf("NewService(native-safetensors): %v", err)
	}
	defer svc.Close()

	shortPCM, err := svc.Synthesize("Hello.", "")
	if err != nil {
		t.Fatalf("Synthesize short: %v", err)
	}
	if len(shortPCM) == 0 {
		t.Fatal("short synthesis produced empty PCM")
	}

	mediumPCM, err := svc.Synthesize("Hello world. This is a test.", "")
	if err != nil {
		t.Fatalf("Synthesize medium: %v", err)
	}
	if len(mediumPCM) == 0 {
		t.Fatal("medium synthesis produced empty PCM")
	}
	if hashPCM32(mediumPCM) == hashPCM32(shortPCM) {
		t.Fatal("medium synthesis hash unexpectedly equals short synthesis hash")
	}

	chunkedInput := "This is sentence one. This is sentence two with more words. This is sentence three to trigger chunking behavior."
	chunkedPCM, err := svc.Synthesize(chunkedInput, "")
	if err != nil {
		t.Fatalf("Synthesize chunked: %v", err)
	}
	if len(chunkedPCM) == 0 {
		t.Fatal("chunked synthesis produced empty PCM")
	}
	if hashPCM32(chunkedPCM) == hashPCM32(shortPCM) {
		t.Fatal("chunked synthesis hash unexpectedly equals short synthesis hash")
	}
}

func TestSynthesizeNativeSafetensors_VoiceConditioningDiffers(t *testing.T) {
	modelPath, tokPath := requireNativeSafetensorsAssets(t)

	cfg := config.DefaultConfig()
	cfg.TTS.Backend = config.BackendNativeSafetensors
	cfg.Paths.ModelPath = modelPath
	cfg.Paths.TokenizerModel = tokPath
	cfg.TTS.MaxSteps = 24

	svc, err := NewService(cfg)
	if err != nil {
		t.Fatalf("NewService(native-safetensors): %v", err)
	}
	defer svc.Close()

	voicePath := filepath.Join(t.TempDir(), "voice.safetensors")
	voiceVals := make([]float32, 1*5*1024)
	for i := range voiceVals {
		voiceVals[i] = float32(math.Sin(float64(i) * 0.01))
	}
	if err := safetensors.WriteFile(voicePath, []safetensors.Tensor{{
		Name:  "voice",
		Shape: []int64{1, 5, 1024},
		Data:  voiceVals,
	}}); err != nil {
		t.Fatalf("write voice safetensors: %v", err)
	}

	input := "Hello world."
	plain, err := svc.Synthesize(input, "")
	if err != nil {
		t.Fatalf("Synthesize without voice: %v", err)
	}
	withVoice, err := svc.Synthesize(input, voicePath)
	if err != nil {
		t.Fatalf("Synthesize with voice: %v", err)
	}
	if len(plain) == 0 || len(withVoice) == 0 {
		t.Fatalf("expected non-empty outputs, got plain=%d withVoice=%d", len(plain), len(withVoice))
	}

	hPlain := hashPCM32(plain)
	hVoice := hashPCM32(withVoice)
	if hPlain == hVoice {
		t.Fatalf("voice-conditioned output hash equals unvoiced hash (%s), expected divergence", hPlain)
	}
}

func requireNativeSafetensorsAssets(t testing.TB) (modelPath, tokPath string) {
	t.Helper()
	modelCandidates := []string{
		filepath.Join("models", "tts_b6369a24.safetensors"),
		filepath.Join("..", "..", "models", "tts_b6369a24.safetensors"),
	}
	tokCandidates := []string{
		filepath.Join("models", "tokenizer.model"),
		filepath.Join("..", "..", "models", "tokenizer.model"),
	}

	for _, p := range modelCandidates {
		if _, err := os.Stat(p); err == nil {
			modelPath = p
			break
		}
	}
	for _, p := range tokCandidates {
		if _, err := os.Stat(p); err == nil {
			tokPath = p
			break
		}
	}
	if modelPath == "" || tokPath == "" {
		t.Skipf("native safetensors assets unavailable (model=%q tokenizer=%q)", modelPath, tokPath)
	}
	return modelPath, tokPath
}

func hashPCM32(samples []float32) string {
	h := sha256.New()
	var b [4]byte
	for _, s := range samples {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(s))
		_, _ = h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}
