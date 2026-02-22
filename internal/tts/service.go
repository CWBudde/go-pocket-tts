package tts

import (
	"context"
	"fmt"
	"strings"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/onnx"
	"github.com/example/go-pocket-tts/internal/safetensors"
	"github.com/example/go-pocket-tts/internal/text"
	"github.com/example/go-pocket-tts/internal/tokenizer"
)

// maxTokensPerChunk is the SentencePiece token budget per synthesis chunk,
// matching the reference implementation's 50-token limit.
const maxTokensPerChunk = 50

// Service orchestrates text preprocessing and ONNX inference.
type Service struct {
	engine    *onnx.Engine
	tokenizer tokenizer.Tokenizer
	ttsCfg    config.TTSConfig
}

// NewService initializes the TTS service with ONNX runners loaded from the manifest.
func NewService(cfg config.Config) (*Service, error) {
	tok, err := tokenizer.NewSentencePieceTokenizer(cfg.Paths.TokenizerModel)
	if err != nil {
		return nil, fmt.Errorf("init tokenizer: %w", err)
	}

	rcfg := onnx.RunnerConfig{
		LibraryPath: cfg.Runtime.ORTLibraryPath,
		APIVersion:  23,
	}
	if rcfg.LibraryPath == "" {
		info, err := onnx.DetectRuntime(cfg.Runtime)
		if err != nil {
			return nil, fmt.Errorf("detect ORT runtime: %w", err)
		}
		rcfg.LibraryPath = info.LibraryPath
	}

	engine, err := onnx.NewEngine(cfg.Paths.ONNXManifest, rcfg)
	if err != nil {
		return nil, fmt.Errorf("init onnx engine: %w", err)
	}
	return &Service{
		engine:    engine,
		tokenizer: tok,
		ttsCfg:    cfg.TTS,
	}, nil
}

// generateConfig builds a GenerateConfig from the stored TTS config,
// overriding FramesAfterEOS per chunk.
func (s *Service) generateConfig(framesAfterEOS int) onnx.GenerateConfig {
	return onnx.GenerateConfig{
		Temperature:    s.ttsCfg.Temperature,
		EOSThreshold:   s.ttsCfg.EOSThreshold,
		MaxSteps:       s.ttsCfg.MaxSteps,
		LSDDecodeSteps: s.ttsCfg.LSDDecodeSteps,
		FramesAfterEOS: framesAfterEOS,
	}
}

// Synthesize converts text to audio samples via ONNX inference.
// Text is preprocessed and split into ≤50-token chunks per the reference
// implementation. Each chunk is generated independently and the resulting
// PCM audio is concatenated.
//
// If voicePath is non-empty, it should point to a .safetensors file containing
// a voice embedding. The embedding is loaded and prepended to the text
// conditioning for each chunk, enabling voice cloning.
func (s *Service) Synthesize(input string, voicePath string) ([]float32, error) {
	chunks, err := text.PrepareChunks(input, s.tokenizer, maxTokensPerChunk)
	if err != nil {
		return nil, fmt.Errorf("no tokens produced from input: %w", err)
	}

	// Load voice embedding if a path was provided.
	var voiceEmb *onnx.Tensor
	if strings.TrimSpace(voicePath) != "" {
		data, shape, err := safetensors.LoadVoiceEmbedding(voicePath)
		if err != nil {
			return nil, fmt.Errorf("load voice embedding: %w", err)
		}
		voiceEmb, err = onnx.NewTensor(data, shape)
		if err != nil {
			return nil, fmt.Errorf("build voice tensor: %w", err)
		}
	}

	ctx := context.Background()
	var allAudio []float32

	for i, chunk := range chunks {
		cfg := s.generateConfig(chunk.FramesAfterEOS())
		cfg.VoiceEmbedding = voiceEmb

		pcm, err := s.engine.GenerateAudio(ctx, chunk.TokenIDs, cfg)
		if err != nil {
			return nil, fmt.Errorf("chunk %d: %w", i, err)
		}
		allAudio = append(allAudio, pcm...)
	}

	return allAudio, nil
}

// Close releases engine resources.
func (s *Service) Close() {
	if s.engine != nil {
		s.engine.Close()
	}
}
