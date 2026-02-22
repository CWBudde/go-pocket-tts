package tts

import (
	"fmt"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/onnx"
	"github.com/example/go-pocket-tts/internal/text"
	"github.com/example/go-pocket-tts/internal/tokenizer"
)

// Service orchestrates text preprocessing and ONNX inference.
type Service struct {
	engine       *onnx.Engine
	preprocessor *text.Preprocessor
	tokenizer    tokenizer.Tokenizer
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
		engine:       engine,
		preprocessor: text.NewPreprocessor(),
		tokenizer:    tok,
	}, nil
}

// Synthesize converts text to audio samples via ONNX inference.
func (s *Service) Synthesize(input string) ([]float32, error) {
	tokens := s.preprocessor.Preprocess(input)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens produced from input")
	}
	return s.engine.Infer(tokens)
}

// Close releases engine resources.
func (s *Service) Close() {
	if s.engine != nil {
		s.engine.Close()
	}
}
