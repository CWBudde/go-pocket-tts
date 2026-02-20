package tts

import (
	"fmt"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/onnx"
	"github.com/example/go-pocket-tts/internal/text"
)

type Service struct {
	engine       *onnx.Engine
	preprocessor *text.Preprocessor
}

func NewService(cfg config.Config) (*Service, error) {
	engine, err := onnx.NewEngine(cfg.Runtime)
	if err != nil {
		return nil, err
	}
	return &Service{
		engine:       engine,
		preprocessor: text.NewPreprocessor(),
	}, nil
}

func (s *Service) Synthesize(input string) ([]float32, error) {
	tokens := s.preprocessor.Preprocess(input)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens produced from input")
	}
	return s.engine.Infer(tokens)
}
