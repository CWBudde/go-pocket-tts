package onnx

import (
	"fmt"
	"math"

	"github.com/example/go-pocket-tts/internal/config"
)

type Engine struct {
	threads int
}

func NewEngine(cfg config.RuntimeConfig) (*Engine, error) {
	if cfg.Threads < 1 {
		return nil, fmt.Errorf("runtime threads must be >= 1")
	}
	if _, err := Bootstrap(cfg); err != nil {
		return nil, fmt.Errorf("bootstrap onnx runtime: %w", err)
	}
	return &Engine{threads: cfg.Threads}, nil
}

func (e *Engine) Infer(tokens []int) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty token list")
	}

	sampleCount := len(tokens) * 512
	out := make([]float32, sampleCount)
	for i := range out {
		t := float64(i) / 22050.0
		out[i] = float32(0.2 * math.Sin(2*math.Pi*220*t))
	}
	return out, nil
}
