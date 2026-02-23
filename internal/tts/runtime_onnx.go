package tts

import (
	"context"
	"fmt"

	"github.com/example/go-pocket-tts/internal/onnx"
)

type onnxRuntime struct {
	engine *onnx.Engine
}

func newONNXRuntime(engine *onnx.Engine) Runtime {
	return &onnxRuntime{engine: engine}
}

func (r *onnxRuntime) GenerateAudio(ctx context.Context, tokens []int64, cfg RuntimeGenerateConfig) ([]float32, error) {
	genCfg := onnx.GenerateConfig{
		Temperature:    cfg.Temperature,
		EOSThreshold:   cfg.EOSThreshold,
		MaxSteps:       cfg.MaxSteps,
		LSDDecodeSteps: cfg.LSDDecodeSteps,
		FramesAfterEOS: cfg.FramesAfterEOS,
	}
	if cfg.VoiceEmbedding != nil {
		voiceTensor, err := onnx.NewTensor(cfg.VoiceEmbedding.Data, cfg.VoiceEmbedding.Shape)
		if err != nil {
			return nil, fmt.Errorf("build voice tensor: %w", err)
		}
		genCfg.VoiceEmbedding = voiceTensor
	}
	return r.engine.GenerateAudio(ctx, tokens, genCfg)
}

func (r *onnxRuntime) Close() {
	if r.engine != nil {
		r.engine.Close()
	}
}
