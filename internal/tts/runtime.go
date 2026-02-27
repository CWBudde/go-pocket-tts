package tts

import (
	"context"
)

// VoiceEmbedding is a runtime-neutral voice conditioning tensor payload.
// Shape is expected to be [1, T, D] when present.
type VoiceEmbedding struct {
	Data  []float32
	Shape []int64
}

// RuntimeGenerateConfig controls a single chunk generation call.
type RuntimeGenerateConfig struct {
	Temperature    float64
	EOSThreshold   float64
	MaxSteps       int
	LSDDecodeSteps int
	FramesAfterEOS int
	VoiceEmbedding *VoiceEmbedding
	// StepCallback is called after each AR step with the 1-based step index
	// and the configured maxSteps ceiling. It may be nil.
	StepCallback func(step, maxSteps int)
}

// PCMChunk is a chunk of PCM audio produced during streaming synthesis.
type PCMChunk struct {
	Samples    []float32 // PCM float32 samples at 24 kHz
	ChunkIndex int       // 0-based index of the text chunk that produced this
	Final      bool      // true if this is the last chunk
}

// Runtime abstracts TTS graph execution so multiple native runtimes can share
// the same service pipeline (tokenization/chunking/voice conditioning).
type Runtime interface {
	GenerateAudio(ctx context.Context, tokens []int64, cfg RuntimeGenerateConfig) ([]float32, error)
	Close()
}
