package tts

import (
	"context"

	"github.com/cwbudde/go-pocket-tts/internal/nativert"
)

// VoiceEmbedding is a runtime-neutral voice conditioning tensor payload.
// Shape is expected to be [1, T, D] when present.
type VoiceEmbedding = nativert.VoiceEmbedding

// RuntimeGenerateConfig controls a single chunk generation call.
type RuntimeGenerateConfig = nativert.Config

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
