package tts

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/example/go-pocket-tts/internal/config"
)

func TestSynthesizeStream_SendsChunksAndCloses(t *testing.T) {
	rt := &captureRuntime{audio: []float32{0.1, 0.2}}
	svc := &Service{
		runtime:   rt,
		tokenizer: fakeTokenizer{},
		ttsCfg:    config.DefaultConfig().TTS,
	}

	ch := make(chan PCMChunk, 4)

	err := svc.SynthesizeStream(context.Background(), "hello", "", ch)
	if err != nil {
		t.Fatalf("SynthesizeStream error: %v", err)
	}

	// Channel should be closed; collect chunks
	var chunks []PCMChunk
	for c := range ch {
		chunks = append(chunks, c)
	}

	if len(chunks) == 0 {
		t.Fatal("no chunks received")
	}

	if !chunks[len(chunks)-1].Final {
		t.Error("last chunk should have Final=true")
	}

	// Verify audio matches what runtime returns
	for _, c := range chunks {
		if len(c.Samples) != 2 {
			t.Errorf("chunk %d: got %d samples; want 2", c.ChunkIndex, len(c.Samples))
		}
	}
}

func TestSynthesizeStream_ContextCancellation(t *testing.T) {
	// Runtime that blocks until context is cancelled
	rt := &slowRuntime{delay: 5 * time.Second, audio: []float32{0.1}}
	svc := &Service{
		runtime:   rt,
		tokenizer: fakeTokenizer{},
		ttsCfg:    config.DefaultConfig().TTS,
	}

	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan PCMChunk, 4)

	done := make(chan error, 1)

	go func() {
		done <- svc.SynthesizeStream(ctx, "hello", "", ch)
	}()

	// Cancel quickly
	time.Sleep(10 * time.Millisecond)
	cancel()

	err := <-done
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestSynthesizeStream_NilRuntime(t *testing.T) {
	svc := &Service{
		runtime:   nil,
		tokenizer: fakeTokenizer{},
		ttsCfg:    config.DefaultConfig().TTS,
	}

	ch := make(chan PCMChunk, 4)

	err := svc.SynthesizeStream(context.Background(), "hello", "", ch)
	if err == nil {
		t.Fatal("expected error for nil runtime")
	}

	// Channel should be closed even on error
	select {
	case _, ok := <-ch:
		if ok {
			t.Error("channel should be closed after error")
		}
	default:
		t.Error("channel should be closed after error")
	}
}

func TestSynthesizeStream_MultipleChunks(t *testing.T) {
	rt := &captureRuntime{audio: []float32{0.5}}
	svc := &Service{
		runtime:   rt,
		tokenizer: wordCountTokenizer{},
		ttsCfg:    config.DefaultConfig().TTS,
	}

	// wordCountTokenizer creates one token per word, maxTokensPerChunk=50.
	// PrepareChunks splits on sentence boundaries, so use periods.
	longSentence := ""

	var longSentenceSb108 strings.Builder
	for range 30 {
		longSentenceSb108.WriteString("word ")
	}

	longSentence += longSentenceSb108.String()

	longText := longSentence + ". " + longSentence + "."

	ch := make(chan PCMChunk, 10)

	err := svc.SynthesizeStream(context.Background(), longText, "", ch)
	if err != nil {
		t.Fatalf("SynthesizeStream error: %v", err)
	}

	var chunks []PCMChunk
	for c := range ch {
		chunks = append(chunks, c)
	}

	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks for long text; got %d", len(chunks))
	}

	// Only the last chunk should be Final
	for i, c := range chunks {
		if i < len(chunks)-1 && c.Final {
			t.Errorf("chunk %d should not be Final", i)
		}
	}

	if !chunks[len(chunks)-1].Final {
		t.Error("last chunk should be Final")
	}
}

// slowRuntime simulates a slow generation that respects context cancellation.
type slowRuntime struct {
	delay time.Duration
	audio []float32
}

func (s *slowRuntime) GenerateAudio(ctx context.Context, _ []int64, _ RuntimeGenerateConfig) ([]float32, error) {
	select {
	case <-time.After(s.delay):
		return s.audio, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *slowRuntime) Close() {}
