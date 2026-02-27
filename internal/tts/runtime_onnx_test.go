package tts

import (
	"context"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/onnx"
)

type closeSpyGraphRunner struct {
	closed bool
}

func (c *closeSpyGraphRunner) Run(context.Context, map[string]*onnx.Tensor) (map[string]*onnx.Tensor, error) {
	return map[string]*onnx.Tensor{}, nil
}

func (c *closeSpyGraphRunner) Name() string { return "close-spy" }

func (c *closeSpyGraphRunner) Close() { c.closed = true }

func TestNewONNXRuntime(t *testing.T) {
	engine := onnx.NewEngineWithRunners(map[string]onnx.GraphRunner{})

	rt := newONNXRuntime(engine)
	if rt == nil {
		t.Fatal("newONNXRuntime returned nil")
	}

	if _, ok := rt.(*onnxRuntime); !ok {
		t.Fatalf("runtime type = %T, want *onnxRuntime", rt)
	}
}

func TestONNXRuntimeGenerateAudio(t *testing.T) {
	t.Run("invalid voice embedding shape is rejected", func(t *testing.T) {
		rt := &onnxRuntime{
			engine: onnx.NewEngineWithRunners(map[string]onnx.GraphRunner{}),
		}

		_, err := rt.GenerateAudio(context.Background(), []int64{1}, RuntimeGenerateConfig{
			VoiceEmbedding: &VoiceEmbedding{
				Data:  []float32{1, 2, 3},
				Shape: []int64{1, 2, 2}, // shape expects 4 values, data has 3.
			},
		})
		if err == nil || !strings.Contains(err.Error(), "build voice tensor") {
			t.Fatalf("expected voice tensor build error, got: %v", err)
		}
	})

	t.Run("engine error is propagated", func(t *testing.T) {
		rt := &onnxRuntime{
			engine: onnx.NewEngineWithRunners(map[string]onnx.GraphRunner{}),
		}

		_, err := rt.GenerateAudio(context.Background(), []int64{1, 2, 3}, RuntimeGenerateConfig{})
		if err == nil || !strings.Contains(err.Error(), "text_conditioner graph not found") {
			t.Fatalf("expected text_conditioner graph error, got: %v", err)
		}
	})
}

func TestONNXRuntimeClose(t *testing.T) {
	rt := &onnxRuntime{}
	rt.Close() // nil engine should be a no-op

	spy := &closeSpyGraphRunner{}
	engine := onnx.NewEngineWithRunners(map[string]onnx.GraphRunner{
		"text_conditioner": spy,
	})

	rt = &onnxRuntime{engine: engine}
	rt.Close()

	if !spy.closed {
		t.Fatal("expected runner Close() to be called")
	}
}
