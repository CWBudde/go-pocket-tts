package onnx

import (
	"context"
	"testing"
)

// ---------------------------------------------------------------------------
// Unit tests for Engine.TextConditioner (no ORT dependency)
// ---------------------------------------------------------------------------

// fakeRunner implements the runnerIface used internally by Engine.
// It lets us inject mock outputs without an ORT session.
type fakeRunner struct {
	name string
	fn   func(ctx context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error)
}

func (f *fakeRunner) Run(ctx context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
	return f.fn(ctx, inputs)
}

func (f *fakeRunner) Name() string { return f.name }

func (f *fakeRunner) Close() {}

// engineWithFakeRunners builds an Engine whose runners map is populated with
// the provided fake runners (bypassing ORT entirely).
func engineWithFakeRunners(runners map[string]runnerIface) *Engine {
	return &Engine{runners: runners}
}

func TestTextConditioner_MissingGraph(t *testing.T) {
	// Engine with no runners at all → error mentioning "text_conditioner".
	e := engineWithFakeRunners(map[string]runnerIface{})
	_, err := e.TextConditioner(context.Background(), []int64{1, 2, 3})
	if err == nil {
		t.Fatal("expected error when text_conditioner graph is absent")
	}
}

func TestTextConditioner_EmptyTokens(t *testing.T) {
	// Even with the graph present, empty token slice should return an error.
	fake := &fakeRunner{
		name: "text_conditioner",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			// Should not be called.
			t.Error("Run should not be called for empty tokens")
			return nil, nil
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"text_conditioner": fake})
	_, err := e.TextConditioner(context.Background(), []int64{})
	if err == nil {
		t.Fatal("expected error for empty token slice")
	}
}

func TestTextConditioner_PropagatesRunnerError(t *testing.T) {
	fake := &fakeRunner{
		name: "text_conditioner",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			return nil, context.DeadlineExceeded
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"text_conditioner": fake})
	_, err := e.TextConditioner(context.Background(), []int64{1, 2})
	if err == nil {
		t.Fatal("expected error propagated from runner")
	}
}

func TestTextConditioner_ReturnsEmbeddings(t *testing.T) {
	const T = 3
	// Fake output: [1, T, 1024] float32
	outputData := make([]float32, 1*T*1024)
	for i := range outputData {
		outputData[i] = float32(i) * 0.001
	}
	fakeTensor, err := NewTensor(outputData, []int64{1, T, 1024})
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}

	fake := &fakeRunner{
		name: "text_conditioner",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			if _, ok := inputs["tokens"]; !ok {
				t.Error("expected 'tokens' input key")
			}
			return map[string]*Tensor{"text_embeddings": fakeTensor}, nil
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"text_conditioner": fake})
	got, err := e.TextConditioner(context.Background(), []int64{1, 2, 3})
	if err != nil {
		t.Fatalf("TextConditioner: %v", err)
	}
	// Shape: [1, T, 1024]
	shape := got.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != T || shape[2] != 1024 {
		t.Errorf("output shape = %v, want [1 %d 1024]", shape, T)
	}
}
