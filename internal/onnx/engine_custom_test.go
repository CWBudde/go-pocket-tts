package onnx

import (
	"context"
	"testing"
)

type closeSpyRunner struct {
	name   string
	closed bool
}

func (c *closeSpyRunner) Run(context.Context, map[string]*Tensor) (map[string]*Tensor, error) {
	return map[string]*Tensor{}, nil
}

func (c *closeSpyRunner) Name() string { return c.name }

func (c *closeSpyRunner) Close() { c.closed = true }

func TestNewEngineWithRunners_CopiesInputMap(t *testing.T) {
	called := false
	tc := &fakeRunner{
		name: "text_conditioner",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			called = true

			out, err := NewTensor([]float32{0.1, 0.2}, []int64{1, 1, 2})
			if err != nil {
				t.Fatalf("NewTensor: %v", err)
			}

			return map[string]*Tensor{"text_embeddings": out}, nil
		},
	}

	orig := map[string]GraphRunner{"text_conditioner": tc}
	e := NewEngineWithRunners(orig)

	delete(orig, "text_conditioner")

	_, err := e.TextConditioner(context.Background(), []int64{1})
	if err != nil {
		t.Fatalf("TextConditioner returned error after map mutation: %v", err)
	}

	if !called {
		t.Fatal("expected copied runner to be called")
	}
}

func TestEngineRunnerAndClose(t *testing.T) {
	spy := &closeSpyRunner{name: "spy"}
	real := &Runner{name: "real"}

	e := &Engine{
		runners: map[string]GraphRunner{
			"spy":  spy,
			"real": real,
		},
	}

	if _, ok := e.Runner("missing"); ok {
		t.Fatal("Runner(missing) should not exist")
	}

	if _, ok := e.Runner("spy"); ok {
		t.Fatal("Runner(spy) should return false for non-*Runner concrete type")
	}

	got, ok := e.Runner("real")
	if !ok {
		t.Fatal("Runner(real) should exist and be concrete *Runner")
	}

	if got.Name() != "real" {
		t.Fatalf("Runner(real).Name() = %q, want real", got.Name())
	}

	e.Close()

	if !spy.closed {
		t.Fatal("expected spy runner to be closed")
	}
}
