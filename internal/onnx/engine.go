package onnx

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
)

// Engine manages ONNX graph runners loaded from a manifest.
type Engine struct {
	runners map[string]GraphRunner
	sm      *SessionManager

	manifestPath      string
	modelWeightsPath  string
	speakerProjOnce   sync.Once
	speakerProjWeight []float32
	speakerProjErr    error
}

// NewEngine loads the ONNX manifest and creates a Runner for each graph.
func NewEngine(manifestPath string, cfg RunnerConfig) (*Engine, error) {
	sm, err := NewSessionManager(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("load manifest: %w", err)
	}

	runners := make(map[string]GraphRunner, len(sm.Sessions()))
	for _, sess := range sm.Sessions() {
		runner, err := NewRunner(sess, cfg)
		if err != nil {
			for _, r := range runners {
				r.Close()
			}

			return nil, fmt.Errorf("create runner %q: %w", sess.Name, err)
		}

		runners[sess.Name] = runner
		slog.Info("created ONNX runner", "graph", sess.Name)
	}

	return &Engine{
		runners:          runners,
		sm:               sm,
		manifestPath:     manifestPath,
		modelWeightsPath: cfg.ModelWeightsPath,
	}, nil
}

// Runner returns the named graph runner, if it exists.
func (e *Engine) Runner(name string) (*Runner, bool) {
	r, ok := e.runners[name]
	if !ok {
		return nil, false
	}

	concrete, ok := r.(*Runner)

	return concrete, ok
}

// Close releases all ORT resources.
func (e *Engine) Close() {
	for _, r := range e.runners {
		r.Close()
	}
}

// TextConditioner runs the text_conditioner ONNX graph and returns text
// embeddings shaped [1, T, 1024] for the given SentencePiece token IDs.
func (e *Engine) TextConditioner(ctx context.Context, tokens []int64) (*Tensor, error) {
	if len(tokens) == 0 {
		return nil, errors.New("text_conditioner: token slice must not be empty")
	}

	runner, ok := e.runners["text_conditioner"]
	if !ok {
		return nil, errors.New("text_conditioner graph not found in manifest")
	}

	T := int64(len(tokens))

	tokenTensor, err := NewTensor(tokens, []int64{1, T})
	if err != nil {
		return nil, fmt.Errorf("text_conditioner: build token tensor: %w", err)
	}

	outputs, err := runner.Run(ctx, map[string]*Tensor{"tokens": tokenTensor})
	if err != nil {
		return nil, fmt.Errorf("text_conditioner: run: %w", err)
	}

	emb, ok := outputs["text_embeddings"]
	if !ok {
		return nil, errors.New("text_conditioner: missing 'text_embeddings' in output")
	}

	return emb, nil
}
