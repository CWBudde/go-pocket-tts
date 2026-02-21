package onnx

import (
	"fmt"
	"log/slog"
)

// Engine manages ONNX graph runners loaded from a manifest.
type Engine struct {
	runners map[string]*Runner
	sm      *SessionManager
}

// NewEngine loads the ONNX manifest and creates a Runner for each graph.
func NewEngine(manifestPath string, cfg RunnerConfig) (*Engine, error) {
	sm, err := NewSessionManager(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("load manifest: %w", err)
	}

	runners := make(map[string]*Runner, len(sm.Sessions()))
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

	return &Engine{runners: runners, sm: sm}, nil
}

// Runner returns the named graph runner, if it exists.
func (e *Engine) Runner(name string) (*Runner, bool) {
	r, ok := e.runners[name]
	return r, ok
}

// Close releases all ORT resources.
func (e *Engine) Close() {
	for _, r := range e.runners {
		r.Close()
	}
}

// Infer is a temporary compatibility shim.
// It will be replaced in Phase 18 with the full generation pipeline.
func (e *Engine) Infer(tokens []int) ([]float32, error) {
	return nil, fmt.Errorf("Engine.Infer not yet implemented; generation pipeline is Phase 18")
}
