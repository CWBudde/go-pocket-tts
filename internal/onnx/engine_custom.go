package onnx

import (
	"context"
	"maps"
)

// GraphRunner is the minimal runner contract required by Engine methods.
// It is useful for alternate runtimes (for example js/wasm bridge runners).
type GraphRunner interface {
	Run(ctx context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error)
	Name() string
	Close()
}

type runnerIface = GraphRunner

// NewEngineWithRunners builds an Engine from externally provided graph runners.
func NewEngineWithRunners(runners map[string]GraphRunner) *Engine {
	internal := make(map[string]GraphRunner, len(runners))
	maps.Copy(internal, runners)

	return &Engine{runners: internal}
}
