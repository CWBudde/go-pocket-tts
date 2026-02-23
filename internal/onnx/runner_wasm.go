//go:build js && wasm

package onnx

import (
	"context"
	"fmt"
)

// RunnerConfig holds ORT library settings for creating runners.
// In js/wasm builds, native ORT is unavailable; this struct is kept so the
// package API remains build-compatible.
type RunnerConfig struct {
	LibraryPath      string
	APIVersion       uint32
	ModelWeightsPath string
}

// Runner is unavailable in js/wasm builds. Use NewEngineWithRunners with a
// custom graph runner implementation (for example, a JS bridge).
type Runner struct {
	name string
}

// NewRunner always returns an error in js/wasm builds.
func NewRunner(meta Session, _ RunnerConfig) (*Runner, error) {
	return nil, fmt.Errorf("native onnx runner is unavailable in js/wasm for graph %q", meta.Name)
}

// Run always returns an error in js/wasm builds.
func (r *Runner) Run(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
	return nil, fmt.Errorf("native onnx runner is unavailable in js/wasm for graph %q", r.name)
}

// Close is a no-op in js/wasm builds.
func (r *Runner) Close() {}

// Name returns the graph name.
func (r *Runner) Name() string {
	return r.name
}
