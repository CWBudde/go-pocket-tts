//go:build windows

package onnx

import (
	"context"
	"fmt"
)

// RunnerConfig holds ORT library settings for creating runners.
// In windows builds, native ORT runner support is currently unavailable.
type RunnerConfig struct {
	LibraryPath      string
	APIVersion       uint32
	ModelWeightsPath string
}

// Runner is unavailable in windows builds.
type Runner struct {
	name string
}

// NewRunner always returns an error in windows builds.
func NewRunner(meta Session, _ RunnerConfig) (*Runner, error) {
	return nil, fmt.Errorf("native onnx runner is unavailable on windows for graph %q", meta.Name)
}

// Run always returns an error in windows builds.
func (r *Runner) Run(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
	return nil, fmt.Errorf("native onnx runner is unavailable on windows for graph %q", r.name)
}

// Close is a no-op in windows builds.
func (r *Runner) Close() {}

// Name returns the graph name.
func (r *Runner) Name() string {
	return r.name
}
