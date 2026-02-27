//go:build !js || !wasm

package onnx

import (
	"context"
	"fmt"

	ort "github.com/shota3506/onnxruntime-purego/onnxruntime"
)

// RunnerConfig holds ORT library settings for creating runners.
type RunnerConfig struct {
	LibraryPath      string
	APIVersion       uint32
	ModelWeightsPath string // Optional .safetensors checkpoint path (used by voice encoding).
}

// Runner wraps an ORT session for a single ONNX graph.
type Runner struct {
	name    string
	runtime *ort.Runtime
	env     *ort.Env
	session *ort.Session
	meta    Session
}

// NewRunner creates a runner for a single ONNX graph session.
func NewRunner(meta Session, cfg RunnerConfig) (*Runner, error) {
	if cfg.APIVersion == 0 {
		cfg.APIVersion = 23
	}

	runtime, err := ort.NewRuntime(cfg.LibraryPath, cfg.APIVersion)
	if err != nil {
		return nil, fmt.Errorf("ort runtime for %q: %w", meta.Name, err)
	}

	env, err := runtime.NewEnv("pockettts-"+meta.Name, ort.LoggingLevelWarning)
	if err != nil {
		_ = runtime.Close()
		return nil, fmt.Errorf("ort env for %q: %w", meta.Name, err)
	}

	session, err := runtime.NewSession(env, meta.Path, nil)
	if err != nil {
		env.Close()
		_ = runtime.Close()

		return nil, fmt.Errorf("ort session for %q (%s): %w", meta.Name, meta.Path, err)
	}

	return &Runner{
		name:    meta.Name,
		runtime: runtime,
		env:     env,
		session: session,
		meta:    meta,
	}, nil
}

// Run executes the ONNX graph with the given named input tensors.
func (r *Runner) Run(ctx context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
	ortInputs := make(map[string]*ort.Value, len(inputs))
	for name, t := range inputs {
		v, err := tensorToORT(r.runtime, t)
		if err != nil {
			closeORTValues(ortInputs)
			return nil, fmt.Errorf("input %q: %w", name, err)
		}

		ortInputs[name] = v
	}

	defer closeORTValues(ortInputs)

	ortOutputs, err := r.session.Run(ctx, ortInputs)
	if err != nil {
		return nil, fmt.Errorf("run %q: %w", r.name, err)
	}
	defer closeORTValues(ortOutputs)

	results := make(map[string]*Tensor, len(ortOutputs))
	for name, v := range ortOutputs {
		t, err := ortToTensor(v)
		if err != nil {
			return nil, fmt.Errorf("output %q: %w", name, err)
		}

		results[name] = t
	}

	return results, nil
}

// Close releases all ORT resources. Safe to call multiple times.
func (r *Runner) Close() {
	if r.session != nil {
		r.session.Close()
		r.session = nil
	}

	if r.env != nil {
		r.env.Close()
		r.env = nil
	}

	if r.runtime != nil {
		_ = r.runtime.Close()
		r.runtime = nil
	}
}

// Name returns the graph name from the manifest.
func (r *Runner) Name() string {
	return r.name
}

func tensorToORT(runtime *ort.Runtime, t *Tensor) (*ort.Value, error) {
	switch data := t.Data().(type) {
	case []float32:
		return ort.NewTensorValue(runtime, data, t.Shape())
	case []int64:
		return ort.NewTensorValue(runtime, data, t.Shape())
	default:
		return nil, fmt.Errorf("unsupported tensor dtype %T", data)
	}
}

func ortToTensor(v *ort.Value) (*Tensor, error) {
	elemType, err := v.GetTensorElementType()
	if err != nil {
		return nil, fmt.Errorf("get element type: %w", err)
	}

	switch elemType {
	case ort.ONNXTensorElementDataTypeFloat:
		data, shape, err := ort.GetTensorData[float32](v)
		if err != nil {
			return nil, err
		}

		return NewTensor(data, shape)
	case ort.ONNXTensorElementDataTypeInt64:
		data, shape, err := ort.GetTensorData[int64](v)
		if err != nil {
			return nil, err
		}

		return NewTensor(data, shape)
	default:
		return nil, fmt.Errorf("unsupported ORT element type %d", elemType)
	}
}

func closeORTValues(vals map[string]*ort.Value) {
	for _, v := range vals {
		if v != nil {
			v.Close()
		}
	}
}
