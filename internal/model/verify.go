package model

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/example/go-pocket-tts/internal/onnx"
	ort "github.com/shota3506/onnxruntime-purego/onnxruntime"
)

type VerifyOptions struct {
	ManifestPath  string
	ORTLibrary    string
	ORTAPIVersion uint32
	Stdout        io.Writer
	Stderr        io.Writer
}

var runNativeVerify = runNativeVerifyImpl

func VerifyONNX(opts VerifyOptions) error {
	if opts.ManifestPath == "" {
		return errors.New("manifest path is required")
	}

	if opts.ORTAPIVersion == 0 {
		opts.ORTAPIVersion = 23
	}

	if opts.Stdout == nil {
		opts.Stdout = io.Discard
	}

	if opts.Stderr == nil {
		opts.Stderr = io.Discard
	}

	sm, err := onnx.NewSessionManager(opts.ManifestPath)
	if err != nil {
		return fmt.Errorf("load sessions: %w", err)
	}

	for _, session := range sm.Sessions() {
		for _, input := range session.Inputs {
			if _, err := onnx.NewZeroTensor(input.DType, input.Shape); err != nil {
				return fmt.Errorf("session %q input %q invalid: %w", session.Name, input.Name, err)
			}
		}
	}

	if err := runNativeVerify(sm.Sessions(), opts); err != nil {
		return err
	}

	return nil
}

func runNativeVerifyImpl(sessions []onnx.Session, opts VerifyOptions) error {
	runtime, err := ort.NewRuntime(opts.ORTLibrary, opts.ORTAPIVersion)
	if err != nil {
		return fmt.Errorf("initialize ONNX Runtime (lib=%q api=%d): %w", opts.ORTLibrary, opts.ORTAPIVersion, err)
	}

	defer func() { _ = runtime.Close() }()

	env, err := runtime.NewEnv("pockettts-model-verify", ort.LoggingLevelWarning)
	if err != nil {
		return fmt.Errorf("create ONNX Runtime env: %w", err)
	}
	defer env.Close()

	var failures []string

	for _, session := range sessions {
		err := runSessionSmoke(context.Background(), runtime, env, session)
		if err != nil {
			_, _ = fmt.Fprintf(opts.Stderr, "FAIL %s: %v\n", session.Name, err)
			failures = append(failures, session.Name)

			continue
		}

		_, _ = fmt.Fprintf(opts.Stdout, "PASS %s\n", session.Name)
	}

	if len(failures) > 0 {
		return fmt.Errorf("verify failed for %d session(s): %s", len(failures), strings.Join(failures, ", "))
	}

	return nil
}

func runSessionSmoke(ctx context.Context, runtime *ort.Runtime, env *ort.Env, session onnx.Session) error {
	s, err := runtime.NewSession(env, session.Path, nil)
	if err != nil {
		return fmt.Errorf("load session model: %w", err)
	}
	defer s.Close()

	inputs := make(map[string]*ort.Value, len(session.Inputs))
	for _, input := range session.Inputs {
		t, err := onnx.NewZeroTensor(input.DType, input.Shape)
		if err != nil {
			return fmt.Errorf("build input %q tensor: %w", input.Name, err)
		}

		v, err := tensorToORTValue(runtime, t)
		if err != nil {
			return fmt.Errorf("convert input %q to runtime tensor: %w", input.Name, err)
		}

		inputs[input.Name] = v
	}

	defer func() {
		for _, v := range inputs {
			v.Close()
		}
	}()

	outputs, err := s.Run(ctx, inputs)
	if err != nil {
		return fmt.Errorf("run inference: %w", err)
	}

	for _, out := range outputs {
		out.Close()
	}

	return nil
}

func tensorToORTValue(runtime *ort.Runtime, t *onnx.Tensor) (*ort.Value, error) {
	switch data := t.Data().(type) {
	case []float32:
		return ort.NewTensorValue(runtime, data, t.Shape())
	case []int64:
		return ort.NewTensorValue(runtime, data, t.Shape())
	default:
		return nil, fmt.Errorf("unsupported tensor backing type %T", data)
	}
}
