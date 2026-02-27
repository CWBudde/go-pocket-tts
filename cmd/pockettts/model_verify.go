package main

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/example/go-pocket-tts/internal/config"
	nativemodel "github.com/example/go-pocket-tts/internal/native"
	"github.com/example/go-pocket-tts/internal/model"
	"github.com/example/go-pocket-tts/internal/safetensors"
	"github.com/spf13/cobra"
)

func newModelVerifyCmd() *cobra.Command {
	var manifestPath string
	var ortAPIVersion uint32
	var backend string

	cmd := &cobra.Command{
		Use:   "verify",
		Short: "Run smoke inference / validation for the configured backend",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			be := backend
			if be == "" {
				be = cfg.TTS.Backend
			}
			be, err = config.NormalizeBackend(be)
			if err != nil {
				return err
			}

			switch be {
			case config.BackendNative:
				return verifyNativeSafetensors(cfg)
			default:
				return verifyONNX(manifestPath, cfg, ortAPIVersion)
			}
		},
	}

	cmd.Flags().StringVar(&backend, "backend", "", "Backend to verify (default: configured backend)")
	cmd.Flags().StringVar(&manifestPath, "manifest", "models/onnx/manifest.json", "Path to ONNX manifest.json")
	cmd.Flags().Uint32Var(&ortAPIVersion, "ort-api-version", 23, "ONNX Runtime C API version expected by the purego binding")

	return cmd
}

func verifyNativeSafetensors(cfg config.Config) error {
	modelPath := cfg.Paths.ModelPath
	fmt.Fprintf(os.Stdout, "verifying safetensors model: %s\n", modelPath)

	// 1. Check file exists.
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("model file not found: %w", err)
	}
	fmt.Fprintf(os.Stdout, "  ✓ file exists\n")

	// 2. Validate header keys.
	if err := safetensors.ValidateModelKeys(modelPath); err != nil {
		return fmt.Errorf("key validation failed: %w", err)
	}
	fmt.Fprintf(os.Stdout, "  ✓ tensor keys valid\n")

	// 3. Smoke load the model.
	m, err := nativemodel.LoadModelFromSafetensors(modelPath, nativemodel.DefaultConfig())
	if err != nil {
		return fmt.Errorf("smoke load failed: %w", err)
	}
	m.Close()
	fmt.Fprintf(os.Stdout, "  ✓ model loads successfully\n")

	// 4. Check tokenizer.
	tokPath := cfg.Paths.TokenizerModel
	if _, err := os.Stat(tokPath); err != nil {
		slog.Warn("tokenizer model not found", "path", tokPath)
		fmt.Fprintf(os.Stdout, "  ⚠ tokenizer model not found: %s\n", tokPath)
	} else {
		fmt.Fprintf(os.Stdout, "  ✓ tokenizer model: %s\n", tokPath)
	}

	fmt.Fprintln(os.Stdout, "native-safetensors model verification passed")
	return nil
}

func verifyONNX(manifestPath string, cfg config.Config, ortAPIVersion uint32) error {
	err := model.VerifyONNX(model.VerifyOptions{
		ManifestPath:  manifestPath,
		ORTLibrary:    cfg.Runtime.ORTLibraryPath,
		ORTAPIVersion: ortAPIVersion,
		Stdout:        os.Stdout,
		Stderr:        os.Stderr,
	})
	if err != nil {
		return fmt.Errorf("model verify failed: %w", err)
	}
	return nil
}
