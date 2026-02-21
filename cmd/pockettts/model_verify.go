package main

import (
	"fmt"
	"os"

	"github.com/example/go-pocket-tts/internal/model"
	"github.com/spf13/cobra"
)

func newModelVerifyCmd() *cobra.Command {
	var manifestPath string
	var ortAPIVersion uint32

	cmd := &cobra.Command{
		Use:   "verify",
		Short: "Run ONNX smoke inference for each exported model graph",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			err = model.VerifyONNX(model.VerifyOptions{
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
		},
	}

	cmd.Flags().StringVar(&manifestPath, "manifest", "models/onnx/manifest.json", "Path to ONNX manifest.json")
	cmd.Flags().Uint32Var(&ortAPIVersion, "ort-api-version", 23, "ONNX Runtime C API version expected by the purego binding")

	return cmd
}
