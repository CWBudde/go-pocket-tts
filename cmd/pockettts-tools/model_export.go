package main

import (
	"fmt"
	"os"

	"github.com/example/go-pocket-tts/internal/model"
	"github.com/spf13/cobra"
)

func newModelExportCmd() *cobra.Command {
	var modelsDir string
	var outDir string
	var int8 bool
	var variant string
	var pythonBin string
	var maxSeq int

	cmd := &cobra.Command{
		Use:   "export",
		Short: "Export PocketTTS PyTorch checkpoints to ONNX subgraphs",
		Long: "Export PocketTTS PyTorch checkpoints to ONNX subgraphs.\n\n" +
			"This is a tooling command and requires Python with pocket-tts/torch/onnx dependencies.",
		RunE: func(cmd *cobra.Command, _ []string) error {
			err := model.ExportONNX(model.ExportOptions{
				ModelsDir: modelsDir,
				OutDir:    outDir,
				Int8:      int8,
				Variant:   variant,
				PythonBin: pythonBin,
				MaxSeq:    maxSeq,
				Stdout:    os.Stdout,
				Stderr:    os.Stderr,
			})
			if err != nil {
				return fmt.Errorf(
					"model export failed: %w\nhint: this command requires Python tooling (pocket-tts, torch, onnx)",
					err,
				)
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&modelsDir, "models-dir", "models", "Directory containing downloaded model files")
	cmd.Flags().StringVar(&outDir, "out-dir", "models/onnx", "Directory for ONNX output files")
	cmd.Flags().BoolVar(&int8, "int8", false, "Enable post-export INT8 quantization")
	cmd.Flags().StringVar(&variant, "variant", "b6369a24", "PocketTTS model variant signature or config alias")
	cmd.Flags().StringVar(&pythonBin, "python-bin", "", "Python interpreter for export helper (auto-detected from pocket-tts by default)")
	cmd.Flags().IntVar(&maxSeq, "max-seq", 0, "KV-cache max sequence length (0 = script default 256; use 512+ for voice conditioning)")

	return cmd
}
