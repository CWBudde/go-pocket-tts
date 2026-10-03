package main

import (
	"fmt"
	"os"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/model"
	"github.com/spf13/cobra"
)

func newModelExportCmd() *cobra.Command {
	var modelsDir string
	var outDir string
	var int8Quant bool
	var variant string
	var configPath string
	var pythonBin string
	var maxSeq int

	cmd := &cobra.Command{
		Use:   "export",
		Short: "Export PocketTTS PyTorch checkpoints to ONNX subgraphs",
		Long: "Export PocketTTS PyTorch checkpoints to ONNX subgraphs.\n\n" +
			"The global --language selects the model config unless --tts-config-path is given.\n" +
			"This is a tooling command and requires Python with pocket-tts/torch/onnx dependencies.",
		RunE: func(_ *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			err = model.ExportONNX(model.ExportOptions{
				ModelsDir: modelsDir,
				OutDir:    outDir,
				Int8:      int8Quant,
				Variant:   variant,
				Language:  exportLanguage(cfg, variant, configPath),
				Config:    configPath,
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
	cmd.Flags().BoolVar(&int8Quant, "int8", false, "Enable post-export INT8 quantization")
	cmd.Flags().StringVar(&variant, "variant", "", "Deprecated compatibility alias; b6369a24 maps to english_2026-01")
	_ = cmd.Flags().MarkDeprecated("variant", "use --language")
	cmd.Flags().StringVar(&configPath, "tts-config-path", "", "Path to an upstream PocketTTS config .yaml")
	cmd.Flags().StringVar(&pythonBin, "python-bin", "", "Python interpreter for export helper (auto-detected from pocket-tts by default)")
	cmd.Flags().IntVar(&maxSeq, "max-seq", 0, "KV-cache max sequence length (0 = script default 256; use 512+ for voice conditioning)")

	return cmd
}

// exportLanguage returns the language to export: the global --language, unless
// a legacy --variant or an upstream config file selects the model instead.
func exportLanguage(cfg config.Config, variant, configPath string) string {
	if variant != "" || configPath != "" {
		return ""
	}

	return cfg.TTS.Language
}
