package main

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/onnx"
	"github.com/example/go-pocket-tts/internal/safetensors"
	"github.com/spf13/cobra"
)

type voiceEncoder interface {
	EncodeVoice(audioPath string) ([]float32, error)
	Close()
}

var buildVoiceEncoder = func(cfg config.Config, modelWeightsPath string) (voiceEncoder, error) {
	rcfg := onnx.RunnerConfig{
		LibraryPath:      cfg.Runtime.ORTLibraryPath,
		APIVersion:       23,
		ModelWeightsPath: modelWeightsPath,
	}
	if rcfg.LibraryPath == "" {
		info, err := onnx.DetectRuntime(cfg.Runtime)
		if err != nil {
			return nil, fmt.Errorf("detect ORT runtime: %w", err)
		}

		rcfg.LibraryPath = info.LibraryPath
	}

	engine, err := onnx.NewEngine(cfg.Paths.ONNXManifest, rcfg)
	if err != nil {
		return nil, fmt.Errorf("init onnx engine: %w", err)
	}

	return engine, nil
}

var writeVoiceSafetensors = func(path string, data []float32, shape []int64) error {
	return safetensors.WriteFile(path, []safetensors.Tensor{
		{
			Name:  "audio_prompt",
			Shape: shape,
			Data:  data,
		},
	})
}

func newExportVoiceCmd() *cobra.Command {
	var inputPath string
	var audioPathAlias string
	var outPath string
	var modelWeightsPath string
	var id string
	var license string

	cmd := &cobra.Command{
		Use:   "export-voice",
		Short: "Export a voice embedding (.safetensors) from a WAV/PCM prompt",
		RunE: func(_ *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			audioPath := strings.TrimSpace(inputPath)
			if audioPath == "" {
				audioPath = strings.TrimSpace(audioPathAlias)
			}

			if audioPath == "" {
				return errors.New("--input is required")
			}

			if strings.TrimSpace(outPath) == "" {
				return errors.New("--out is required")
			}

			_, err = os.Stat(audioPath)
			if err != nil {
				return fmt.Errorf("read --input %q: %w", audioPath, err)
			}

			resolvedWeightsPath := resolveExportVoiceModelPath(cfg, modelWeightsPath)

			encoder, err := buildVoiceEncoder(cfg, resolvedWeightsPath)
			if err != nil {
				return err
			}
			defer encoder.Close()

			embedding, err := encoder.EncodeVoice(audioPath)
			if err != nil {
				return err
			}

			if len(embedding) == 0 {
				return errors.New("encoded voice embedding is empty")
			}

			if len(embedding)%onnx.VoiceEmbeddingDim != 0 {
				return fmt.Errorf(
					"encoded voice embedding length %d is not divisible by %d",
					len(embedding),
					onnx.VoiceEmbeddingDim,
				)
			}

			shape := []int64{1, int64(len(embedding) / onnx.VoiceEmbeddingDim), onnx.VoiceEmbeddingDim}

			err = writeVoiceSafetensors(outPath, embedding, shape)
			if err != nil {
				return fmt.Errorf("write voice safetensors: %w", err)
			}

			_, _ = fmt.Fprintln(os.Stdout, "export-voice completed")
			_, _ = fmt.Fprintf(os.Stdout, "Suggested manifest entry:\n")
			_, _ = fmt.Fprintf(os.Stdout, "{\"id\":\"%s\",\"path\":\"%s\",\"license\":\"%s\"}\n", id, outPath, license)

			return nil
		},
	}

	cmd.Flags().StringVar(&inputPath, "input", "", "Input speaker audio WAV or raw PCM16 path")
	cmd.Flags().StringVar(&audioPathAlias, "audio", "", "Alias for --input")
	cmd.Flags().StringVar(&outPath, "out", "", "Output voice .safetensors path")
	cmd.Flags().StringVar(
		&modelWeightsPath,
		"model-safetensors",
		"",
		"Model .safetensors path (defaults to --paths-model-path when it points to .safetensors)",
	)
	cmd.Flags().StringVar(&id, "id", "custom-voice", "Voice ID for suggested manifest entry")
	cmd.Flags().StringVar(&license, "license", "unknown", "License label for suggested manifest entry")

	return cmd
}

func resolveExportVoiceModelPath(cfg config.Config, flagPath string) string {
	if p := strings.TrimSpace(flagPath); p != "" {
		return p
	}

	if p := strings.TrimSpace(cfg.Paths.ModelPath); strings.HasSuffix(strings.ToLower(p), ".safetensors") {
		return p
	}

	return ""
}
