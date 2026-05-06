package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/onnx"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
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

const (
	exportVoiceFormatLegacyEmbedding = "legacy-embedding"
	exportVoiceFormatModelState      = "model-state"
)

var exportVoiceModelState = func(ctx context.Context, cfg config.Config, audioPath, outPath, language string) error {
	exe, err := resolvePocketTTSCLI(cfg.TTS.CLIPath)
	if err != nil {
		return err
	}

	args := []string{"export-voice", audioPath, outPath}
	if cfg.TTS.Quiet {
		args = append(args, "--quiet")
	}

	if configPath := strings.TrimSpace(cfg.TTS.CLIConfigPath); configPath != "" {
		args = append(args, "--config", configPath)
	} else if lang := strings.TrimSpace(language); lang != "" {
		args = append(args, "--language", lang)
	}

	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("run pocket-tts export-voice: %w", err)
	}

	kind, err := safetensors.InspectVoiceFile(outPath)
	if err != nil {
		return fmt.Errorf("inspect exported voice state: %w", err)
	}

	if kind != safetensors.VoiceFileModelState {
		return fmt.Errorf("exported voice file kind %q, want %q", kind, safetensors.VoiceFileModelState)
	}

	return nil
}

func newExportVoiceCmd() *cobra.Command {
	var inputPath string
	var audioPathAlias string
	var outPath string
	var modelWeightsPath string
	var format string
	var language string
	var id string
	var license string

	cmd := &cobra.Command{
		Use:   "export-voice",
		Short: "Export a voice .safetensors file from a WAV/PCM prompt",
		RunE: func(cmd *cobra.Command, _ []string) error {
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

			switch normalizeExportVoiceFormat(format) {
			case exportVoiceFormatModelState:
				err = exportVoiceModelState(cmd.Context(), cfg, audioPath, outPath, language)
				if err != nil {
					return err
				}

				_, _ = fmt.Fprintln(os.Stdout, "export-voice completed")
				_, _ = fmt.Fprintf(os.Stdout, "Suggested manifest entry:\n")
				_, _ = fmt.Fprintf(os.Stdout, "{\"id\":\"%s\",\"path\":\"%s\",\"license\":\"%s\"}\n", id, outPath, license)

				return nil
			case exportVoiceFormatLegacyEmbedding:
				// Continue below through the native ONNX voice-embedding path.
			default:
				return fmt.Errorf(
					"unsupported --format %q (supported: %s, %s)",
					format,
					exportVoiceFormatLegacyEmbedding,
					exportVoiceFormatModelState,
				)
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
	cmd.Flags().StringVar(
		&format,
		"format",
		exportVoiceFormatLegacyEmbedding,
		"Output format: legacy-embedding or model-state",
	)
	cmd.Flags().StringVar(
		&language,
		"language",
		"english_2026-01",
		"Python pocket-tts language for --format=model-state when --tts-cli-config-path is not set",
	)
	cmd.Flags().StringVar(&id, "id", "custom-voice", "Voice ID for suggested manifest entry")
	cmd.Flags().StringVar(&license, "license", "unknown", "License label for suggested manifest entry")

	return cmd
}

func normalizeExportVoiceFormat(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "", exportVoiceFormatLegacyEmbedding, "legacy", "embedding", "audio-prompt", "audio_prompt":
		return exportVoiceFormatLegacyEmbedding
	case exportVoiceFormatModelState, "modelstate", "state", "upstream", "python":
		return exportVoiceFormatModelState
	default:
		return strings.TrimSpace(format)
	}
}

func resolvePocketTTSCLI(flagPath string) (string, error) {
	if p := strings.TrimSpace(flagPath); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}

		resolved, err := exec.LookPath(p)
		if err == nil {
			return resolved, nil
		}

		return "", fmt.Errorf("pocket-tts executable %q not found: %w", p, err)
	}

	if resolved, err := exec.LookPath("pocket-tts"); err == nil {
		return resolved, nil
	}

	local := filepath.Join("original", "pockettts", ".venv", "bin", "pocket-tts")
	if _, err := os.Stat(local); err == nil {
		return local, nil
	}

	return "", errors.New("pocket-tts executable not found; set --tts-cli-path or install original/pockettts/.venv")
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
