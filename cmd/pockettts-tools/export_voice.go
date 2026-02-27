package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"

	pockettts "github.com/MeKo-Christian/go-call-pocket-tts"
	"github.com/spf13/cobra"
)

func newExportVoiceCmd() *cobra.Command {
	var audioPath string
	var outPath string
	var id string
	var license string

	cmd := &cobra.Command{
		Use:   "export-voice",
		Short: "Export a voice embedding (.safetensors) from a WAV prompt",
		Long: "Export a voice embedding (.safetensors) from a WAV prompt.\n\n" +
			"This is an optional tooling command and requires a Python pocket-tts installation.",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			if audioPath == "" {
				return errors.New("--audio is required")
			}

			if outPath == "" {
				return errors.New("--out is required")
			}

			exe := cfg.TTS.CLIPath
			if exe == "" {
				exe = "pocket-tts"
			}

			_, err = exec.LookPath(exe)
			if err != nil {
				return fmt.Errorf(
					"export-voice requires the pocket-tts CLI (Python tooling) on PATH or --tts-cli-path: %w",
					err,
				)
			}

			err = pockettts.ExportVoice(cmd.Context(), audioPath, outPath, &pockettts.ExportVoiceOptions{
				Config:         cfg.TTS.CLIConfigPath,
				Quiet:          cfg.TTS.Quiet,
				ExecutablePath: cfg.TTS.CLIPath,
				LogWriter:      os.Stderr,
			})
			if err != nil {
				var notFound *pockettts.ErrExecutableNotFound
				if errors.As(err, &notFound) {
					return fmt.Errorf(
						"export-voice requires the pocket-tts CLI (Python tooling): %w",
						err,
					)
				}

				return err
			}

			_, _ = fmt.Fprintln(os.Stdout, "export-voice completed")
			_, _ = fmt.Fprintf(os.Stdout, "Suggested manifest entry:\n")
			_, _ = fmt.Fprintf(os.Stdout, "{\"id\":\"%s\",\"path\":\"%s\",\"license\":\"%s\"}\n", id, outPath, license)

			return nil
		},
	}

	cmd.Flags().StringVar(&audioPath, "audio", "", "Input speaker audio WAV path")
	cmd.Flags().StringVar(&outPath, "out", "", "Output voice .safetensors path")
	cmd.Flags().StringVar(&id, "id", "custom-voice", "Voice ID for suggested manifest entry")
	cmd.Flags().StringVar(&license, "license", "unknown", "License label for suggested manifest entry")

	return cmd
}
