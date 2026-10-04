package main

import (
	"fmt"
	"os"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/model"
	"github.com/spf13/cobra"
)

func newVoiceDownloadCmd() *cobra.Command {
	var outDir string
	var voices []string

	cmd := &cobra.Command{
		Use:   "download",
		Short: "Download voice safetensors from Hugging Face",
		Long: "Download the predefined voices of --language from Hugging Face.\n\n" +
			"The voices go next to the voice manifest (--paths-voice-manifest, which follows --language),\n" +
			"or into --out-dir; the manifest lists them. Only the tracked flat voices/manifest.json of\n" +
			config.DefaultLanguage + " is left as is.",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			target, err := model.ResolveVoiceTarget(cfg, outDir, cmd.Flags().Changed("out-dir"))
			if err != nil {
				return err
			}

			err = model.DownloadVoices(target, voices, os.Stdout)
			if err != nil {
				return fmt.Errorf("voice download failed: %w", err)
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&outDir, "out-dir", "voices", "Directory where voice files and their manifest.json are stored (default: the directory of --paths-voice-manifest)")
	cmd.Flags().StringArrayVar(&voices, "voice", nil, "Predefined voice to download (repeatable; default: all voices of the language)")

	return cmd
}
