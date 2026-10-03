package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

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
			"For languages other than " + config.DefaultLanguage + " this also writes manifest.json\n" +
			"next to the voices (the file --paths-voice-manifest points to).",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			language, outDir, err := resolveVoiceDownload(cfg, outDir, cmd.Flags().Changed("out-dir"))
			if err != nil {
				return err
			}

			manifest, err := model.VoiceManifestForLanguage(language, voices)
			if err != nil {
				return fmt.Errorf("voice download failed: %w", err)
			}

			err = model.DownloadManifest(model.DownloadOptions{
				OutDir: outDir,
				Stdout: os.Stdout,
				Stderr: os.Stderr,
			}, manifest)
			if err != nil {
				return fmt.Errorf("voice download failed: %w", err)
			}

			// The flat layout keeps its checked-in voices/manifest.json.
			if language == model.FlatLayoutLanguage {
				return nil
			}

			ids := make([]string, 0, len(manifest.Files))
			for _, f := range manifest.Files {
				ids = append(ids, strings.TrimSuffix(f.LocalPath, ".safetensors"))
			}

			err = model.WriteVoiceIndex(outDir, ids)
			if err != nil {
				return fmt.Errorf("voice download failed: %w", err)
			}

			_, _ = fmt.Fprintf(os.Stdout, "wrote voice manifest: %s\n", filepath.Join(outDir, "manifest.json"))

			return nil
		},
	}

	cmd.Flags().StringVar(&outDir, "out-dir", "voices", "Directory where voice files are stored (voices/<language> for languages other than "+config.DefaultLanguage+")")
	cmd.Flags().StringArrayVar(&voices, "voice", nil, "Predefined voice to download (repeatable; default: all voices of the language)")

	return cmd
}

// resolveVoiceDownload returns the language whose voices to download
// (--language) and the directory they go to: --out-dir when set, else the
// directory of the language's default voice manifest (voices/ or
// voices/<language>).
func resolveVoiceDownload(cfg config.Config, outDir string, outDirSet bool) (language, dir string, err error) {
	if cfg.TTS.ModelConfigPath != "" {
		return "", "", errors.New("custom model configs (--model-config) have no predefined voices to download; use --language")
	}

	language = cfg.TTS.Language
	if !outDirSet {
		outDir = filepath.Dir(filepath.FromSlash(config.PathsForLanguage(language).VoiceManifest))
	}

	return language, outDir, nil
}
