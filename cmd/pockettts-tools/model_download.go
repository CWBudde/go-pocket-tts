package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/model"
	"github.com/spf13/cobra"
)

func newModelDownloadCmd() *cobra.Command {
	var hfRepo string
	var outDir string
	var hfToken string
	var fallbackUngated bool
	var fallbackRepo string

	cmd := &cobra.Command{
		Use:   "download",
		Short: "Download PocketTTS model files from Hugging Face",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			language, outDir, err := resolveModelDownload(cfg, outDir, cmd.Flags().Changed("out-dir"))
			if err != nil {
				return err
			}

			if hfToken == "" {
				hfToken = os.Getenv("HF_TOKEN")
			}

			err = model.Download(model.DownloadOptions{
				Repo:     hfRepo,
				Language: language,
				OutDir:   outDir,
				HFToken:  hfToken,
				Stdout:   os.Stdout,
				Stderr:   os.Stderr,
			})
			if err == nil {
				return nil
			}

			var denied *model.AccessDeniedError
			if fallbackUngated && hfToken == "" && errors.As(err, &denied) && hfRepo == "kyutai/pocket-tts" {
				_, _ = fmt.Fprintf(
					os.Stderr,
					"warning: %v; retrying with ungated repo %q\n",
					err,
					fallbackRepo,
				)

				err = model.Download(model.DownloadOptions{
					Repo:     fallbackRepo,
					Language: language,
					OutDir:   outDir,
					HFToken:  "",
					Stdout:   os.Stdout,
					Stderr:   os.Stderr,
				})
				if err == nil {
					_, _ = fmt.Fprintf(
						os.Stderr,
						"note: downloaded ungated model set (without voice cloning).\n",
					)

					return nil
				}
			}

			if err != nil {
				return fmt.Errorf("model download failed: %w", err)
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&hfRepo, "hf-repo", "kyutai/pocket-tts", "Hugging Face model repository")
	cmd.Flags().StringVar(&outDir, "out-dir", "models", "Directory where model files are stored (models/<language> for languages other than "+config.DefaultLanguage+")")
	cmd.Flags().StringVar(&hfToken, "hf-token", "", "Hugging Face token (falls back to HF_TOKEN env var)")
	cmd.Flags().BoolVar(&fallbackUngated, "fallback-ungated", true, "On gated access failure without token, retry with ungated repo")
	cmd.Flags().StringVar(&fallbackRepo, "fallback-repo", "kyutai/pocket-tts-without-voice-cloning", "Ungated repo used when --fallback-ungated is enabled")

	return cmd
}

// resolveModelDownload returns the language to download (--language) and the
// directory it goes to: --out-dir when set, else the directory of the
// language's default model path (models/ or models/<language>).
func resolveModelDownload(cfg config.Config, outDir string, outDirSet bool) (language, dir string, err error) {
	if cfg.TTS.ModelConfigPath != "" {
		return "", "", errors.New("custom model configs (--model-config) have no pinned download files; use --language")
	}

	language = cfg.TTS.Language
	if !outDirSet {
		outDir = filepath.Dir(filepath.FromSlash(config.PathsForLanguage(language).ModelPath))
	}

	return language, outDir, nil
}
