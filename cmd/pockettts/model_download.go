package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/model"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/spf13/cobra"
)

func newModelDownloadCmd() *cobra.Command {
	var hfRepo string
	var outDir string
	var hfToken string
	var fallbackUngated bool
	var fallbackRepo string
	var voices []string
	var allVoices bool
	var noVoices bool

	cmd := &cobra.Command{
		Use:   "download",
		Short: "Download PocketTTS model files from Hugging Face",
		Long: "Download the model and tokenizer of --language from Hugging Face, then its default voice\n" +
			"(or --voice / --all-voices) next to the voice manifest (--paths-voice-manifest).\n" +
			"The tracked voices/manifest.json of " + config.DefaultLanguage + " lists every voice, so there\n" +
			"all of them are fetched unless --voice is given.\n" +
			"Voices always come from the ungated repo, pinned like the model files.",
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

			err = downloadModel(hfRepo, language, outDir, hfToken, fallbackUngated, fallbackRepo)
			if err != nil {
				return fmt.Errorf("model download failed: %w", err)
			}

			if noVoices {
				return nil
			}

			target, selected, err := modelDownloadVoiceTarget(cfg, voices, allVoices)
			if err != nil {
				return err
			}

			err = model.DownloadVoices(target, selected, os.Stdout)
			if err != nil {
				return fmt.Errorf("voice download failed: %w", err)
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&hfRepo, "hf-repo", "kyutai/pocket-tts", "Hugging Face model repository")
	cmd.Flags().StringVar(&outDir, "out-dir", "models", "Directory where model files are stored (models/<language> for languages other than "+config.DefaultLanguage+")")
	cmd.Flags().StringVar(&hfToken, "hf-token", "", "Hugging Face token (falls back to HF_TOKEN env var)")
	cmd.Flags().BoolVar(&fallbackUngated, "fallback-ungated", true, "On gated access failure without token, retry with ungated repo")
	cmd.Flags().StringVar(&fallbackRepo, "fallback-repo", "kyutai/pocket-tts-without-voice-cloning", "Ungated repo used when --fallback-ungated is enabled")

	cmd.Flags().StringArrayVar(&voices, "voice", nil, "Predefined voice to download (repeatable; default: the language's default voice, or every voice of the tracked voices/manifest.json)")
	cmd.Flags().BoolVar(&allVoices, "all-voices", false, "Download every predefined voice of the language")
	cmd.Flags().BoolVar(&noVoices, "no-voices", false, "Download only the model and tokenizer")
	cmd.MarkFlagsMutuallyExclusive("voice", "all-voices", "no-voices")

	return cmd
}

// downloadModel fetches the model files of language from hfRepo. Without a
// token, a gated kyutai/pocket-tts failure retries with fallbackRepo when
// fallbackUngated is set.
func downloadModel(hfRepo, language, outDir, hfToken string, fallbackUngated bool, fallbackRepo string) error {
	err := model.Download(model.DownloadOptions{
		Repo:     hfRepo,
		Language: language,
		OutDir:   outDir,
		HFToken:  hfToken,
		Stdout:   os.Stdout,
		Stderr:   os.Stderr,
	})

	var denied *model.AccessDeniedError
	if err == nil || !fallbackUngated || hfToken != "" || !errors.As(err, &denied) || hfRepo != "kyutai/pocket-tts" {
		return err
	}

	_, _ = fmt.Fprintf(os.Stderr, "warning: %v; retrying with ungated repo %q\n", err, fallbackRepo)

	err = model.Download(model.DownloadOptions{
		Repo:     fallbackRepo,
		Language: language,
		OutDir:   outDir,
		Stdout:   os.Stdout,
		Stderr:   os.Stderr,
	})
	if err != nil {
		return err
	}

	_, _ = fmt.Fprintf(os.Stderr, "note: downloaded ungated model set (without voice cloning).\n")

	return nil
}

// modelDownloadVoiceTarget returns where model download puts the voices and
// which ones it fetches.
func modelDownloadVoiceTarget(cfg config.Config, voices []string, all bool) (model.VoiceTarget, []string, error) {
	target, err := model.ResolveVoiceTarget(cfg, "", false)
	if err != nil {
		return model.VoiceTarget{}, nil, err
	}

	selected, err := modelDownloadVoices(cfg, voices, all || !target.WriteIndex)
	if err != nil {
		return model.VoiceTarget{}, nil, err
	}

	return target, selected, nil
}

// modelDownloadVoices returns the voices model download fetches: the --voice
// ids, else every one (nil) with allByDefault (--all-voices, or the tracked
// flat voices/manifest.json, which lists them all and is not rewritten), else
// the language's default voice (upstream get_default_voice_for_language).
func modelDownloadVoices(cfg config.Config, voices []string, allByDefault bool) ([]string, error) {
	if len(voices) > 0 {
		return voices, nil
	}

	if allByDefault {
		return nil, nil
	}

	mc := cfg.Model
	if mc == nil {
		var err error

		mc, err = modelcfg.Lookup(cfg.TTS.Language)
		if err != nil {
			return nil, err
		}
	}

	return []string{mc.DefaultVoice}, nil
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
