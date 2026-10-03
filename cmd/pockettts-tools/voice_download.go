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
			"The voices go next to the voice manifest (--paths-voice-manifest, which follows --language),\n" +
			"or into --out-dir; the manifest lists them. Only the tracked flat voices/manifest.json of\n" +
			config.DefaultLanguage + " is left as is.",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			target, err := resolveVoiceDownload(cfg, outDir, cmd.Flags().Changed("out-dir"))
			if err != nil {
				return err
			}

			manifest, err := model.VoiceManifestForLanguage(target.language, voices)
			if err != nil {
				return fmt.Errorf("voice download failed: %w", err)
			}

			err = model.DownloadManifest(model.DownloadOptions{
				OutDir: target.dir,
				Stdout: os.Stdout,
				Stderr: os.Stderr,
			}, manifest)
			if err != nil {
				return fmt.Errorf("voice download failed: %w", err)
			}

			if !target.writeIndex {
				return nil
			}

			ids := make([]string, 0, len(manifest.Files))
			for _, f := range manifest.Files {
				ids = append(ids, strings.TrimSuffix(f.LocalPath, ".safetensors"))
			}

			err = model.WriteVoiceIndex(target.manifest, ids)
			if err != nil {
				return fmt.Errorf("voice download failed: %w", err)
			}

			_, _ = fmt.Fprintf(os.Stdout, "wrote voice manifest: %s\n", target.manifest)

			return nil
		},
	}

	cmd.Flags().StringVar(&outDir, "out-dir", "voices", "Directory where voice files and their manifest.json are stored (default: the directory of --paths-voice-manifest)")
	cmd.Flags().StringArrayVar(&voices, "voice", nil, "Predefined voice to download (repeatable; default: all voices of the language)")

	return cmd
}

// voiceDownloadTarget says where a voice download goes.
type voiceDownloadTarget struct {
	language   string // --language whose predefined voices to fetch
	dir        string // directory for the voice files
	manifest   string // voice manifest listing them
	writeIndex bool   // false only for the tracked flat voices/manifest.json
}

// resolveVoiceDownload targets the configured voice manifest
// (paths.voice_manifest, which follows --language unless set) and its
// directory, or dir/manifest.json when --out-dir is set.
func resolveVoiceDownload(cfg config.Config, outDir string, outDirSet bool) (voiceDownloadTarget, error) {
	if cfg.TTS.ModelConfigPath != "" {
		return voiceDownloadTarget{}, errors.New("custom model configs (--model-config) have no predefined voices to download; use --language")
	}

	manifest := filepath.FromSlash(cfg.Paths.VoiceManifest)
	if outDirSet {
		manifest = filepath.Join(outDir, "manifest.json")
	}

	flatDefault := filepath.FromSlash(config.PathsForLanguage(config.DefaultLanguage).VoiceManifest)

	return voiceDownloadTarget{
		language:   cfg.TTS.Language,
		dir:        filepath.Dir(manifest),
		manifest:   manifest,
		writeIndex: cfg.TTS.Language != model.FlatLayoutLanguage || filepath.Clean(manifest) != flatDefault,
	}, nil
}
