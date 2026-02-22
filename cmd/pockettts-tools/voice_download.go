package main

import (
	"fmt"
	"os"

	"github.com/example/go-pocket-tts/internal/model"
	"github.com/spf13/cobra"
)

func newVoiceDownloadCmd() *cobra.Command {
	var outDir string

	cmd := &cobra.Command{
		Use:   "download",
		Short: "Download voice embeddings from Hugging Face",
		RunE: func(_ *cobra.Command, _ []string) error {
			manifest := model.VoiceManifest()
			err := model.DownloadManifest(model.DownloadOptions{
				OutDir: outDir,
				Stdout: os.Stdout,
				Stderr: os.Stderr,
			}, manifest)
			if err != nil {
				return fmt.Errorf("voice download failed: %w", err)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&outDir, "out-dir", "voices", "Directory where voice files are stored")

	return cmd
}
