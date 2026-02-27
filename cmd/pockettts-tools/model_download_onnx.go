package main

import (
	"fmt"
	"os"

	"github.com/example/go-pocket-tts/internal/model"
	"github.com/spf13/cobra"
)

func newModelDownloadONNXCmd() *cobra.Command {
	var bundleID string
	var bundleURL string
	var bundleSHA string
	var variant string
	var outDir string
	var lockFile string

	cmd := &cobra.Command{
		Use:   "download-onnx",
		Short: "Download a prebuilt ONNX bundle and verify manifest",
		Long: "Download a prebuilt ONNX bundle (zip/tar.gz), verify checksum, extract into models/onnx,\n" +
			"and validate manifest + required graph files.\n\n" +
			"By default this resolves bundle metadata from bundles/onnx-bundles.lock.json.",
		RunE: func(_ *cobra.Command, _ []string) error {
			err := model.DownloadONNXBundle(model.DownloadONNXBundleOptions{
				BundleID:  bundleID,
				Variant:   variant,
				BundleURL: bundleURL,
				SHA256:    bundleSHA,
				LockFile:  lockFile,
				OutDir:    outDir,
				Stdout:    os.Stdout,
				Stderr:    os.Stderr,
			})
			if err != nil {
				return fmt.Errorf("download ONNX bundle failed: %w", err)
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&bundleID, "bundle-id", "", "Bundle ID from lock file (overrides --variant lookup)")
	cmd.Flags().StringVar(&bundleURL, "bundle-url", "", "Direct ONNX bundle URL/path (http(s) or file path)")
	cmd.Flags().StringVar(&bundleSHA, "sha256", "", "Expected SHA256 for the archive (optional when provided by lock file)")
	cmd.Flags().StringVar(&variant, "variant", "b6369a24", "Variant used for lock-file bundle selection")
	cmd.Flags().StringVar(&outDir, "out-dir", "models/onnx", "Output directory for extracted ONNX bundle")
	cmd.Flags().StringVar(&lockFile, "lock-file", "bundles/onnx-bundles.lock.json", "Lock file containing pinned ONNX bundle URLs/checksums")

	return cmd
}
