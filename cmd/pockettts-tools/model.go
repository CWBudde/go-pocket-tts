package main

import "github.com/spf13/cobra"

func newModelCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "model",
		Short: "Model tooling commands",
	}

	cmd.AddCommand(newModelDownloadCmd())
	cmd.AddCommand(newModelDownloadONNXCmd())
	cmd.AddCommand(newModelExportCmd())
	return cmd
}
