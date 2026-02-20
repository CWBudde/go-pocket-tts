package main

import "github.com/spf13/cobra"

func newModelCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "model",
		Short: "Model tooling commands",
	}

	cmd.AddCommand(newModelExportCmd())
	return cmd
}
