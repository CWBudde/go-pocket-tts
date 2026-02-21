package main

import "github.com/spf13/cobra"

func newModelCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "model",
		Short: "Model acquisition and verification commands",
	}

	cmd.AddCommand(newModelDownloadCmd())
	cmd.AddCommand(newModelVerifyCmd())
	return cmd
}
