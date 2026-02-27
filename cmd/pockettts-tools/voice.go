package main

import "github.com/spf13/cobra"

func newVoiceCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "voice",
		Short: "Voice embedding commands",
	}

	cmd.AddCommand(newVoiceDownloadCmd())

	return cmd
}
