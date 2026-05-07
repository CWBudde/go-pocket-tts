package main

import "github.com/spf13/cobra"

func newVoiceCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "voice",
		Short: "Voice safetensors commands",
	}

	cmd.AddCommand(newVoiceDownloadCmd())

	return cmd
}
