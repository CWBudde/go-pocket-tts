package main

import (
	"fmt"
	"os"

	"github.com/example/go-pocket-tts/internal/server"
	"github.com/spf13/cobra"
)

func newHealthCmd() *cobra.Command {
	var addr string

	cmd := &cobra.Command{
		Use:   "health",
		Short: "Check server health endpoint",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}
			if addr == "" {
				addr = cfg.Server.ListenAddr
			}
			if err := server.ProbeHTTP(addr); err != nil {
				return err
			}
			_, err = fmt.Fprintln(os.Stdout, "ok")
			return err
		},
	}

	cmd.Flags().StringVar(&addr, "addr", "", "HTTP server address to probe")

	return cmd
}
