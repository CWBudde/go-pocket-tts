package main

import (
	"context"
	"os/signal"
	"syscall"
	"time"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/server"
	"github.com/example/go-pocket-tts/internal/tts"
	"github.com/spf13/cobra"
)

func newServeCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Run PocketTTS HTTP server",
		RunE: func(_ *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			backend, err := config.NormalizeBackend(cfg.TTS.Backend)
			if err != nil {
				return err
			}

			var svc *tts.Service
			if backend == config.BackendNative || backend == config.BackendNativeONNX {
				svc, err = tts.NewService(cfg)
				if err != nil {
					return err
				}
			}

			srv := server.New(cfg, svc).
				WithShutdownTimeout(time.Duration(cfg.Server.ShutdownTimeout) * time.Second)

			ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
			defer stop()

			return srv.Start(ctx)
		},
	}

	defaults := config.DefaultConfig()
	config.RegisterFlags(cmd.Flags(), defaults)

	return cmd
}
