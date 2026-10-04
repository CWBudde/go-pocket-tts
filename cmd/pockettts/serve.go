package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/model"
	"github.com/cwbudde/go-pocket-tts/internal/server"
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

			// Other backends reject --default-voice at startup; don't download
			// a voice for them first.
			if backend == config.BackendNative {
				cfg.Server.DefaultVoice, err = resolveServeDefaultVoice(
					cfg.Server.DefaultVoice, userCacheVoiceFetcher(nil, os.Stdout))
				if err != nil {
					return err
				}
			}

			// The server loads the model itself, after it has checked the
			// default voice, so a broken voice fails before the weights load.
			srv := server.New(cfg, nil).
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

// resolveServeDefaultVoice turns --default-voice into what the server takes:
// a voice ID or .safetensors path passes through (the server resolves and
// loads it at startup), and an https:// or hf:// voice is downloaded by
// fetch. WAV prompts need the Mimi encoder, which the native backend lacks.
func resolveServeDefaultVoice(ref string, fetch func(string) (string, error)) (string, error) {
	ref = strings.TrimSpace(ref)

	if model.VoiceRefExt(ref) == ".wav" {
		return "", fmt.Errorf("--default-voice %q: WAV voices are not supported yet (the native backend has no "+
			"Mimi encoder); pass a .safetensors voice, e.g. from 'pockettts export-voice'", ref)
	}

	if !model.IsRemoteVoice(ref) {
		return ref, nil
	}

	path, err := fetch(ref)
	if err != nil {
		return "", fmt.Errorf("--default-voice: %w", err)
	}

	return path, nil
}

// userCacheVoiceFetcher downloads URL voices once into
// <user cache dir>/pockettts/voices. hf:// downloads send HF_TOKEN when set.
// A nil client means http.DefaultClient.
func userCacheVoiceFetcher(client *http.Client, stdout io.Writer) func(string) (string, error) {
	return func(ref string) (string, error) {
		cacheDir, err := os.UserCacheDir()
		if err != nil {
			return "", fmt.Errorf("voice %s: no user cache dir: %w", ref, err)
		}

		return model.FetchVoice(ref, model.FetchVoiceOptions{
			Client:   client,
			CacheDir: filepath.Join(cacheDir, "pockettts", "voices"),
			Token:    os.Getenv("HF_TOKEN"),
			Stdout:   stdout,
		})
	}
}
