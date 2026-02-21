package main

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/server"
	"github.com/spf13/cobra"
)

var (
	cfgFile   string
	activeCfg config.Config
)

func NewRootCmd() *cobra.Command {
	defaults := config.DefaultConfig()

	cmd := &cobra.Command{
		Use:   "pockettts",
		Short: "PocketTTS command line",
		PersistentPreRunE: func(cmd *cobra.Command, _ []string) error {
			loaded, err := config.Load(config.LoadOptions{
				Cmd:        cmd,
				ConfigFile: cfgFile,
				Defaults:   defaults,
			})
			if err != nil {
				return err
			}
			activeCfg = loaded
			setupLogger(loaded.LogLevel)
			return nil
		},
	}

	cmd.PersistentFlags().StringVar(&cfgFile, "config", "", "Optional config file (yaml|toml|json)")
	config.RegisterFlags(cmd.PersistentFlags(), defaults)

	cmd.AddCommand(newSynthCmd())
	cmd.AddCommand(newBenchCmd())
	cmd.AddCommand(newModelCmd())
	cmd.AddCommand(newServeCmd())
	cmd.AddCommand(newHealthCmd())
	cmd.AddCommand(newDoctorCmd())

	return cmd
}

// setupLogger configures the process-wide slog default logger.
func setupLogger(levelStr string) {
	lvl, err := server.ParseLogLevel(levelStr)
	if err != nil {
		lvl = slog.LevelInfo
	}
	h := slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: lvl})
	slog.SetDefault(slog.New(h))
}

func requireConfig() (config.Config, error) {
	if activeCfg.Paths.ModelPath == "" {
		return config.Config{}, fmt.Errorf("configuration not loaded")
	}
	return activeCfg, nil
}
