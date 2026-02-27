package main

import (
	"errors"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/spf13/cobra"
)

var (
	cfgFile   string
	activeCfg config.Config
)

func NewRootCmd() *cobra.Command {
	defaults := config.DefaultConfig()

	cmd := &cobra.Command{
		Use:   "pockettts-tools",
		Short: "PocketTTS tooling commands (Python-dependent)",
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

			return nil
		},
	}

	cmd.PersistentFlags().StringVar(&cfgFile, "config", "", "Optional config file (yaml|toml|json)")
	config.RegisterFlags(cmd.PersistentFlags(), defaults)

	cmd.AddCommand(newModelCmd())
	cmd.AddCommand(newVoiceCmd())
	cmd.AddCommand(newExportVoiceCmd())

	return cmd
}

func requireConfig() (config.Config, error) {
	if activeCfg.Paths.ModelPath == "" {
		return config.Config{}, errors.New("configuration not loaded")
	}

	return activeCfg, nil
}
