package main

import (
	"testing"

	"github.com/example/go-pocket-tts/internal/config"
)

func TestNewRootCmd_HasExpectedSubcommands(t *testing.T) {
	root := NewRootCmd()

	want := []string{"synth", "export-voice", "bench", "model", "serve", "health", "doctor"}
	for _, name := range want {
		found := false

		for _, sub := range root.Commands() {
			if sub.Name() == name {
				found = true
				break
			}
		}

		if !found {
			t.Errorf("expected subcommand %q not found in root", name)
		}
	}
}

func TestNewRootCmd_HasPersistentConfigFlag(t *testing.T) {
	root := NewRootCmd()
	if root.PersistentFlags().Lookup("config") == nil {
		t.Error("expected --config persistent flag to be registered")
	}
}

func TestSetupLogger_DoesNotPanic(_ *testing.T) {
	for _, level := range []string{"debug", "info", "warn", "error"} {
		setupLogger(level)
	}
}

func TestSetupLogger_InvalidLevelFallsBackToInfo(_ *testing.T) {
	// Should not panic on invalid level.
	setupLogger("not-a-level")
}

func TestRequireConfig_FailsWhenNotInitialized(t *testing.T) {
	orig := activeCfg

	t.Cleanup(func() { activeCfg = orig })

	// Zero-value config has empty Paths.ModelPath → requireConfig returns error.
	activeCfg = config.Config{}

	_, err := requireConfig()
	if err == nil {
		t.Fatal("expected error when config is not loaded")
	}
}

func TestRequireConfig_SucceedsWhenLoaded(t *testing.T) {
	orig := activeCfg

	t.Cleanup(func() { activeCfg = orig })

	activeCfg = config.Config{
		Paths: config.PathsConfig{ModelPath: "/some/model/path"},
	}

	got, err := requireConfig()
	if err != nil {
		t.Fatalf("requireConfig returned unexpected error: %v", err)
	}

	if got.Paths.ModelPath != "/some/model/path" {
		t.Errorf("unexpected ModelPath: %q", got.Paths.ModelPath)
	}
}
