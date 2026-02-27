package main

import (
	"bytes"
	"strings"
	"testing"
)

// --- NewRootCmd ---

func TestNewRootCmd_Use(t *testing.T) {
	cmd := NewRootCmd()
	if cmd.Use != "pockettts-tools" {
		t.Errorf("Use = %q; want %q", cmd.Use, "pockettts-tools")
	}
}

func TestNewRootCmd_HasSubcommands(t *testing.T) {
	cmd := NewRootCmd()

	names := make(map[string]bool)
	for _, sub := range cmd.Commands() {
		names[sub.Name()] = true
	}

	for _, want := range []string{"model", "export-voice"} {
		if !names[want] {
			t.Errorf("subcommand %q not registered", want)
		}
	}
}

func TestNewRootCmd_ModelHasExportSubcommand(t *testing.T) {
	cmd := NewRootCmd()
	var modelCmd interface {
		Commands() []*interface{ Name() string }
	}
	_ = modelCmd

	for _, sub := range cmd.Commands() {
		if sub.Name() == "model" {
			subNames := make(map[string]bool)
			for _, s := range sub.Commands() {
				subNames[s.Name()] = true
			}

			if !subNames["export"] {
				t.Error("model subcommand 'export' not registered")
			}

			return
		}
	}

	t.Error("'model' command not found")
}

func TestNewRootCmd_PersistentFlagConfig(t *testing.T) {
	cmd := NewRootCmd()

	f := cmd.PersistentFlags().Lookup("config")
	if f == nil {
		t.Fatal("--config flag not registered")
	}

	if f.DefValue != "" {
		t.Errorf("--config default = %q; want empty string", f.DefValue)
	}
}

func TestNewRootCmd_PersistentFlagsIncludeBackend(t *testing.T) {
	cmd := NewRootCmd()

	knownFlags := []string{"backend", "log-level", "workers", "paths-model-path"}
	for _, name := range knownFlags {
		if cmd.PersistentFlags().Lookup(name) == nil {
			t.Errorf("persistent flag %q not registered", name)
		}
	}
}

// --- requireConfig ---

func TestRequireConfig_FailsWhenNotLoaded(t *testing.T) {
	// Reset global state.
	activeCfg.Paths.ModelPath = ""

	_, err := requireConfig()
	if err == nil {
		t.Error("requireConfig() = nil; want error when config not loaded")
	}

	if !strings.Contains(err.Error(), "not loaded") {
		t.Errorf("error %q does not mention 'not loaded'", err.Error())
	}
}

func TestRequireConfig_SucceedsWhenLoaded(t *testing.T) {
	activeCfg.Paths.ModelPath = "models/model.onnx"

	t.Cleanup(func() { activeCfg.Paths.ModelPath = "" })

	cfg, err := requireConfig()
	if err != nil {
		t.Fatalf("requireConfig() error = %v", err)
	}

	if cfg.Paths.ModelPath != "models/model.onnx" {
		t.Errorf("ModelPath = %q; want %q", cfg.Paths.ModelPath, "models/model.onnx")
	}
}

// --- model export command flags ---

func TestModelExportCmd_Flags(t *testing.T) {
	cmd := newModelExportCmd()

	flags := []struct {
		name     string
		defValue string
	}{
		{"models-dir", "models"},
		{"out-dir", "models/onnx"},
		{"variant", "b6369a24"},
		{"python-bin", ""},
	}

	for _, f := range flags {
		flag := cmd.Flags().Lookup(f.name)
		if flag == nil {
			t.Errorf("flag %q not registered", f.name)
			continue
		}

		if flag.DefValue != f.defValue {
			t.Errorf("flag %q default = %q; want %q", f.name, flag.DefValue, f.defValue)
		}
	}

	// Bool flag
	int8Flag := cmd.Flags().Lookup("int8")
	if int8Flag == nil {
		t.Error("flag 'int8' not registered")
	} else if int8Flag.DefValue != "false" {
		t.Errorf("flag 'int8' default = %q; want %q", int8Flag.DefValue, "false")
	}
}

func TestModelExportCmd_Use(t *testing.T) {
	cmd := newModelExportCmd()
	if cmd.Use != "export" {
		t.Errorf("Use = %q; want %q", cmd.Use, "export")
	}
}

// --- export-voice command flags ---

func TestExportVoiceCmd_Flags(t *testing.T) {
	cmd := newExportVoiceCmd()

	flags := []struct {
		name     string
		defValue string
	}{
		{"audio", ""},
		{"out", ""},
		{"id", "custom-voice"},
		{"license", "unknown"},
	}

	for _, f := range flags {
		flag := cmd.Flags().Lookup(f.name)
		if flag == nil {
			t.Errorf("flag %q not registered", f.name)
			continue
		}

		if flag.DefValue != f.defValue {
			t.Errorf("flag %q default = %q; want %q", f.name, flag.DefValue, f.defValue)
		}
	}
}

func TestExportVoiceCmd_Use(t *testing.T) {
	cmd := newExportVoiceCmd()
	if cmd.Use != "export-voice" {
		t.Errorf("Use = %q; want %q", cmd.Use, "export-voice")
	}
}

func TestExportVoiceCmd_RequiresAudioFlag(t *testing.T) {
	// Simulate config loaded so requireConfig() passes.
	activeCfg.Paths.ModelPath = "models/model.onnx"

	t.Cleanup(func() { activeCfg.Paths.ModelPath = "" })

	cmd := NewRootCmd()
	cmd.SilenceUsage = true
	cmd.SetArgs([]string{"export-voice", "--out=/tmp/out.safetensors"})
	var errBuf bytes.Buffer
	cmd.SetErr(&errBuf)

	err := cmd.Execute()
	if err == nil {
		t.Error("Execute() = nil; want error when --audio is missing")
	}

	if !strings.Contains(err.Error(), "--audio") {
		t.Errorf("error %q does not mention '--audio'", err.Error())
	}
}

func TestExportVoiceCmd_RequiresOutFlag(t *testing.T) {
	activeCfg.Paths.ModelPath = "models/model.onnx"

	t.Cleanup(func() { activeCfg.Paths.ModelPath = "" })

	cmd := NewRootCmd()
	cmd.SilenceUsage = true
	cmd.SetArgs([]string{"export-voice", "--audio=/tmp/voice.wav"})
	var errBuf bytes.Buffer
	cmd.SetErr(&errBuf)

	err := cmd.Execute()
	if err == nil {
		t.Error("Execute() = nil; want error when --out is missing")
	}

	if !strings.Contains(err.Error(), "--out") {
		t.Errorf("error %q does not mention '--out'", err.Error())
	}
}
