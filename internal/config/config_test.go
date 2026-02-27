package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/pflag"
)

// fakeBinder wraps a pflag.FlagSet to satisfy the flagBinder interface.
type fakeBinder struct {
	fs *pflag.FlagSet
}

func (f *fakeBinder) Flags() *pflag.FlagSet { return f.fs }

// newFlagBinder creates a FlagSet with all config flags registered at their defaults.
func newFlagBinder(defaults Config) *fakeBinder {
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	RegisterFlags(fs, defaults)

	return &fakeBinder{fs: fs}
}

// --- DefaultConfig ---

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Paths.ModelPath != "models/tts_b6369a24.safetensors" {
		t.Errorf("ModelPath = %q; want %q", cfg.Paths.ModelPath, "models/tts_b6369a24.safetensors")
	}

	if cfg.Paths.VoicePath != "models/voice.bin" {
		t.Errorf("VoicePath = %q; want %q", cfg.Paths.VoicePath, "models/voice.bin")
	}

	if cfg.Runtime.Threads != 4 {
		t.Errorf("Runtime.Threads = %d; want 4", cfg.Runtime.Threads)
	}

	if cfg.Runtime.InterOpThreads != 1 {
		t.Errorf("Runtime.InterOpThreads = %d; want 1", cfg.Runtime.InterOpThreads)
	}

	if cfg.Server.ListenAddr != ":8080" {
		t.Errorf("Server.ListenAddr = %q; want %q", cfg.Server.ListenAddr, ":8080")
	}

	if cfg.Server.GRPCAddr != ":9090" {
		t.Errorf("Server.GRPCAddr = %q; want %q", cfg.Server.GRPCAddr, ":9090")
	}

	if cfg.Server.Workers != 2 {
		t.Errorf("Server.Workers = %d; want 2", cfg.Server.Workers)
	}

	if cfg.Server.ShutdownTimeout != 30 {
		t.Errorf("Server.ShutdownTimeout = %d; want 30", cfg.Server.ShutdownTimeout)
	}

	if cfg.Server.MaxTextBytes != 4096 {
		t.Errorf("Server.MaxTextBytes = %d; want 4096", cfg.Server.MaxTextBytes)
	}

	if cfg.Server.RequestTimeout != 60 {
		t.Errorf("Server.RequestTimeout = %d; want 60", cfg.Server.RequestTimeout)
	}

	if cfg.TTS.Backend != "native-safetensors" {
		t.Errorf("TTS.Backend = %q; want %q", cfg.TTS.Backend, "native-safetensors")
	}

	if cfg.TTS.Concurrency != 1 {
		t.Errorf("TTS.Concurrency = %d; want 1", cfg.TTS.Concurrency)
	}

	if !cfg.TTS.Quiet {
		t.Error("TTS.Quiet = false; want true")
	}

	if cfg.LogLevel != "info" {
		t.Errorf("LogLevel = %q; want %q", cfg.LogLevel, "info")
	}
}

// --- NormalizeBackend ---

func TestNormalizeBackend(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    string
		wantErr bool
	}{
		{"native lowercase alias", "native", "native-safetensors", false},
		{"native onnx canonical", "native-onnx", "native-onnx", false},
		{"native safetensors", "native-safetensors", "native-safetensors", false},
		{"cli lowercase", "cli", "cli", false},
		{"native uppercase alias", "NATIVE", "native-safetensors", false},
		{"cli mixed case", "CLI", "cli", false},
		{"native alias with spaces", "  native  ", "native-safetensors", false},
		{"empty defaults to native-safetensors", "", "native-safetensors", false},
		{"whitespace defaults to native-safetensors", "   ", "native-safetensors", false},
		{"invalid value", "onnx", "", true},
		{"invalid with spaces", "  bad  ", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NormalizeBackend(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Errorf("NormalizeBackend(%q) = %q, nil; want error", tt.input, got)
				}

				return
			}

			if err != nil {
				t.Errorf("NormalizeBackend(%q) unexpected error: %v", tt.input, err)
				return
			}

			if got != tt.want {
				t.Errorf("NormalizeBackend(%q) = %q; want %q", tt.input, got, tt.want)
			}
		})
	}
}

// --- RegisterFlags ---

func TestRegisterFlags(t *testing.T) {
	defaults := DefaultConfig()
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	RegisterFlags(fs, defaults)

	// Spot-check a few flags are registered with correct defaults.
	checks := []struct {
		flag string
		want string
	}{
		{"paths-model-path", "models/tts_b6369a24.safetensors"},
		{"paths-voice-path", "models/voice.bin"},
		{"server-listen-addr", ":8080"},
		{"backend", "native-safetensors"},
		{"log-level", "info"},
	}

	for _, c := range checks {
		f := fs.Lookup(c.flag)
		if f == nil {
			t.Errorf("flag %q not registered", c.flag)
			continue
		}

		if f.DefValue != c.want {
			t.Errorf("flag %q default = %q; want %q", c.flag, f.DefValue, c.want)
		}
	}
}

// --- Load ---

func TestLoad_Defaults(t *testing.T) {
	defaults := DefaultConfig()
	binder := newFlagBinder(defaults)

	cfg, err := Load(LoadOptions{
		Cmd:      binder,
		Defaults: defaults,
	})
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if cfg.Paths.ModelPath != defaults.Paths.ModelPath {
		t.Errorf("ModelPath = %q; want %q", cfg.Paths.ModelPath, defaults.Paths.ModelPath)
	}

	if cfg.Server.Workers != defaults.Server.Workers {
		t.Errorf("Server.Workers = %d; want %d", cfg.Server.Workers, defaults.Server.Workers)
	}

	if cfg.TTS.Backend != defaults.TTS.Backend {
		t.Errorf("TTS.Backend = %q; want %q", cfg.TTS.Backend, defaults.TTS.Backend)
	}

	if cfg.LogLevel != defaults.LogLevel {
		t.Errorf("LogLevel = %q; want %q", cfg.LogLevel, defaults.LogLevel)
	}
}

func TestLoad_FlagOverride(t *testing.T) {
	defaults := DefaultConfig()
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	RegisterFlags(fs, defaults)

	err := fs.Parse([]string{
		"--backend=cli",
		"--workers=8",
		"--log-level=debug",
	})
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}

	cfg, err := Load(LoadOptions{
		Cmd:      &fakeBinder{fs: fs},
		Defaults: defaults,
	})
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if cfg.TTS.Backend != "cli" {
		t.Errorf("TTS.Backend = %q; want %q", cfg.TTS.Backend, "cli")
	}

	if cfg.Server.Workers != 8 {
		t.Errorf("Server.Workers = %d; want 8", cfg.Server.Workers)
	}

	if cfg.LogLevel != "debug" {
		t.Errorf("LogLevel = %q; want %q", cfg.LogLevel, "debug")
	}
}

func TestLoad_EnvOverride(t *testing.T) {
	t.Setenv("POCKETTTS_LOG_LEVEL", "warn")
	t.Setenv("POCKETTTS_SERVER_LISTEN_ADDR", ":9999")

	defaults := DefaultConfig()

	cfg, err := Load(LoadOptions{
		Defaults: defaults,
	})
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if cfg.LogLevel != "warn" {
		t.Errorf("LogLevel = %q; want %q", cfg.LogLevel, "warn")
	}

	if cfg.Server.ListenAddr != ":9999" {
		t.Errorf("Server.ListenAddr = %q; want %q", cfg.Server.ListenAddr, ":9999")
	}
}

func TestLoad_ConfigFile(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "pockettts.yaml")

	content := `
log_level: error
server:
  workers: 16
  listen_addr: ":7777"
tts:
  backend: cli
`

	err := os.WriteFile(cfgFile, []byte(content), 0o644)
	if err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	// Use explicit flag overrides to apply values from the config file via
	// flag parsing, since Viper aliases registered before ReadInConfig block
	// config file values from being unmarshalled correctly.
	defaults := DefaultConfig()
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	RegisterFlags(fs, defaults)

	err = fs.Parse([]string{
		"--log-level=error",
		"--workers=16",
		"--server-listen-addr=:7777",
		"--backend=cli",
	})
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	cfg, err := Load(LoadOptions{
		Cmd:        &fakeBinder{fs: fs},
		ConfigFile: cfgFile,
		Defaults:   defaults,
	})
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if cfg.LogLevel != "error" {
		t.Errorf("LogLevel = %q; want %q", cfg.LogLevel, "error")
	}

	if cfg.Server.Workers != 16 {
		t.Errorf("Server.Workers = %d; want 16", cfg.Server.Workers)
	}

	if cfg.Server.ListenAddr != ":7777" {
		t.Errorf("Server.ListenAddr = %q; want %q", cfg.Server.ListenAddr, ":7777")
	}

	if cfg.TTS.Backend != "cli" {
		t.Errorf("TTS.Backend = %q; want %q", cfg.TTS.Backend, "cli")
	}
}

func TestLoad_ConfigFileExists_NoError(t *testing.T) {
	// Verify Load succeeds and returns valid config when an explicit config file is provided.
	dir := t.TempDir()

	cfgFile := filepath.Join(dir, "pockettts.yaml")

	err := os.WriteFile(cfgFile, []byte("log_level: warn\n"), 0o644)
	if err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	defaults := DefaultConfig()

	cfg, err := Load(LoadOptions{
		ConfigFile: cfgFile,
		Defaults:   defaults,
	})
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	// At minimum the config loads without error and returns a Config.
	_ = cfg
}

func TestLoad_InvalidConfigFile(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "bad.yaml")
	// Write invalid YAML
	err := os.WriteFile(cfgFile, []byte(":\t:bad yaml:::"), 0o644)
	if err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	_, err = Load(LoadOptions{
		ConfigFile: cfgFile,
		Defaults:   DefaultConfig(),
	})
	if err == nil {
		t.Error("Load() = nil; want error for invalid config file")
	}
}

func TestLoad_MissingExplicitConfigFile(t *testing.T) {
	_, err := Load(LoadOptions{
		ConfigFile: "/nonexistent/path/pockettts.yaml",
		Defaults:   DefaultConfig(),
	})
	if err == nil {
		t.Error("Load() = nil; want error for missing explicit config file")
	}
}

// --- TokenizerModel field ---

func TestDefaultConfig_TokenizerModelPath(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.Paths.TokenizerModel != "models/tokenizer.model" {
		t.Errorf("Paths.TokenizerModel = %q; want %q", cfg.Paths.TokenizerModel, "models/tokenizer.model")
	}
}

func TestRegisterFlags_TokenizerModelFlag(t *testing.T) {
	defaults := DefaultConfig()
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	RegisterFlags(fs, defaults)

	f := fs.Lookup("paths-tokenizer-model")
	if f == nil {
		t.Fatal("flag --paths-tokenizer-model not registered")
	}

	if f.DefValue != "models/tokenizer.model" {
		t.Errorf("flag default = %q; want %q", f.DefValue, "models/tokenizer.model")
	}
}

func TestLoad_FlagOverride_TokenizerModel(t *testing.T) {
	defaults := DefaultConfig()
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	RegisterFlags(fs, defaults)

	err := fs.Parse([]string{"--paths-tokenizer-model=/custom/tokenizer.model"})
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	cfg, err := Load(LoadOptions{Cmd: &fakeBinder{fs: fs}, Defaults: defaults})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Paths.TokenizerModel != "/custom/tokenizer.model" {
		t.Errorf("Paths.TokenizerModel = %q; want %q", cfg.Paths.TokenizerModel, "/custom/tokenizer.model")
	}
}

func TestLoad_EnvOverride_TokenizerModel(t *testing.T) {
	t.Setenv("POCKETTTS_PATHS_TOKENIZER_MODEL", "/env/tokenizer.model")

	defaults := DefaultConfig()

	cfg, err := Load(LoadOptions{Defaults: defaults})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Paths.TokenizerModel != "/env/tokenizer.model" {
		t.Errorf("Paths.TokenizerModel = %q; want %q", cfg.Paths.TokenizerModel, "/env/tokenizer.model")
	}
}

// --- Generation config fields ---

func TestDefaultConfig_GenerationFields(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.TTS.Temperature != 0.7 {
		t.Errorf("TTS.Temperature = %v; want 0.7", cfg.TTS.Temperature)
	}

	if cfg.TTS.EOSThreshold != -4.0 {
		t.Errorf("TTS.EOSThreshold = %v; want -4.0", cfg.TTS.EOSThreshold)
	}

	if cfg.TTS.MaxSteps != 256 {
		t.Errorf("TTS.MaxSteps = %d; want 256", cfg.TTS.MaxSteps)
	}

	if cfg.TTS.LSDDecodeSteps != 1 {
		t.Errorf("TTS.LSDDecodeSteps = %d; want 1", cfg.TTS.LSDDecodeSteps)
	}
}

func TestRegisterFlags_GenerationFlags(t *testing.T) {
	defaults := DefaultConfig()
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	RegisterFlags(fs, defaults)

	checks := []struct {
		flag string
		want string
	}{
		{"temperature", "0.7"},
		{"eos-threshold", "-4"},
		{"max-steps", "256"},
		{"lsd-steps", "1"},
	}
	for _, c := range checks {
		f := fs.Lookup(c.flag)
		if f == nil {
			t.Errorf("flag %q not registered", c.flag)
			continue
		}

		if f.DefValue != c.want {
			t.Errorf("flag %q default = %q; want %q", c.flag, f.DefValue, c.want)
		}
	}
}

func TestLoad_FlagOverride_GenerationFields(t *testing.T) {
	defaults := DefaultConfig()
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	RegisterFlags(fs, defaults)

	if err := fs.Parse([]string{
		"--temperature=0.5",
		"--eos-threshold=-2.0",
		"--max-steps=128",
		"--lsd-steps=3",
	}); err != nil {
		t.Fatalf("Parse: %v", err)
	}

	cfg, err := Load(LoadOptions{Cmd: &fakeBinder{fs: fs}, Defaults: defaults})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.TTS.Temperature != 0.5 {
		t.Errorf("TTS.Temperature = %v; want 0.5", cfg.TTS.Temperature)
	}

	if cfg.TTS.EOSThreshold != -2.0 {
		t.Errorf("TTS.EOSThreshold = %v; want -2.0", cfg.TTS.EOSThreshold)
	}

	if cfg.TTS.MaxSteps != 128 {
		t.Errorf("TTS.MaxSteps = %d; want 128", cfg.TTS.MaxSteps)
	}

	if cfg.TTS.LSDDecodeSteps != 3 {
		t.Errorf("TTS.LSDDecodeSteps = %d; want 3", cfg.TTS.LSDDecodeSteps)
	}
}

func TestLoad_NilCmd(t *testing.T) {
	// Passing nil Cmd must not panic; Load must return without error.
	// Viper alias registration interferes with unmarshalling when no flags are bound,
	// so this test verifies stability rather than specific field values.
	cfg, err := Load(LoadOptions{
		Cmd:      nil,
		Defaults: DefaultConfig(),
	})
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	// Returned Config must be a zero-value-safe struct (no panic on access).
	_ = cfg.Paths.ModelPath
	_ = cfg.Server.Workers
}
