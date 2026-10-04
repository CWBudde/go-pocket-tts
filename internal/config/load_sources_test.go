package config

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/spf13/pflag"
)

// keySource describes one config key for the source tests: its flag, a
// non-default value that every source can carry, and how to read it back.
type keySource struct {
	key   string
	flag  string
	value string
	get   func(Config) any
}

var keySources = []keySource{
	{"paths.model_path", "paths-model-path", "/x/model.safetensors", func(c Config) any { return c.Paths.ModelPath }},
	{"paths.voice_path", "paths-voice-path", "/x/voice.bin", func(c Config) any { return c.Paths.VoicePath }},
	{"paths.onnx_manifest", "paths-onnx-manifest", "/x/manifest.json", func(c Config) any { return c.Paths.ONNXManifest }},
	{"paths.tokenizer_model", "paths-tokenizer-model", "/x/tok.model", func(c Config) any { return c.Paths.TokenizerModel }},
	{"paths.voice_manifest", "paths-voice-manifest", "/x/voices.json", func(c Config) any { return c.Paths.VoiceManifest }},
	{"runtime.threads", "runtime-threads", "7", func(c Config) any { return c.Runtime.Threads }},
	{"runtime.inter_op_threads", "runtime-inter-op-threads", "3", func(c Config) any { return c.Runtime.InterOpThreads }},
	{"runtime.workers", "runtime-workers", "5", func(c Config) any { return c.Runtime.Workers }},
	{"runtime.conv_workers", "conv-workers", "6", func(c Config) any { return c.Runtime.ConvWorkers }},
	{"runtime.ort_library_path", "runtime-ort-library-path", "/x/libort.so", func(c Config) any { return c.Runtime.ORTLibraryPath }},
	{"runtime.ort_version", "runtime-ort-version", "1.2.3", func(c Config) any { return c.Runtime.ORTVersion }},
	{"server.listen_addr", "server-listen-addr", ":7777", func(c Config) any { return c.Server.ListenAddr }},
	{"server.grpc_addr", "server-grpc-addr", ":7778", func(c Config) any { return c.Server.GRPCAddr }},
	{"server.workers", "workers", "16", func(c Config) any { return c.Server.Workers }},
	{"server.shutdown_timeout_secs", "shutdown-timeout", "11", func(c Config) any { return c.Server.ShutdownTimeout }},
	{"server.max_text_bytes", "max-text-bytes", "1234", func(c Config) any { return c.Server.MaxTextBytes }},
	{"server.request_timeout_secs", "request-timeout", "12", func(c Config) any { return c.Server.RequestTimeout }},
	{"server.default_voice", "default-voice", "/x/voice.safetensors", func(c Config) any { return c.Server.DefaultVoice }},
	{"server.languages", "server-languages", "german,english_2026-09", func(c Config) any { return strings.Join(c.Server.Languages, ",") }},
	{"server.max_languages", "server-max-languages", "3", func(c Config) any { return c.Server.MaxLanguages }},
	{"tts.backend", "backend", "cli", func(c Config) any { return c.TTS.Backend }},
	{"tts.voice", "tts-voice", "marius", func(c Config) any { return c.TTS.Voice }},
	{"tts.cli_path", "tts-cli-path", "/x/pocket-tts", func(c Config) any { return c.TTS.CLIPath }},
	{"tts.cli_config_path", "tts-cli-config-path", "/x/cfg.yaml", func(c Config) any { return c.TTS.CLIConfigPath }},
	{"tts.concurrency", "tts-concurrency", "3", func(c Config) any { return c.TTS.Concurrency }},
	{"tts.quiet", "tts-quiet", "false", func(c Config) any { return c.TTS.Quiet }},
	{"tts.temperature", "temperature", "0.5", func(c Config) any { return c.TTS.Temperature }},
	{"tts.eos_threshold", "eos-threshold", "-2.5", func(c Config) any { return c.TTS.EOSThreshold }},
	{"tts.max_steps", "max-steps", "99", func(c Config) any { return c.TTS.MaxSteps }},
	{"tts.language", "language", "german", func(c Config) any { return c.TTS.Language }},
	{"tts.sampler_decode_steps", "sampler-decode-steps", "4", func(c Config) any { return c.TTS.SamplerDecodeSteps }},
	{"log_level", "log-level", "debug", func(c Config) any { return c.LogLevel }},
}

// sourceEnvName derives POCKETTTS_<NAME> from a config key or flag name.
func sourceEnvName(name string) string {
	return "POCKETTTS_" + strings.ToUpper(strings.NewReplacer(".", "_", "-", "_").Replace(name))
}

// yamlFor renders key=value as a nested YAML document, quoting non-scalars.
func yamlFor(key, value string) string {
	scalar := strconv.Quote(value)

	_, floatErr := strconv.ParseFloat(value, 64)
	if floatErr == nil || value == "true" || value == "false" {
		scalar = value
	}

	parts := strings.Split(key, ".")

	var b strings.Builder

	for i, part := range parts[:len(parts)-1] {
		fmt.Fprintf(&b, "%s%s:\n", strings.Repeat("  ", i), part)
	}

	fmt.Fprintf(&b, "%s%s: %s\n", strings.Repeat("  ", len(parts)-1), parts[len(parts)-1], scalar)

	return b.String()
}

type loadInput struct {
	args  []string
	env   map[string]string
	file  string
	noCmd bool
}

func loadWith(t *testing.T, in loadInput) Config {
	t.Helper()

	cfg, err := tryLoad(t, in)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	return cfg
}

func tryLoad(t *testing.T, in loadInput) (Config, error) {
	t.Helper()

	for k, v := range in.env {
		t.Setenv(k, v)
	}

	defaults := DefaultConfig()
	opts := LoadOptions{Defaults: defaults}

	if !in.noCmd {
		fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
		fs.SetOutput(io.Discard)
		RegisterFlags(fs, defaults)

		err := fs.Parse(in.args)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}

		opts.Cmd = &fakeBinder{fs: fs}
	}

	if in.file != "" {
		opts.ConfigFile = filepath.Join(t.TempDir(), "pockettts.yaml")

		err := os.WriteFile(opts.ConfigFile, []byte(in.file), 0o644)
		if err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
	}

	return Load(opts)
}

// TestLoad_KeySources checks that every config key is reachable from the
// config file, the section-style env var (POCKETTTS_<SECTION>_<KEY>), the
// flag-style env var (POCKETTTS_<FLAG>) and its flag, with and without flags
// bound, as README documents.
func TestLoad_KeySources(t *testing.T) {
	for _, ks := range keySources {
		t.Run(ks.key, func(t *testing.T) {
			sectionEnv := sourceEnvName(ks.key)
			flagEnv := sourceEnvName(ks.flag)

			inputs := map[string]loadInput{
				"file":                         {file: yamlFor(ks.key, ks.value)},
				"file without flags":           {file: yamlFor(ks.key, ks.value), noCmd: true},
				"section env":                  {env: map[string]string{sectionEnv: ks.value}},
				"section env without flags":    {env: map[string]string{sectionEnv: ks.value}, noCmd: true},
				"flag-style env":               {env: map[string]string{flagEnv: ks.value}},
				"flag-style env without flags": {env: map[string]string{flagEnv: ks.value}, noCmd: true},
				"flag":                         {args: []string{"--" + ks.flag + "=" + ks.value}},
			}

			for name, in := range inputs {
				t.Run(name, func(t *testing.T) {
					got := fmt.Sprint(ks.get(loadWith(t, in)))
					if got != ks.value {
						t.Errorf("%s = %s; want %s", ks.key, got, ks.value)
					}
				})
			}
		})
	}
}

// TestLoad_KeySources_Precedence checks flags > env > config file > defaults.
func TestLoad_KeySources_Precedence(t *testing.T) {
	file := "tts:\n  max_steps: 10\n"
	sectionEnv := map[string]string{"POCKETTTS_TTS_MAX_STEPS": "20"}

	tests := []struct {
		name string
		in   loadInput
		want int
	}{
		{"default", loadInput{}, 256},
		{"file beats default", loadInput{file: file}, 10},
		{"env beats file", loadInput{file: file, env: sectionEnv}, 20},
		{"section env beats flag-style env", loadInput{env: map[string]string{
			"POCKETTTS_TTS_MAX_STEPS": "20",
			"POCKETTTS_MAX_STEPS":     "21",
		}}, 20},
		{"flag beats env and file", loadInput{file: file, env: sectionEnv, args: []string{"--max-steps=30"}}, 30},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := loadWith(t, tc.in).TTS.MaxSteps; got != tc.want {
				t.Errorf("TTS.MaxSteps = %d; want %d", got, tc.want)
			}
		})
	}
}

// TestLoad_ORTLib covers the extra names of runtime.ort_library_path.
func TestLoad_ORTLib(t *testing.T) {
	tests := []struct {
		name string
		in   loadInput
		want string
	}{
		{"POCKETTTS_ORT_LIB", loadInput{env: map[string]string{"POCKETTTS_ORT_LIB": "/a"}}, "/a"},
		{"ORT_LIBRARY_PATH", loadInput{env: map[string]string{"ORT_LIBRARY_PATH": "/b"}}, "/b"},
		{"ORT_LIBRARY_PATH without flags", loadInput{env: map[string]string{"ORT_LIBRARY_PATH": "/b"}, noCmd: true}, "/b"},
		{"section env beats POCKETTTS_ORT_LIB", loadInput{env: map[string]string{
			"POCKETTTS_RUNTIME_ORT_LIBRARY_PATH": "/c",
			"POCKETTTS_ORT_LIB":                  "/a",
		}}, "/c"},
		{"POCKETTTS_ORT_LIB beats ORT_LIBRARY_PATH", loadInput{env: map[string]string{
			"POCKETTTS_ORT_LIB": "/a",
			"ORT_LIBRARY_PATH":  "/b",
		}}, "/a"},
		{"--ort-lib", loadInput{args: []string{"--ort-lib=/d"}}, "/d"},
		{"--ort-lib beats env", loadInput{args: []string{"--ort-lib=/d"}, env: map[string]string{"POCKETTTS_ORT_LIB": "/a"}}, "/d"},
		{"long flag beats --ort-lib", loadInput{args: []string{"--ort-lib=/d", "--runtime-ort-library-path=/e"}}, "/e"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := loadWith(t, tc.in).Runtime.ORTLibraryPath; got != tc.want {
				t.Errorf("Runtime.ORTLibraryPath = %q; want %q", got, tc.want)
			}
		})
	}
}

// TestKeyBindings_CoverConfigAndFlags guards against a new Config field or
// flag that Load would silently ignore.
func TestKeyBindings_CoverConfigAndFlags(t *testing.T) {
	bound := map[string]bool{}
	boundFlags := map[string]bool{}

	for _, b := range keyBindings {
		bound[b.key] = true
		boundFlags[b.flag] = true
	}

	for _, key := range configLeafKeys(reflect.TypeFor[Config](), "") {
		if !bound[key] {
			t.Errorf("config key %q has no entry in keyBindings", key)
		}
	}

	for _, ks := range keySources {
		if !bound[ks.key] {
			t.Errorf("keySources entry %q has no entry in keyBindings", ks.key)
		}
	}

	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	RegisterFlags(fs, DefaultConfig())

	aliases := map[string]bool{"lsd-steps": true}
	for _, fa := range flagAliases {
		aliases[fa.alias] = true
	}

	fs.VisitAll(func(f *pflag.Flag) {
		if !boundFlags[f.Name] && !aliases[f.Name] {
			t.Errorf("flag --%s is neither bound to a key nor a known alias", f.Name)
		}
	})
}

// configLeafKeys returns the dotted mapstructure keys of every leaf field.
func configLeafKeys(typ reflect.Type, prefix string) []string {
	var keys []string

	for i := range typ.NumField() {
		field := typ.Field(i)

		tag := field.Tag.Get("mapstructure")
		if tag == "" || tag == "-" {
			continue
		}

		if field.Type.Kind() == reflect.Struct {
			keys = append(keys, configLeafKeys(field.Type, prefix+tag+".")...)

			continue
		}

		keys = append(keys, prefix+tag)
	}

	return keys
}
