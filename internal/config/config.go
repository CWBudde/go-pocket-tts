package config

import (
	"errors"
	"fmt"
	"os"
	"path"
	"strings"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

type Config struct {
	Paths    PathsConfig   `mapstructure:"paths"`
	Runtime  RuntimeConfig `mapstructure:"runtime"`
	Server   ServerConfig  `mapstructure:"server"`
	TTS      TTSConfig     `mapstructure:"tts"`
	LogLevel string        `mapstructure:"log_level"`

	// Model is the model config of TTS.Language, or of TTS.ModelConfigPath
	// when set. Load resolves it; it is not a config key itself.
	Model *modelcfg.ModelConfig `mapstructure:"-"`
}

type PathsConfig struct {
	ModelPath      string `mapstructure:"model_path"`
	VoicePath      string `mapstructure:"voice_path"`
	ONNXManifest   string `mapstructure:"onnx_manifest"`
	TokenizerModel string `mapstructure:"tokenizer_model"`
	VoiceManifest  string `mapstructure:"voice_manifest"`
}

type RuntimeConfig struct {
	Threads        int    `mapstructure:"threads"`
	InterOpThreads int    `mapstructure:"inter_op_threads"`
	Workers        int    `mapstructure:"workers"`
	ConvWorkers    int    `mapstructure:"conv_workers"`
	ORTLibraryPath string `mapstructure:"ort_library_path"`
	ORTVersion     string `mapstructure:"ort_version"`
}

type ServerConfig struct {
	ListenAddr      string `mapstructure:"listen_addr"`
	GRPCAddr        string `mapstructure:"grpc_addr"`
	Workers         int    `mapstructure:"workers"`
	ShutdownTimeout int    `mapstructure:"shutdown_timeout_secs"`
	MaxTextBytes    int    `mapstructure:"max_text_bytes"`
	RequestTimeout  int    `mapstructure:"request_timeout_secs"`
}

type TTSConfig struct {
	Backend string `mapstructure:"backend"`
	// Language selects the embedded model config (upstream --language) and,
	// unless set explicitly, the model, tokenizer and voice manifest paths;
	// see PathsForLanguage.
	Language string `mapstructure:"language"`
	// ModelConfigPath is a custom model config file (upstream --config). It
	// cannot be combined with an explicit Language and needs explicit model
	// and tokenizer paths.
	ModelConfigPath string  `mapstructure:"model_config"`
	Voice           string  `mapstructure:"voice"`
	CLIPath         string  `mapstructure:"cli_path"`
	CLIConfigPath   string  `mapstructure:"cli_config_path"`
	Concurrency     int     `mapstructure:"concurrency"`
	Quiet           bool    `mapstructure:"quiet"`
	Temperature     float64 `mapstructure:"temperature"`
	EOSThreshold    float64 `mapstructure:"eos_threshold"`
	MaxSteps        int     `mapstructure:"max_steps"`
	// SamplerDecodeSteps is the number of sampler integration steps per latent
	// frame (upstream --sampler-decode-steps, formerly --lsd-decode-steps). The
	// deprecated key tts.lsd_decode_steps, the env vars
	// POCKETTTS_TTS_LSD_DECODE_STEPS / POCKETTTS_LSD_STEPS and the hidden
	// --lsd-steps flag are still accepted; see keyBindings and
	// applyDeprecatedSamplerDecodeSteps.
	SamplerDecodeSteps int `mapstructure:"sampler_decode_steps"`
}

type LoadOptions struct {
	Cmd        flagBinder
	ConfigFile string
	Defaults   Config
}

type flagBinder interface {
	Flags() *pflag.FlagSet
}

func DefaultConfig() Config {
	return Config{
		Paths: PathsConfig{
			ModelPath:      "models/tts_b6369a24.safetensors",
			VoicePath:      "models/voice.bin",
			ONNXManifest:   "models/onnx/manifest.json",
			TokenizerModel: "models/tokenizer.model",
			VoiceManifest:  "voices/manifest.json",
		},
		Runtime: RuntimeConfig{
			Threads:        4,
			InterOpThreads: 1,
			Workers:        0,
			ConvWorkers:    2,
			ORTLibraryPath: "",
			ORTVersion:     "",
		},
		Server: ServerConfig{
			ListenAddr:      ":8080",
			GRPCAddr:        ":9090",
			Workers:         2,
			ShutdownTimeout: 30,
			MaxTextBytes:    4096,
			RequestTimeout:  60,
		},
		TTS: TTSConfig{
			Backend:            BackendNative,
			Language:           DefaultLanguage,
			ModelConfigPath:    "",
			Voice:              "",
			CLIPath:            "",
			CLIConfigPath:      "",
			Concurrency:        1,
			Quiet:              true,
			Temperature:        0.3, // DefaultLanguage's default_temperature; Load uses the model's
			EOSThreshold:       -4.0,
			MaxSteps:           256,
			SamplerDecodeSteps: 1,
		},
		LogLevel: "info",
	}
}

func RegisterFlags(fs *pflag.FlagSet, defaults Config) {
	fs.String("paths-model-path", defaults.Paths.ModelPath,
		"Path to model file (.safetensors for native, .onnx for native-onnx; follows --language unless set)")
	fs.String("paths-voice-path", defaults.Paths.VoicePath, "Path to voice/profile asset")
	fs.String("paths-onnx-manifest", defaults.Paths.ONNXManifest, "Path to ONNX model manifest JSON")
	fs.String("paths-tokenizer-model", defaults.Paths.TokenizerModel,
		"Path to SentencePiece tokenizer model (follows --language unless set)")
	fs.String("paths-voice-manifest", defaults.Paths.VoiceManifest,
		"Path to the voice manifest JSON (follows --language unless set)")
	fs.Int("runtime-threads", defaults.Runtime.Threads, "Inference thread count (ONNX intra-op for native-onnx backend)")
	fs.Int("runtime-inter-op-threads", defaults.Runtime.InterOpThreads, "Inter-op thread count (ONNX-only, native-onnx backend)")
	fs.Int(
		"runtime-workers",
		defaults.Runtime.Workers,
		"Parallel goroutines for tensor kernels (Linear/MatMul/Attention path); 0 falls back to --conv-workers",
	)
	fs.Int("conv-workers", defaults.Runtime.ConvWorkers, "Parallel goroutines for Conv1D/ConvTranspose1D (1 = sequential, default 2)")
	fs.String("runtime-ort-library-path", defaults.Runtime.ORTLibraryPath, "Path to ONNX Runtime shared library")
	fs.String("ort-lib", defaults.Runtime.ORTLibraryPath, "Path to ONNX Runtime shared library (alias for --runtime-ort-library-path)")
	fs.String("runtime-ort-version", defaults.Runtime.ORTVersion, "Expected ONNX Runtime version")
	fs.String("server-listen-addr", defaults.Server.ListenAddr, "HTTP listen address")
	fs.String("server-grpc-addr", defaults.Server.GRPCAddr, "gRPC listen address")
	fs.Int("workers", defaults.Server.Workers, "Max concurrent pocket-tts subprocesses for serve command")
	fs.Int("shutdown-timeout", defaults.Server.ShutdownTimeout, "Graceful shutdown drain timeout in seconds")
	fs.Int("max-text-bytes", defaults.Server.MaxTextBytes, "Maximum POST /tts text size in bytes")
	fs.Int("request-timeout", defaults.Server.RequestTimeout, "Per-request synthesis timeout in seconds")
	fs.String(
		"backend",
		defaults.TTS.Backend,
		"Synthesis backend (native-safetensors|native-onnx|cli; native is alias for native-safetensors)",
	)
	fs.String("language", defaults.TTS.Language,
		"Model language config ("+strings.Join(modelcfg.Languages(), ", ")+"); incompatible with --model-config")
	fs.String("model-config", defaults.TTS.ModelConfigPath,
		"Custom model config .yaml (upstream format); needs --paths-model-path and --paths-tokenizer-model")
	fs.String("tts-voice", defaults.TTS.Voice, "Voice name or .safetensors file path")
	fs.String("tts-cli-path", defaults.TTS.CLIPath, "Path to pocket-tts executable")
	fs.String("tts-cli-config-path", defaults.TTS.CLIConfigPath, "Path to pocket-tts config file")
	fs.Int("tts-concurrency", defaults.TTS.Concurrency, "Max concurrent pocket-tts subprocesses")
	fs.Bool("tts-quiet", defaults.TTS.Quiet, "Pass --quiet to pocket-tts generate")
	fs.Float64("temperature", defaults.TTS.Temperature,
		"Noise temperature for flow sampling (follows the model config's default_temperature unless set)")
	fs.Float64("eos-threshold", defaults.TTS.EOSThreshold, "Raw logit threshold for EOS detection")
	fs.Int("max-steps", defaults.TTS.MaxSteps, "Maximum autoregressive generation steps")
	fs.Int("sampler-decode-steps", defaults.TTS.SamplerDecodeSteps, "Sampler integration steps per latent frame")
	registerDeprecatedIntFlag(fs, "lsd-steps", defaults.TTS.SamplerDecodeSteps, "sampler-decode-steps")
	fs.String("log-level", defaults.LogLevel, "Log level (debug|info|warn|error)")
}

// registerDeprecatedIntFlag registers name as a hidden, deprecated alias of
// replacement. pflag prints a deprecation notice when the flag is used; the
// value itself is applied in Load.
func registerDeprecatedIntFlag(fs *pflag.FlagSet, name string, value int, replacement string) {
	fs.Int(name, value, "Deprecated: use --"+replacement)

	// MarkDeprecated also hides the flag from usage output. It only fails for
	// an unknown flag or an empty message, neither of which can happen here.
	_ = fs.MarkDeprecated(name, "use --"+replacement)
}

func Load(opts LoadOptions) (Config, error) {
	v := viper.New()

	setDefaults(v, opts.Defaults)

	var flags *pflag.FlagSet
	if opts.Cmd != nil {
		flags = opts.Cmd.Flags()
	}

	err := bindKeys(v, flags)
	if err != nil {
		return Config{}, err
	}

	//nolint:nestif // Distinguish explicit config-file errors from optional auto-discovery behavior.
	if opts.ConfigFile != "" {
		v.SetConfigFile(opts.ConfigFile)

		err := v.ReadInConfig()
		if err != nil {
			return Config{}, fmt.Errorf("read config file: %w", err)
		}
	} else {
		v.SetConfigName("pockettts")
		v.AddConfigPath(".")

		err := v.ReadInConfig()
		if err != nil {
			var configFileNotFoundError viper.ConfigFileNotFoundError
			if !errors.As(err, &configFileNotFoundError) {
				return Config{}, fmt.Errorf("read config file: %w", err)
			}
		}
	}

	err = applyFlagAliases(v, flags)
	if err != nil {
		return Config{}, err
	}

	err = applyDeprecatedSamplerDecodeSteps(v, flags)
	if err != nil {
		return Config{}, err
	}

	var cfg Config

	err = v.Unmarshal(&cfg)
	if err != nil {
		return Config{}, fmt.Errorf("decode config: %w", err)
	}

	err = resolveModel(v, flags, &cfg)
	if err != nil {
		return Config{}, err
	}

	applyModelDefaults(v, flags, &cfg)

	return cfg, nil
}

// applyModelDefaults fills the generation settings the user did not set
// explicitly from the resolved model config. Upstream: --temperature defaults
// to None, which means the model's default_temperature.
func applyModelDefaults(v *viper.Viper, flags *pflag.FlagSet, cfg *Config) {
	if !isExplicit(v, flags, "tts.temperature") {
		cfg.TTS.Temperature = cfg.Model.DefaultTemperature
	}
}

// DefaultLanguage is the model config used without --language. It stays the
// model of earlier releases until German support switches it (PLAN.md
// Phase 6).
const DefaultLanguage = "english_2026-01"

// LanguagePaths are the local files of one language.
type LanguagePaths struct {
	ModelPath      string
	TokenizerModel string
	VoiceManifest  string
}

// PathsForLanguage returns the local layout of language:
// models/<lang>/model.safetensors, models/<lang>/tokenizer.model and
// voices/<lang>/manifest.json. DefaultLanguage keeps the flat layout of
// earlier releases.
func PathsForLanguage(language string) LanguagePaths {
	if language == DefaultLanguage {
		return LanguagePaths{
			ModelPath:      "models/tts_b6369a24.safetensors",
			TokenizerModel: "models/tokenizer.model",
			VoiceManifest:  "voices/manifest.json",
		}
	}

	return LanguagePaths{
		ModelPath:      path.Join("models", language, "model.safetensors"),
		TokenizerModel: path.Join("models", language, "tokenizer.model"),
		VoiceManifest:  path.Join("voices", language, "manifest.json"),
	}
}

// resolveModel loads cfg.Model and fills the language-derived paths the user
// did not set explicitly. It must run after the config file was read.
func resolveModel(v *viper.Viper, flags *pflag.FlagSet, cfg *Config) error {
	explicitModelPath := isExplicit(v, flags, "paths.model_path")
	explicitTokenizer := isExplicit(v, flags, "paths.tokenizer_model")

	if cfg.TTS.ModelConfigPath != "" {
		return resolveCustomModel(v, flags, cfg, explicitModelPath, explicitTokenizer)
	}

	model, err := modelcfg.Lookup(cfg.TTS.Language)
	if err != nil {
		return fmt.Errorf("--language: %w", err)
	}

	cfg.Model = model

	derived := PathsForLanguage(cfg.TTS.Language)

	if !explicitModelPath {
		cfg.Paths.ModelPath = derived.ModelPath
	}

	if !explicitTokenizer {
		cfg.Paths.TokenizerModel = derived.TokenizerModel
	}

	if !isExplicit(v, flags, "paths.voice_manifest") {
		cfg.Paths.VoiceManifest = derived.VoiceManifest
	}

	return nil
}

// resolveCustomModel handles --model-config. Like upstream, it excludes
// --language; the paths must be explicit because there is no language to
// derive them from.
func resolveCustomModel(v *viper.Viper, flags *pflag.FlagSet, cfg *Config, explicitModelPath, explicitTokenizer bool) error {
	if isExplicit(v, flags, "tts.language") {
		return errors.New("--model-config and --language are incompatible; use one of them")
	}

	var missing []string

	if !explicitModelPath {
		missing = append(missing, "--paths-model-path")
	}

	if !explicitTokenizer {
		missing = append(missing, "--paths-tokenizer-model")
	}

	if len(missing) > 0 {
		return fmt.Errorf("--model-config needs explicit %s", strings.Join(missing, " and "))
	}

	model, err := modelcfg.LoadCustom(cfg.TTS.ModelConfigPath)
	if err != nil {
		return fmt.Errorf("--model-config: %w", err)
	}

	cfg.Model = model

	return nil
}

// isExplicit reports whether key was set by a flag, an env var or the config
// file rather than coming from a default.
func isExplicit(v *viper.Viper, flags *pflag.FlagSet, key string) bool {
	if v.InConfig(key) {
		return true
	}

	for _, b := range keyBindings {
		if b.key != key {
			continue
		}

		if flags != nil && flags.Changed(b.flag) {
			return true
		}

		for _, name := range b.envNames() {
			if os.Getenv(name) != "" {
				return true
			}
		}
	}

	return false
}

func setDefaults(v *viper.Viper, c Config) {
	v.SetDefault("paths.model_path", c.Paths.ModelPath)
	v.SetDefault("paths.voice_path", c.Paths.VoicePath)
	v.SetDefault("paths.onnx_manifest", c.Paths.ONNXManifest)
	v.SetDefault("paths.tokenizer_model", c.Paths.TokenizerModel)
	v.SetDefault("paths.voice_manifest", c.Paths.VoiceManifest)
	v.SetDefault("runtime.threads", c.Runtime.Threads)
	v.SetDefault("runtime.inter_op_threads", c.Runtime.InterOpThreads)
	v.SetDefault("runtime.workers", c.Runtime.Workers)
	v.SetDefault("runtime.conv_workers", c.Runtime.ConvWorkers)
	v.SetDefault("runtime.ort_library_path", c.Runtime.ORTLibraryPath)
	v.SetDefault("runtime.ort_version", c.Runtime.ORTVersion)
	v.SetDefault("server.listen_addr", c.Server.ListenAddr)
	v.SetDefault("server.grpc_addr", c.Server.GRPCAddr)
	v.SetDefault("server.workers", c.Server.Workers)
	v.SetDefault("server.shutdown_timeout_secs", c.Server.ShutdownTimeout)
	v.SetDefault("server.max_text_bytes", c.Server.MaxTextBytes)
	v.SetDefault("server.request_timeout_secs", c.Server.RequestTimeout)
	v.SetDefault("tts.backend", c.TTS.Backend)
	v.SetDefault("tts.language", c.TTS.Language)
	v.SetDefault("tts.model_config", c.TTS.ModelConfigPath)
	v.SetDefault("tts.voice", c.TTS.Voice)
	v.SetDefault("tts.cli_path", c.TTS.CLIPath)
	v.SetDefault("tts.cli_config_path", c.TTS.CLIConfigPath)
	v.SetDefault("tts.concurrency", c.TTS.Concurrency)
	v.SetDefault("tts.quiet", c.TTS.Quiet)
	v.SetDefault("tts.temperature", c.TTS.Temperature)
	v.SetDefault("tts.eos_threshold", c.TTS.EOSThreshold)
	v.SetDefault("tts.max_steps", c.TTS.MaxSteps)
	v.SetDefault(samplerDecodeStepsKey, c.TTS.SamplerDecodeSteps)
	v.SetDefault("log_level", c.LogLevel)
}

const (
	samplerDecodeStepsKey           = "tts.sampler_decode_steps"
	deprecatedSamplerDecodeStepsKey = "tts.lsd_decode_steps"
	ortLibraryPathKey               = "runtime.ort_library_path"
)

// keyBinding ties a config key to its flag and environment variables. Every
// key is read from POCKETTTS_<SECTION>_<KEY> first, then from the flag-style
// POCKETTTS_<FLAG>, then from env in order.
type keyBinding struct {
	key  string
	flag string
	env  []string
}

// keyBindings lists every config key. Each key is bound to its flag with
// BindPFlag; a viper alias from the key to the flag name instead would make
// config-file values and section-style env vars unreachable.
var keyBindings = []keyBinding{
	{key: "paths.model_path", flag: "paths-model-path"},
	{key: "paths.voice_path", flag: "paths-voice-path"},
	{key: "paths.onnx_manifest", flag: "paths-onnx-manifest"},
	{key: "paths.tokenizer_model", flag: "paths-tokenizer-model"},
	{key: "paths.voice_manifest", flag: "paths-voice-manifest"},
	{key: "runtime.threads", flag: "runtime-threads"},
	{key: "runtime.inter_op_threads", flag: "runtime-inter-op-threads"},
	{key: "runtime.workers", flag: "runtime-workers"},
	{key: "runtime.conv_workers", flag: "conv-workers"},
	{key: ortLibraryPathKey, flag: "runtime-ort-library-path", env: []string{"POCKETTTS_ORT_LIB", "ORT_LIBRARY_PATH"}},
	{key: "runtime.ort_version", flag: "runtime-ort-version"},
	{key: "server.listen_addr", flag: "server-listen-addr"},
	{key: "server.grpc_addr", flag: "server-grpc-addr"},
	{key: "server.workers", flag: "workers"},
	{key: "server.shutdown_timeout_secs", flag: "shutdown-timeout"},
	{key: "server.max_text_bytes", flag: "max-text-bytes"},
	{key: "server.request_timeout_secs", flag: "request-timeout"},
	{key: "tts.backend", flag: "backend"},
	{key: "tts.language", flag: "language"},
	{key: "tts.model_config", flag: "model-config"},
	{key: "tts.voice", flag: "tts-voice"},
	{key: "tts.cli_path", flag: "tts-cli-path"},
	{key: "tts.cli_config_path", flag: "tts-cli-config-path"},
	{key: "tts.concurrency", flag: "tts-concurrency"},
	{key: "tts.quiet", flag: "tts-quiet"},
	{key: "tts.temperature", flag: "temperature"},
	{key: "tts.eos_threshold", flag: "eos-threshold"},
	{key: "tts.max_steps", flag: "max-steps"},
	{
		key:  samplerDecodeStepsKey,
		flag: "sampler-decode-steps",
		env:  []string{"POCKETTTS_TTS_LSD_DECODE_STEPS", "POCKETTTS_LSD_STEPS"},
	},
	{key: "log_level", flag: "log-level"},
}

// flagAliases lists string flags that set the same key as a canonical flag.
// An alias only applies when the canonical flag was not given; see
// applyFlagAliases.
var flagAliases = []struct{ alias, canonical, key string }{
	{alias: "ort-lib", canonical: "runtime-ort-library-path", key: ortLibraryPathKey},
}

// envNames returns the env vars of b, highest precedence first.
func (b keyBinding) envNames() []string {
	names := []string{envName(b.key)}
	if flagEnv := envName(b.flag); flagEnv != names[0] {
		names = append(names, flagEnv)
	}

	return append(names, b.env...)
}

func envName(name string) string {
	return "POCKETTTS_" + strings.ToUpper(strings.NewReplacer(".", "_", "-", "_").Replace(name))
}

// bindKeys binds every key of keyBindings to its flag (when flags has it) and
// its env vars.
func bindKeys(v *viper.Viper, flags *pflag.FlagSet) error {
	for _, b := range keyBindings {
		if flags != nil {
			if f := flags.Lookup(b.flag); f != nil {
				err := v.BindPFlag(b.key, f)
				if err != nil {
					return fmt.Errorf("bind %s flag: %w", b.flag, err)
				}
			}
		}

		err := v.BindEnv(append([]string{b.key}, b.envNames()...)...)
		if err != nil {
			return fmt.Errorf("bind %s env vars: %w", b.key, err)
		}
	}

	return nil
}

// applyFlagAliases applies an explicitly given alias flag (e.g. --ort-lib)
// unless its canonical flag was given too. It must run after the config file
// was read.
func applyFlagAliases(v *viper.Viper, flags *pflag.FlagSet) error {
	if flags == nil {
		return nil
	}

	for _, fa := range flagAliases {
		if !flags.Changed(fa.alias) || flags.Changed(fa.canonical) {
			continue
		}

		value, err := flags.GetString(fa.alias)
		if err != nil {
			return fmt.Errorf("read %s flag: %w", fa.alias, err)
		}

		v.Set(fa.key, value)
	}

	return nil
}

// applyDeprecatedSamplerDecodeSteps maps the deprecated lsd names onto
// tts.sampler_decode_steps. It must run after the config file was read.
//
//   - An explicit --lsd-steps overrides everything, like upstream's
//     --lsd-decode-steps overrides --sampler-decode-steps.
//   - A tts.lsd_decode_steps config-file key only replaces the default, so
//     the new key, env vars and flags still take precedence over it.
func applyDeprecatedSamplerDecodeSteps(v *viper.Viper, flags *pflag.FlagSet) error {
	if flags != nil && flags.Changed("lsd-steps") {
		steps, err := flags.GetInt("lsd-steps")
		if err != nil {
			return fmt.Errorf("read deprecated lsd-steps flag: %w", err)
		}

		v.Set(samplerDecodeStepsKey, steps)

		return nil
	}

	if v.InConfig(deprecatedSamplerDecodeStepsKey) {
		v.SetDefault(samplerDecodeStepsKey, v.Get(deprecatedSamplerDecodeStepsKey))
	}

	return nil
}
