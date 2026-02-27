package config

import (
	"fmt"
	"strings"

	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

type Config struct {
	Paths    PathsConfig   `mapstructure:"paths"`
	Runtime  RuntimeConfig `mapstructure:"runtime"`
	Server   ServerConfig  `mapstructure:"server"`
	TTS      TTSConfig     `mapstructure:"tts"`
	LogLevel string        `mapstructure:"log_level"`
}

type PathsConfig struct {
	ModelPath      string `mapstructure:"model_path"`
	VoicePath      string `mapstructure:"voice_path"`
	ONNXManifest   string `mapstructure:"onnx_manifest"`
	TokenizerModel string `mapstructure:"tokenizer_model"`
}

type RuntimeConfig struct {
	Threads        int    `mapstructure:"threads"`
	InterOpThreads int    `mapstructure:"inter_op_threads"`
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
	Backend        string  `mapstructure:"backend"`
	Voice          string  `mapstructure:"voice"`
	CLIPath        string  `mapstructure:"cli_path"`
	CLIConfigPath  string  `mapstructure:"cli_config_path"`
	Concurrency    int     `mapstructure:"concurrency"`
	Quiet          bool    `mapstructure:"quiet"`
	Temperature    float64 `mapstructure:"temperature"`
	EOSThreshold   float64 `mapstructure:"eos_threshold"`
	MaxSteps       int     `mapstructure:"max_steps"`
	LSDDecodeSteps int     `mapstructure:"lsd_decode_steps"`
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
		},
		Runtime: RuntimeConfig{
			Threads:        4,
			InterOpThreads: 1,
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
			Backend:        BackendNative,
			Voice:          "",
			CLIPath:        "",
			CLIConfigPath:  "",
			Concurrency:    1,
			Quiet:          true,
			Temperature:    0.7,
			EOSThreshold:   -4.0,
			MaxSteps:       256,
			LSDDecodeSteps: 1,
		},
		LogLevel: "info",
	}
}

func RegisterFlags(fs *pflag.FlagSet, defaults Config) {
	fs.String("paths-model-path", defaults.Paths.ModelPath, "Path to model file (.safetensors for native, .onnx for native-onnx)")
	fs.String("paths-voice-path", defaults.Paths.VoicePath, "Path to voice/profile asset")
	fs.String("paths-onnx-manifest", defaults.Paths.ONNXManifest, "Path to ONNX model manifest JSON")
	fs.String("paths-tokenizer-model", defaults.Paths.TokenizerModel, "Path to SentencePiece tokenizer model")
	fs.Int("runtime-threads", defaults.Runtime.Threads, "Inference thread count (ONNX intra-op for native-onnx backend)")
	fs.Int("runtime-inter-op-threads", defaults.Runtime.InterOpThreads, "Inter-op thread count (ONNX-only, native-onnx backend)")
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
	fs.String("tts-voice", defaults.TTS.Voice, "Voice name or .safetensors file path")
	fs.String("tts-cli-path", defaults.TTS.CLIPath, "Path to pocket-tts executable")
	fs.String("tts-cli-config-path", defaults.TTS.CLIConfigPath, "Path to pocket-tts config file")
	fs.Int("tts-concurrency", defaults.TTS.Concurrency, "Max concurrent pocket-tts subprocesses")
	fs.Bool("tts-quiet", defaults.TTS.Quiet, "Pass --quiet to pocket-tts generate")
	fs.Float64("temperature", defaults.TTS.Temperature, "Noise temperature for flow sampling")
	fs.Float64("eos-threshold", defaults.TTS.EOSThreshold, "Raw logit threshold for EOS detection")
	fs.Int("max-steps", defaults.TTS.MaxSteps, "Maximum autoregressive generation steps")
	fs.Int("lsd-steps", defaults.TTS.LSDDecodeSteps, "Euler integration steps per latent frame")
	fs.String("log-level", defaults.LogLevel, "Log level (debug|info|warn|error)")
}

func Load(opts LoadOptions) (Config, error) {
	v := viper.New()

	setDefaults(v, opts.Defaults)
	if opts.Cmd != nil {
		if err := v.BindPFlags(opts.Cmd.Flags()); err != nil {
			return Config{}, fmt.Errorf("bind flags: %w", err)
		}
	}
	registerAliases(v)

	v.SetEnvPrefix("POCKETTTS")
	replacer := strings.NewReplacer("-", "_", ".", "_", "__", "_")
	v.SetEnvKeyReplacer(replacer)
	if err := v.BindEnv("runtime.ort_library_path", "POCKETTTS_ORT_LIB", "ORT_LIBRARY_PATH"); err != nil {
		return Config{}, fmt.Errorf("bind ort env vars: %w", err)
	}
	v.AutomaticEnv()

	if opts.ConfigFile != "" {
		v.SetConfigFile(opts.ConfigFile)
		if err := v.ReadInConfig(); err != nil {
			return Config{}, fmt.Errorf("read config file: %w", err)
		}
	} else {
		v.SetConfigName("pockettts")
		v.AddConfigPath(".")
		if err := v.ReadInConfig(); err != nil {
			if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
				return Config{}, fmt.Errorf("read config file: %w", err)
			}
		}
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return Config{}, fmt.Errorf("decode config: %w", err)
	}

	return cfg, nil
}

func setDefaults(v *viper.Viper, c Config) {
	v.SetDefault("paths.model_path", c.Paths.ModelPath)
	v.SetDefault("paths.voice_path", c.Paths.VoicePath)
	v.SetDefault("paths.onnx_manifest", c.Paths.ONNXManifest)
	v.SetDefault("paths.tokenizer_model", c.Paths.TokenizerModel)
	v.SetDefault("runtime.threads", c.Runtime.Threads)
	v.SetDefault("runtime.inter_op_threads", c.Runtime.InterOpThreads)
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
	v.SetDefault("tts.voice", c.TTS.Voice)
	v.SetDefault("tts.cli_path", c.TTS.CLIPath)
	v.SetDefault("tts.cli_config_path", c.TTS.CLIConfigPath)
	v.SetDefault("tts.concurrency", c.TTS.Concurrency)
	v.SetDefault("tts.quiet", c.TTS.Quiet)
	v.SetDefault("tts.temperature", c.TTS.Temperature)
	v.SetDefault("tts.eos_threshold", c.TTS.EOSThreshold)
	v.SetDefault("tts.max_steps", c.TTS.MaxSteps)
	v.SetDefault("tts.lsd_decode_steps", c.TTS.LSDDecodeSteps)
	v.SetDefault("log_level", c.LogLevel)
}

func registerAliases(v *viper.Viper) {
	v.RegisterAlias("paths.model_path", "paths-model-path")
	v.RegisterAlias("paths.voice_path", "paths-voice-path")
	v.RegisterAlias("paths.onnx_manifest", "paths-onnx-manifest")
	v.RegisterAlias("paths.tokenizer_model", "paths-tokenizer-model")
	v.RegisterAlias("runtime.threads", "runtime-threads")
	v.RegisterAlias("runtime.inter_op_threads", "runtime-inter-op-threads")
	v.RegisterAlias("runtime.conv_workers", "conv-workers")
	v.RegisterAlias("runtime.ort_library_path", "runtime-ort-library-path")
	v.RegisterAlias("runtime.ort_library_path", "ort-lib")
	v.RegisterAlias("runtime.ort_version", "runtime-ort-version")
	v.RegisterAlias("server.listen_addr", "server-listen-addr")
	v.RegisterAlias("server.grpc_addr", "server-grpc-addr")
	v.RegisterAlias("server.workers", "workers")
	v.RegisterAlias("server.shutdown_timeout_secs", "shutdown-timeout")
	v.RegisterAlias("server.max_text_bytes", "max-text-bytes")
	v.RegisterAlias("server.request_timeout_secs", "request-timeout")
	v.RegisterAlias("tts.backend", "backend")
	v.RegisterAlias("tts.voice", "tts-voice")
	v.RegisterAlias("tts.cli_path", "tts-cli-path")
	v.RegisterAlias("tts.cli_config_path", "tts-cli-config-path")
	v.RegisterAlias("tts.concurrency", "tts-concurrency")
	v.RegisterAlias("tts.quiet", "tts-quiet")
	v.RegisterAlias("tts.temperature", "temperature")
	v.RegisterAlias("tts.eos_threshold", "eos-threshold")
	v.RegisterAlias("tts.max_steps", "max-steps")
	v.RegisterAlias("tts.lsd_decode_steps", "lsd-steps")
	v.RegisterAlias("log_level", "log-level")
}
