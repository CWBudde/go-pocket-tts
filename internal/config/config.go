package config

import (
	"fmt"
	"strings"

	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

type Config struct {
	Paths   PathsConfig   `mapstructure:"paths"`
	Runtime RuntimeConfig `mapstructure:"runtime"`
	Server  ServerConfig  `mapstructure:"server"`
	TTS     TTSConfig     `mapstructure:"tts"`
}

type PathsConfig struct {
	ModelPath string `mapstructure:"model_path"`
	VoicePath string `mapstructure:"voice_path"`
}

type RuntimeConfig struct {
	Threads        int    `mapstructure:"threads"`
	InterOpThreads int    `mapstructure:"inter_op_threads"`
	ORTLibraryPath string `mapstructure:"ort_library_path"`
	ORTVersion     string `mapstructure:"ort_version"`
}

type ServerConfig struct {
	ListenAddr string `mapstructure:"listen_addr"`
	GRPCAddr   string `mapstructure:"grpc_addr"`
}

type TTSConfig struct {
	Voice         string `mapstructure:"voice"`
	CLIPath       string `mapstructure:"cli_path"`
	CLIConfigPath string `mapstructure:"cli_config_path"`
	Concurrency   int    `mapstructure:"concurrency"`
	Quiet         bool   `mapstructure:"quiet"`
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
			ModelPath: "models/model.onnx",
			VoicePath: "models/voice.bin",
		},
		Runtime: RuntimeConfig{
			Threads:        4,
			InterOpThreads: 1,
			ORTLibraryPath: "",
			ORTVersion:     "",
		},
		Server: ServerConfig{
			ListenAddr: ":8080",
			GRPCAddr:   ":9090",
		},
		TTS: TTSConfig{
			Voice:         "",
			CLIPath:       "",
			CLIConfigPath: "",
			Concurrency:   1,
			Quiet:         true,
		},
	}
}

func RegisterFlags(fs *pflag.FlagSet, defaults Config) {
	fs.String("paths-model-path", defaults.Paths.ModelPath, "Path to ONNX model")
	fs.String("paths-voice-path", defaults.Paths.VoicePath, "Path to voice/profile asset")
	fs.Int("runtime-threads", defaults.Runtime.Threads, "ONNX Runtime intra-op thread count")
	fs.Int("runtime-inter-op-threads", defaults.Runtime.InterOpThreads, "ONNX Runtime inter-op thread count")
	fs.String("runtime-ort-library-path", defaults.Runtime.ORTLibraryPath, "Path to ONNX Runtime shared library")
	fs.String("ort-lib", defaults.Runtime.ORTLibraryPath, "Path to ONNX Runtime shared library (alias for --runtime-ort-library-path)")
	fs.String("runtime-ort-version", defaults.Runtime.ORTVersion, "Expected ONNX Runtime version")
	fs.String("server-listen-addr", defaults.Server.ListenAddr, "HTTP listen address")
	fs.String("server-grpc-addr", defaults.Server.GRPCAddr, "gRPC listen address")
	fs.String("tts-voice", defaults.TTS.Voice, "Voice name or .safetensors file path")
	fs.String("tts-cli-path", defaults.TTS.CLIPath, "Path to pocket-tts executable")
	fs.String("tts-cli-config-path", defaults.TTS.CLIConfigPath, "Path to pocket-tts config file")
	fs.Int("tts-concurrency", defaults.TTS.Concurrency, "Max concurrent pocket-tts subprocesses")
	fs.Bool("tts-quiet", defaults.TTS.Quiet, "Pass --quiet to pocket-tts generate")
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
	v.SetDefault("runtime.threads", c.Runtime.Threads)
	v.SetDefault("runtime.inter_op_threads", c.Runtime.InterOpThreads)
	v.SetDefault("runtime.ort_library_path", c.Runtime.ORTLibraryPath)
	v.SetDefault("runtime.ort_version", c.Runtime.ORTVersion)
	v.SetDefault("server.listen_addr", c.Server.ListenAddr)
	v.SetDefault("server.grpc_addr", c.Server.GRPCAddr)
	v.SetDefault("tts.voice", c.TTS.Voice)
	v.SetDefault("tts.cli_path", c.TTS.CLIPath)
	v.SetDefault("tts.cli_config_path", c.TTS.CLIConfigPath)
	v.SetDefault("tts.concurrency", c.TTS.Concurrency)
	v.SetDefault("tts.quiet", c.TTS.Quiet)
}

func registerAliases(v *viper.Viper) {
	v.RegisterAlias("paths.model_path", "paths-model-path")
	v.RegisterAlias("paths.voice_path", "paths-voice-path")
	v.RegisterAlias("runtime.threads", "runtime-threads")
	v.RegisterAlias("runtime.inter_op_threads", "runtime-inter-op-threads")
	v.RegisterAlias("runtime.ort_library_path", "runtime-ort-library-path")
	v.RegisterAlias("runtime.ort_library_path", "ort-lib")
	v.RegisterAlias("runtime.ort_version", "runtime-ort-version")
	v.RegisterAlias("server.listen_addr", "server-listen-addr")
	v.RegisterAlias("server.grpc_addr", "server-grpc-addr")
	v.RegisterAlias("tts.voice", "tts-voice")
	v.RegisterAlias("tts.cli_path", "tts-cli-path")
	v.RegisterAlias("tts.cli_config_path", "tts-cli-config-path")
	v.RegisterAlias("tts.concurrency", "tts-concurrency")
	v.RegisterAlias("tts.quiet", "tts-quiet")
}
