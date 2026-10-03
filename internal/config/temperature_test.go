package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
)

// Upstream: --temperature defaults to None, i.e. the model's default_temperature.
func TestLoad_Temperature_FromModel(t *testing.T) {
	for name, in := range map[string]loadInput{
		"default":  {},
		"german":   {args: []string{"--language=german"}},
		"no flags": {noCmd: true},
	} {
		t.Run(name, func(t *testing.T) {
			cfg := loadWith(t, in)

			if cfg.TTS.Temperature != cfg.Model.DefaultTemperature || cfg.TTS.Temperature != 0.3 {
				t.Errorf("TTS.Temperature = %v; want the model's default_temperature 0.3 (model has %v)",
					cfg.TTS.Temperature, cfg.Model.DefaultTemperature)
			}
		})
	}
}

func TestLoad_Temperature_ExplicitWins(t *testing.T) {
	for name, in := range map[string]loadInput{
		"flag":           {args: []string{"--temperature=0.7"}},
		"section env":    {env: map[string]string{"POCKETTTS_TTS_TEMPERATURE": "0.7"}},
		"flag-style env": {env: map[string]string{"POCKETTTS_TEMPERATURE": "0.7"}},
		"file":           {file: "tts:\n  temperature: 0.7\n"},
		"file no flags":  {file: "tts:\n  temperature: 0.7\n", noCmd: true},
		// An explicit zero must not be mistaken for unset.
		"flag zero": {args: []string{"--temperature=0"}},
	} {
		t.Run(name, func(t *testing.T) {
			cfg := loadWith(t, in)

			want := 0.7
			if name == "flag zero" {
				want = 0
			}

			if cfg.TTS.Temperature != want {
				t.Errorf("TTS.Temperature = %v; want %v", cfg.TTS.Temperature, want)
			}
		})
	}
}

func TestLoad_Temperature_CustomModelConfig(t *testing.T) {
	data, err := os.ReadFile(germanModelConfig)
	if err != nil {
		t.Fatal(err)
	}

	custom := filepath.Join(t.TempDir(), "custom.yaml")

	err = os.WriteFile(custom, append(data, []byte("\ndefault_temperature: 0.5\n")...), 0o644)
	if err != nil {
		t.Fatal(err)
	}

	cfg := loadWith(t, loadInput{args: []string{
		"--model-config=" + custom, "--paths-model-path=/m.safetensors", "--paths-tokenizer-model=/t.model",
	}})

	if cfg.TTS.Temperature != 0.5 {
		t.Errorf("TTS.Temperature = %v; want the custom config's 0.5", cfg.TTS.Temperature)
	}
}

// DefaultConfig serves callers that never run Load (the wasm build), so it
// must agree with the default language's model config.
func TestDefaultConfig_TemperatureMatchesDefaultLanguage(t *testing.T) {
	model, err := modelcfg.Lookup(DefaultLanguage)
	if err != nil {
		t.Fatal(err)
	}

	if got := DefaultConfig().TTS.Temperature; got != model.DefaultTemperature {
		t.Errorf("DefaultConfig().TTS.Temperature = %v; want %s default_temperature %v",
			got, DefaultLanguage, model.DefaultTemperature)
	}
}
