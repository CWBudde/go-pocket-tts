package config

import (
	"strings"
	"testing"
)

// germanModelConfig is a model config file outside the embedded registry.
const germanModelConfig = "../modelcfg/testdata/german.yaml"

func TestLoad_Language_Default(t *testing.T) {
	cfg := loadWith(t, loadInput{})

	if cfg.TTS.Language != "english_2026-01" {
		t.Errorf("TTS.Language = %q; want english_2026-01", cfg.TTS.Language)
	}

	if cfg.Model == nil || cfg.Model.FlowLM.Transformer.NumLayers != 6 || cfg.Model.DefaultVoice != "alba" {
		t.Fatalf("Model = %+v; want the english_2026-01 config with default voice alba", cfg.Model)
	}

	// The default language keeps the flat layout of earlier releases.
	want := PathsConfig{
		ModelPath:      "models/tts_b6369a24.safetensors",
		TokenizerModel: "models/tokenizer.model",
		VoiceManifest:  "voices/manifest.json",
	}
	checkLanguagePaths(t, cfg, want)
}

func TestLoad_Language_DerivesPaths(t *testing.T) {
	inputs := map[string]loadInput{
		"flag":                   {args: []string{"--language=german"}},
		"section env":            {env: map[string]string{"POCKETTTS_TTS_LANGUAGE": "german"}},
		"flag-style env":         {env: map[string]string{"POCKETTTS_LANGUAGE": "german"}},
		"file":                   {file: "tts:\n  language: german\n"},
		"file without flags":     {file: "tts:\n  language: german\n", noCmd: true},
		"env without flags":      {env: map[string]string{"POCKETTTS_TTS_LANGUAGE": "german"}, noCmd: true},
		"flag-style without cmd": {env: map[string]string{"POCKETTTS_LANGUAGE": "german"}, noCmd: true},
	}

	want := PathsConfig{
		ModelPath:      "models/german/model.safetensors",
		TokenizerModel: "models/german/tokenizer.json",
		VoiceManifest:  "voices/german/manifest.json",
	}

	for name, in := range inputs {
		t.Run(name, func(t *testing.T) {
			cfg := loadWith(t, in)

			if cfg.TTS.Language != "german" {
				t.Errorf("TTS.Language = %q; want german", cfg.TTS.Language)
			}

			if cfg.Model == nil || !cfg.Model.FlowLM.InsertBOSBeforeVoice || cfg.Model.DefaultVoice != "juergen" {
				t.Fatalf("Model = %+v; want the german config with default voice juergen", cfg.Model)
			}

			checkLanguagePaths(t, cfg, want)
		})
	}
}

func TestLoad_Language_ExplicitPathsWin(t *testing.T) {
	inputs := map[string]loadInput{
		"flag": {args: []string{"--language=german", "--paths-model-path=/m.safetensors"}},
		"env": {
			args: []string{"--language=german"},
			env:  map[string]string{"POCKETTTS_PATHS_MODEL_PATH": "/m.safetensors"},
		},
		"file": {file: "tts:\n  language: german\npaths:\n  model_path: /m.safetensors\n"},
	}

	for name, in := range inputs {
		t.Run(name, func(t *testing.T) {
			cfg := loadWith(t, in)

			checkLanguagePaths(t, cfg, PathsConfig{
				ModelPath:      "/m.safetensors",
				TokenizerModel: "models/german/tokenizer.json",
				VoiceManifest:  "voices/german/manifest.json",
			})
		})
	}

	// An explicit value equal to the flat default still counts as explicit.
	cfg := loadWith(t, loadInput{args: []string{
		"--language=german",
		"--paths-model-path=models/tts_b6369a24.safetensors",
	}})
	if cfg.Paths.ModelPath != "models/tts_b6369a24.safetensors" {
		t.Errorf("explicit flat model path was replaced by %q", cfg.Paths.ModelPath)
	}
}

func TestLoad_Language_Unknown(t *testing.T) {
	_, err := tryLoad(t, loadInput{args: []string{"--language=klingon"}})
	if err == nil || !strings.Contains(err.Error(), "klingon") || !strings.Contains(err.Error(), "german") {
		t.Fatalf("Load error = %v; want an error naming klingon and the valid languages", err)
	}
}

func TestLoad_ModelConfig(t *testing.T) {
	paths := []string{"--paths-model-path=/m.safetensors", "--paths-tokenizer-model=/t.model"}

	inputs := map[string]loadInput{
		"flag": {args: append([]string{"--model-config=" + germanModelConfig}, paths...)},
		"env": {
			args: paths,
			env:  map[string]string{"POCKETTTS_TTS_MODEL_CONFIG": germanModelConfig},
		},
		"file": {file: "tts:\n  model_config: " + germanModelConfig +
			"\npaths:\n  model_path: /m.safetensors\n  tokenizer_model: /t.model\n"},
	}

	for name, in := range inputs {
		t.Run(name, func(t *testing.T) {
			cfg := loadWith(t, in)

			// Upstream: a custom config defaults to alba's voice.
			if cfg.Model == nil || !cfg.Model.FlowLM.InsertBOSBeforeVoice || cfg.Model.DefaultVoice != "alba" {
				t.Fatalf("Model = %+v; want the german test config with default voice alba", cfg.Model)
			}

			if cfg.Paths.ModelPath != "/m.safetensors" || cfg.Paths.TokenizerModel != "/t.model" {
				t.Errorf("Paths = %+v; want the explicit paths", cfg.Paths)
			}
		})
	}
}

func TestLoad_ModelConfig_Errors(t *testing.T) {
	paths := []string{"--paths-model-path=/m.safetensors", "--paths-tokenizer-model=/t.model"}

	tests := []struct {
		name    string
		in      loadInput
		wantErr string
	}{
		{
			name:    "with explicit language",
			in:      loadInput{args: append([]string{"--model-config=" + germanModelConfig, "--language=german"}, paths...)},
			wantErr: "--language",
		},
		{
			name:    "without model and tokenizer paths",
			in:      loadInput{args: []string{"--model-config=" + germanModelConfig}},
			wantErr: "--paths-model-path and --paths-tokenizer-model",
		},
		{
			name: "without tokenizer path",
			in: loadInput{args: []string{
				"--model-config=" + germanModelConfig,
				"--paths-model-path=/m.safetensors",
			}},
			wantErr: "--paths-tokenizer-model",
		},
		{
			name:    "missing file",
			in:      loadInput{args: append([]string{"--model-config=/nonexistent/model.yaml"}, paths...)},
			wantErr: "/nonexistent/model.yaml",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tryLoad(t, tc.in)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Load error = %v; want an error containing %q", err, tc.wantErr)
			}
		})
	}
}

func checkLanguagePaths(t *testing.T, cfg Config, want PathsConfig) {
	t.Helper()

	if cfg.Paths.ModelPath != want.ModelPath {
		t.Errorf("Paths.ModelPath = %q; want %q", cfg.Paths.ModelPath, want.ModelPath)
	}

	if cfg.Paths.TokenizerModel != want.TokenizerModel {
		t.Errorf("Paths.TokenizerModel = %q; want %q", cfg.Paths.TokenizerModel, want.TokenizerModel)
	}

	if cfg.Paths.VoiceManifest != want.VoiceManifest {
		t.Errorf("Paths.VoiceManifest = %q; want %q", cfg.Paths.VoiceManifest, want.VoiceManifest)
	}
}

func TestForLanguage(t *testing.T) {
	base := loadWith(t, loadInput{args: []string{
		"--paths-model-path=/x/model.safetensors",
		"--paths-voice-manifest=/x/voices.json",
		"--default-voice=marius",
	}})

	cfg, err := ForLanguage(base, "german")
	if err != nil {
		t.Fatalf("ForLanguage: %v", err)
	}

	if cfg.TTS.Language != "german" || cfg.Model == nil || cfg.Model.DefaultVoice != "juergen" {
		t.Fatalf("ForLanguage(german) = language %q, model %+v; want german with default voice juergen",
			cfg.TTS.Language, cfg.Model)
	}

	// The startup language's explicit paths and default voice do not carry over.
	checkLanguagePaths(t, cfg, PathsConfig{
		ModelPath:      "models/german/model.safetensors",
		TokenizerModel: "models/german/tokenizer.json",
		VoiceManifest:  "voices/german/manifest.json",
	})

	if cfg.Server.DefaultVoice != "" {
		t.Errorf("Server.DefaultVoice = %q; want empty (other languages use their built-in voice)", cfg.Server.DefaultVoice)
	}

	if base.Paths.ModelPath != "/x/model.safetensors" || base.Model.DefaultVoice != "alba" {
		t.Errorf("ForLanguage changed its base: %+v", base.Paths)
	}
}

func TestForLanguage_Temperature(t *testing.T) {
	// Without --temperature, the temperature follows the other model.
	base := loadWith(t, loadInput{})
	base.TTS.Temperature = 0.9

	cfg, err := ForLanguage(base, "german")
	if err != nil {
		t.Fatal(err)
	}

	if cfg.TTS.Temperature != cfg.Model.DefaultTemperature {
		t.Errorf("Temperature = %v; want german's default_temperature %v", cfg.TTS.Temperature, cfg.Model.DefaultTemperature)
	}

	// An explicit --temperature applies to every language.
	cfg, err = ForLanguage(loadWith(t, loadInput{args: []string{"--temperature=0.7"}}), "german")
	if err != nil {
		t.Fatal(err)
	}

	if cfg.TTS.Temperature != 0.7 {
		t.Errorf("Temperature = %v; want the explicit 0.7", cfg.TTS.Temperature)
	}
}

func TestForLanguage_Errors(t *testing.T) {
	_, err := ForLanguage(loadWith(t, loadInput{}), "klingon")
	if err == nil || !strings.Contains(err.Error(), "unknown language") {
		t.Errorf("ForLanguage(klingon) error = %v; want an unknown language error", err)
	}

	custom := loadWith(t, loadInput{args: []string{
		"--model-config=" + germanModelConfig,
		"--paths-model-path=/x/model.safetensors",
		"--paths-tokenizer-model=/x/tok.model",
	}})

	_, err = ForLanguage(custom, "german")
	if err == nil || !strings.Contains(err.Error(), "model-config") {
		t.Errorf("ForLanguage(custom model) error = %v; want a --model-config error", err)
	}
}
