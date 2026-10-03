package main

import (
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
)

func TestResolveVoiceDownload(t *testing.T) {
	tests := []struct {
		name      string
		language  string
		outDir    string
		outDirSet bool
		wantDir   string
	}{
		{"default language keeps flat voices/", config.DefaultLanguage, "voices", false, "voices"},
		{"other language gets voices/<lang>", "german", "voices", false, filepath.Join("voices", "german")},
		{"explicit out-dir wins", "german", "/data/de", true, "/data/de"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := config.DefaultConfig()
			cfg.TTS.Language = tc.language

			language, dir, err := resolveVoiceDownload(cfg, tc.outDir, tc.outDirSet)
			if err != nil {
				t.Fatal(err)
			}

			if language != tc.language || dir != tc.wantDir {
				t.Errorf("resolveVoiceDownload = %q, %q; want %q, %q", language, dir, tc.language, tc.wantDir)
			}
		})
	}
}

func TestResolveVoiceDownload_CustomModelConfig(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TTS.ModelConfigPath = "custom.yaml"

	_, _, err := resolveVoiceDownload(cfg, "voices", false)
	if err == nil {
		t.Error("resolveVoiceDownload with --model-config = nil error; want error")
	}
}
