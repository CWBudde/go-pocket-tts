package main

import (
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
)

func TestResolveModelDownload(t *testing.T) {
	tests := []struct {
		name      string
		language  string
		outDir    string
		outDirSet bool
		wantDir   string
	}{
		{"default language keeps flat models/", config.DefaultLanguage, "models", false, "models"},
		{"other language gets models/<lang>", "german", "models", false, filepath.Join("models", "german")},
		{"explicit out-dir wins", "german", "/data/de", true, "/data/de"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := config.DefaultConfig()
			cfg.TTS.Language = tc.language

			language, dir, err := resolveModelDownload(cfg, tc.outDir, tc.outDirSet)
			if err != nil {
				t.Fatal(err)
			}

			if language != tc.language || dir != tc.wantDir {
				t.Errorf("resolveModelDownload = %q, %q; want %q, %q", language, dir, tc.language, tc.wantDir)
			}
		})
	}
}

func TestResolveModelDownload_CustomModelConfig(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TTS.ModelConfigPath = "custom.yaml"

	_, _, err := resolveModelDownload(cfg, "models", false)
	if err == nil {
		t.Error("resolveModelDownload with --model-config = nil error; want error")
	}
}
