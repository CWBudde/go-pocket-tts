package main

import (
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
)

func TestResolveVoiceDownload(t *testing.T) {
	tests := []struct {
		name          string
		language      string
		voiceManifest string // paths.voice_manifest; "" keeps the language default
		outDir        string
		outDirSet     bool
		want          voiceDownloadTarget
	}{
		{
			name: "default language keeps the tracked flat manifest", language: config.DefaultLanguage, outDir: "voices",
			want: voiceDownloadTarget{language: config.DefaultLanguage, dir: "voices", manifest: filepath.Join("voices", "manifest.json")},
		},
		{
			name: "other language gets voices/<lang>", language: "german", outDir: "voices",
			want: voiceDownloadTarget{
				language: "german", dir: filepath.Join("voices", "german"),
				manifest: filepath.Join("voices", "german", "manifest.json"), writeIndex: true,
			},
		},
		{
			name: "configured voice manifest is the destination", language: "german", voiceManifest: "/data/de/stimmen.json", outDir: "voices",
			want: voiceDownloadTarget{language: "german", dir: "/data/de", manifest: "/data/de/stimmen.json", writeIndex: true},
		},
		{
			name: "configured manifest for the flat language gets an index too", language: config.DefaultLanguage,
			voiceManifest: "/data/en/manifest.json", outDir: "voices",
			want: voiceDownloadTarget{language: config.DefaultLanguage, dir: "/data/en", manifest: "/data/en/manifest.json", writeIndex: true},
		},
		{
			name: "explicit out-dir wins", language: "german", voiceManifest: "/data/de/stimmen.json", outDir: "/tmp/v", outDirSet: true,
			want: voiceDownloadTarget{language: "german", dir: "/tmp/v", manifest: filepath.Join("/tmp/v", "manifest.json"), writeIndex: true},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := config.DefaultConfig()
			cfg.TTS.Language = tc.language

			cfg.Paths.VoiceManifest = config.PathsForLanguage(tc.language).VoiceManifest
			if tc.voiceManifest != "" {
				cfg.Paths.VoiceManifest = tc.voiceManifest
			}

			got, err := resolveVoiceDownload(cfg, tc.outDir, tc.outDirSet)
			if err != nil {
				t.Fatal(err)
			}

			if got != tc.want {
				t.Errorf("resolveVoiceDownload = %+v; want %+v", got, tc.want)
			}
		})
	}
}

func TestResolveVoiceDownload_CustomModelConfig(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TTS.ModelConfigPath = "custom.yaml"

	_, err := resolveVoiceDownload(cfg, "voices", false)
	if err == nil {
		t.Error("resolveVoiceDownload with --model-config = nil error; want error")
	}
}
