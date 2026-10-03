package main

import (
	"io"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
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

func TestModelDownloadVoices(t *testing.T) {
	tests := []struct {
		name     string
		language string
		voices   []string
		all      bool
		want     []string
	}{
		{"default voice of german", "german", nil, false, []string{"juergen"}},
		{"default voice of the flat language", config.DefaultLanguage, nil, false, []string{"alba"}},
		{"german_24l shares the german default", "german_24l", nil, false, []string{"juergen"}},
		{"explicit voices win", "german", []string{"alba", "juergen"}, false, []string{"alba", "juergen"}},
		{"all voices", "german", nil, true, nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := config.DefaultConfig()
			cfg.TTS.Language = tc.language

			mc, err := modelcfg.Lookup(tc.language)
			if err != nil {
				t.Fatal(err)
			}

			cfg.Model = mc

			got, err := modelDownloadVoices(cfg, tc.voices, tc.all)
			if err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(got, tc.want) || (got == nil) != (tc.want == nil) {
				t.Errorf("modelDownloadVoices = %#v; want %#v", got, tc.want)
			}
		})
	}
}

func TestModelDownloadCmd_VoiceFlagsAreExclusive(t *testing.T) {
	for _, args := range [][]string{
		{"--no-voices", "--voice", "juergen"},
		{"--no-voices", "--all-voices"},
		{"--all-voices", "--voice", "juergen"},
	} {
		cmd := newModelDownloadCmd()
		cmd.SetArgs(args)
		cmd.SetOut(io.Discard)
		cmd.SetErr(io.Discard)

		err := cmd.Execute()
		if err == nil || !strings.Contains(err.Error(), "none of the others can be") {
			t.Errorf("model download %v: error = %v; want a mutually exclusive flags error", args, err)
		}
	}
}
