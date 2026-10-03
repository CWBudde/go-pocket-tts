package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
)

// Upstream generate falls back to get_default_voice_for_language; without a
// voice the native model stops almost at once.
func TestResolveNativeVoice_DefaultVoice(t *testing.T) {
	manifestPath, voiceFile := writeVoiceManifest(t)

	cfg := config.DefaultConfig()
	cfg.Paths.VoiceManifest = manifestPath
	cfg.Model = &modelcfg.ModelConfig{DefaultVoice: "alice"}

	got, err := resolveNativeVoice(cfg, "")
	if err != nil || got != voiceFile {
		t.Errorf("resolveNativeVoice(no voice) = %q, %v; want the default voice %q", got, err, voiceFile)
	}

	explicit := filepath.Join("elsewhere", "bob.safetensors")

	got, err = resolveNativeVoice(cfg, explicit)
	if err != nil || got != explicit {
		t.Errorf("resolveNativeVoice(%q) = %q, %v; want the explicit voice", explicit, got, err)
	}
}

func TestResolveNativeVoice_MissingDefaultVoiceFails(t *testing.T) {
	manifestPath, _ := writeVoiceManifest(t)

	for name, manifest := range map[string]string{
		"not in manifest":  manifestPath,
		"missing manifest": filepath.Join(t.TempDir(), "manifest.json"),
	} {
		cfg := config.DefaultConfig()
		cfg.Paths.VoiceManifest = manifest
		cfg.Model = &modelcfg.ModelConfig{DefaultVoice: "juergen"}

		got, err := resolveNativeVoice(cfg, "")
		if err == nil || !strings.Contains(err.Error(), "default voice") || !strings.Contains(err.Error(), `"juergen"`) {
			t.Errorf("%s: resolveNativeVoice = %q, %v; want an error naming the default voice juergen", name, got, err)
		}
	}
}

// Without a model config (tests, DefaultConfig) there is no default voice.
func TestResolveNativeVoice_NoModelConfigKeepsEmptyVoice(t *testing.T) {
	cfg := config.DefaultConfig()

	got, err := resolveNativeVoice(cfg, "")
	if err != nil || got != "" {
		t.Errorf("resolveNativeVoice without model config = %q, %v; want empty", got, err)
	}
}
