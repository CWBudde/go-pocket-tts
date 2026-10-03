package modelcfg

import (
	"embed"
	"fmt"
	"io/fs"
	"path"
	"slices"
	"strings"
)

// VoicesRevision is the revision of kyutai/pocket-tts-without-voice-cloning
// that holds the predefined voice embeddings (upstream: get_predefined_voice).
const VoicesRevision = "1e08e6a23401048648a9fdcfde2f89348215c2a7"

// defaultVoiceFallback is used when no entry of defaultVoiceForLanguage
// matches (upstream: DEFAULT_VOICE_FALLBACK).
const defaultVoiceFallback = "alba"

// defaultVoiceForLanguage mirrors upstream DEFAULT_VOICE_FOR_LANGUAGE.
var defaultVoiceForLanguage = []struct{ key, voice string }{
	{"italian", "giovanni"},
	{"spanish", "lola"},
	{"german", "juergen"},
	{"portuguese", "rafael"},
	{"french", "estelle"},
	{"dutch", "daan"},
}

//go:embed configs/*.yaml
var embeddedConfigs embed.FS

// Languages returns the names of the embedded model configs, sorted.
func Languages() []string {
	entries, err := fs.ReadDir(embeddedConfigs, "configs")
	if err != nil {
		panic(fmt.Sprintf("modelcfg: read embedded configs: %v", err))
	}

	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, strings.TrimSuffix(entry.Name(), ".yaml"))
	}

	slices.Sort(names)

	return names
}

// Lookup parses the embedded model config for language and fills in the
// Go-only fields DefaultVoice and VoicesRevision. Every call returns a fresh
// copy.
func Lookup(language string) (*ModelConfig, error) {
	if !slices.Contains(Languages(), language) {
		return nil, fmt.Errorf("modelcfg: unknown language %q (available: %s)",
			language, strings.Join(Languages(), ", "))
	}

	data, err := embeddedConfigs.ReadFile(path.Join("configs", language+".yaml"))
	if err != nil {
		return nil, fmt.Errorf("modelcfg: read embedded %s: %w", language, err)
	}

	cfg, err := Parse(data)
	if err != nil {
		return nil, fmt.Errorf("modelcfg: embedded %s: %w", language, err)
	}

	cfg.DefaultVoice = defaultVoiceFor(language)
	cfg.VoicesRevision = VoicesRevision

	return cfg, nil
}

// LoadCustom reads a model config file that is not one of the embedded
// languages (upstream: --config). Like upstream, its default voice is the
// fallback voice, which any model can clone.
func LoadCustom(path string) (*ModelConfig, error) {
	cfg, err := Load(path)
	if err != nil {
		return nil, err
	}

	cfg.DefaultVoice = defaultVoiceFallback
	cfg.VoicesRevision = VoicesRevision

	return cfg, nil
}

// defaultVoiceFor matches language by substring like upstream
// get_default_voice_for_language, so "german_24l" maps to the german voice.
func defaultVoiceFor(language string) string {
	for _, entry := range defaultVoiceForLanguage {
		if strings.Contains(language, entry.key) {
			return entry.voice
		}
	}

	return defaultVoiceFallback
}
