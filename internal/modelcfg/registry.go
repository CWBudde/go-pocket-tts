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

// defaultTextForLanguage mirrors upstream DEFAULT_TEXT_FOR_LANGUAGE, in its
// order, since the first substring match wins. The first entry is the
// fallback (upstream: DEFAULT_LANGUAGE).
var defaultTextForLanguage = []struct{ key, text string }{
	{"english", "Hello world. I am Kyutai's Pocket TTS. " +
		"I'm fast enough to run on small CPUs. " +
		"I hope you'll like me."},
	{"french", "Bonjour le monde. Je suis le TTS de poche de Kyutai. " +
		"Je suis assez rapide pour fonctionner sur de petits CPU. " +
		"J'espère que vous m'aimerez."},
	{"german", "Hallo Welt. Ich bin Pocket TTS von Kyutai. " +
		"Ich bin schnell genug, um auch auf kleinen CPUs zu laufen. " +
		"Ich hoffe, ich gefalle dir."},
	{"portuguese", "Olá mundo. Eu sou o Pocket TTS da Kyutai. " +
		"Sou rápido o suficiente para rodar em CPUs pequenas. " +
		"Espero que você goste de mim."},
	{"italian", "Ciao mondo. Sono il Pocket TTS di Kyutai. " +
		"Sono abbastanza veloce da funzionare su piccole CPU. " +
		"Spero che ti piacerò."},
	{"spanish", "Hola mundo. Soy el Pocket TTS de Kyutai. " +
		"Soy lo suficientemente rápido para funcionar en pequeñas CPU. " +
		"Espero que te guste."},
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
	cfg.DefaultText = defaultTextFor(language)
	cfg.VoicesRevision = VoicesRevision

	return cfg, nil
}

// LoadCustom reads a model config file that is not one of the embedded
// languages (upstream: --config). Like upstream, its default voice is the
// fallback voice, which any model can clone, and its default text English.
func LoadCustom(path string) (*ModelConfig, error) {
	cfg, err := Load(path)
	if err != nil {
		return nil, err
	}

	cfg.DefaultVoice = defaultVoiceFallback
	cfg.DefaultText = defaultTextFor("")
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

// defaultTextFor matches language by substring like upstream
// get_default_text_for_language and falls back to the first (English) text.
func defaultTextFor(language string) string {
	for _, entry := range defaultTextForLanguage {
		if language != "" && strings.Contains(language, entry.key) {
			return entry.text
		}
	}

	return defaultTextForLanguage[0].text
}
