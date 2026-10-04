// Package webmanifest builds the language catalog of the web app
// (web/languages.json): for every embedded model config, the Hugging Face URLs
// of its model, tokenizer and predefined voices at the revisions the CLI
// downloads, plus the defaults the page applies on a switch.
package webmanifest

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/model"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
)

//go:generate go run ./internal/gen -out ../../web/languages.json

// tooLargeForBrowser lists the configs left out of the catalog because they
// do not fit in 32-bit WASM memory (4 GB). Empty since the engine drops the
// checkpoint bytes after decoding: english_2026-09_24l (1.3 GB, mostly F32)
// ran out of memory while they were kept next to the weights.
var tooLargeForBrowser = map[string]bool{}

// Catalog lists the configs the web app offers.
type Catalog struct {
	Default   string     `json:"default"`
	Languages []Language `json:"languages"`
}

// Language is one model config with its files and defaults.
type Language struct {
	Name               string  `json:"name"`
	Layers             int     `json:"layers"`
	DefaultVoice       string  `json:"default_voice"`
	DefaultTemperature float64 `json:"default_temperature"`
	DefaultText        string  `json:"default_text"`
	Model              string  `json:"model"`
	Tokenizer          string  `json:"tokenizer"`
	Voices             []Voice `json:"voices"`
}

// Voice is a predefined voice of a config.
type Voice struct {
	ID  string `json:"id"`
	URL string `json:"url"`
}

// Build returns the catalog of every embedded config that fits in the
// browser (see tooLargeForBrowser). It uses the ungated repo the CLI
// downloads from by default, so english_2026-01 keeps its flat checkpoint,
// and every config gets its tokenizer.json.
func Build() (Catalog, error) {
	catalog := Catalog{Default: config.DefaultLanguage}

	for _, name := range modelcfg.Languages() {
		if tooLargeForBrowser[name] {
			continue
		}

		lang, err := language(name)
		if err != nil {
			return Catalog{}, fmt.Errorf("%s: %w", name, err)
		}

		catalog.Languages = append(catalog.Languages, lang)
	}

	return catalog, nil
}

// JSON returns the catalog as web/languages.json stores it.
func JSON() ([]byte, error) {
	catalog, err := Build()
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer

	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")

	err = enc.Encode(catalog)
	if err != nil {
		return nil, fmt.Errorf("encode catalog: %w", err)
	}

	return buf.Bytes(), nil
}

func language(name string) (Language, error) {
	cfg, err := modelcfg.Lookup(name)
	if err != nil {
		return Language{}, err
	}

	files, err := model.LanguageManifest(name, model.VoiceRepo)
	if err != nil {
		return Language{}, err
	}

	weights, err := modelFile(files)
	if err != nil {
		return Language{}, err
	}

	repo, file, revision, err := model.ParseHFRef(cfg.FlowLM.LookupTable.TokenizerPath)
	if err != nil {
		return Language{}, fmt.Errorf("tokenizer: %w", err)
	}

	voices, err := model.VoiceManifestForLanguage(name, nil)
	if err != nil {
		return Language{}, err
	}

	lang := Language{
		Name:               name,
		Layers:             cfg.FlowLM.Transformer.NumLayers,
		DefaultVoice:       cfg.DefaultVoice,
		DefaultTemperature: cfg.DefaultTemperature,
		DefaultText:        cfg.DefaultText,
		Model:              weights,
		Tokenizer:          resolveURL(repo, revision, file),
	}

	for _, f := range voices.Files {
		lang.Voices = append(lang.Voices, Voice{
			ID:  strings.TrimSuffix(f.LocalPath, ".safetensors"),
			URL: resolveURL(fileRepo(voices, f), f.Revision, f.Filename),
		})
	}

	return lang, nil
}

// modelFile returns the URL of the manifest's safetensors checkpoint.
func modelFile(m model.Manifest) (string, error) {
	for _, f := range m.Files {
		if strings.HasSuffix(f.Filename, ".safetensors") {
			return resolveURL(fileRepo(m, f), f.Revision, f.Filename), nil
		}
	}

	return "", errors.New("no safetensors checkpoint in the model manifest")
}

func fileRepo(m model.Manifest, f model.ModelFile) string {
	if f.Repo != "" {
		return f.Repo
	}

	return m.Repo
}

func resolveURL(repo, revision, file string) string {
	return "https://huggingface.co/" + repo + "/resolve/" + revision + "/" + file
}
