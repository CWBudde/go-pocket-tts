package pockettts

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"slices"
)

//go:generate go run ./internal/gencatalog -out catalog.json

// catalogJSON is the generated model catalog. TestCatalogMatchesManifests
// checks it against the pinned download manifests offline; the sizes come
// from the Hugging Face tree API when it is regenerated.
//
//go:embed catalog.json
var catalogJSON []byte

// Catalog lists every embedded model config with its files.
type Catalog struct {
	// Default names the model used when none is chosen.
	Default string  `json:"default"`
	Models  []Model `json:"models"`
}

// Model is one model config: its weights, tokenizer, predefined voices and
// the defaults synthesis starts from.
type Model struct {
	Name string `json:"name"`
	// Label is a human-readable name, e.g. "German, 24 layers".
	Label string `json:"label"`
	// Language is the ISO 639-1 code of the language the model speaks.
	Language           string      `json:"language"`
	Layers             int         `json:"layers"`
	DefaultVoice       string      `json:"default_voice"`
	DefaultTemperature float64     `json:"default_temperature"`
	DefaultText        string      `json:"default_text"`
	Weights            File        `json:"weights"`
	Tokenizer          File        `json:"tokenizer"`
	Voices             []VoiceFile `json:"voices"`
}

// File is a pinned download: its revision-pinned URL, SHA-256 and size, and
// the slash-separated path it has below a model root (see LoadDir).
type File struct {
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
	Path   string `json:"path"`
}

// VoiceFile is a predefined voice of a model.
type VoiceFile struct {
	File

	ID string `json:"id"`
}

// CatalogJSON returns the catalog as JSON, for callers that hand it on
// unchanged (e.g. to a web page).
func CatalogJSON() []byte {
	return slices.Clone(catalogJSON)
}

// LoadCatalog decodes the embedded catalog; every call returns a fresh copy.
func LoadCatalog() (Catalog, error) {
	var catalog Catalog

	err := json.Unmarshal(catalogJSON, &catalog)
	if err != nil {
		return Catalog{}, fmt.Errorf("pockettts: decode catalog: %w", err)
	}

	return catalog, nil
}

// LookupModel returns the catalog entry of the model called name.
func LookupModel(name string) (Model, error) {
	catalog, err := LoadCatalog()
	if err != nil {
		return Model{}, err
	}

	names := make([]string, len(catalog.Models))

	for i, m := range catalog.Models {
		if m.Name == name {
			return m, nil
		}

		names[i] = m.Name
	}

	return Model{}, fmt.Errorf("pockettts: unknown model %q (available: %v)", name, names)
}

// Files returns every file of m: weights, tokenizer and all voices.
func (m Model) Files() []File {
	files := make([]File, 0, 2+len(m.Voices))
	files = append(files, m.Weights, m.Tokenizer)

	for _, v := range m.Voices {
		files = append(files, v.File)
	}

	return files
}

// Voice returns the predefined voice id of m.
func (m Model) Voice(id string) (VoiceFile, bool) {
	i := slices.IndexFunc(m.Voices, func(v VoiceFile) bool { return v.ID == id })
	if i < 0 {
		return VoiceFile{}, false
	}

	return m.Voices[i], true
}
