// Package catalogsrc builds the public model catalog (catalog.json at the
// module root) from the pinned download manifests in internal/model. The
// generator adds file sizes from Hugging Face; the drift test reuses the
// embedded sizes, so it runs offline.
package catalogsrc

import (
	"errors"
	"fmt"
	"path"
	"strings"

	pockettts "github.com/cwbudde/go-pocket-tts"
	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/model"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
)

// Ref names a pinned file on Hugging Face.
type Ref struct {
	Repo     string
	Revision string
	File     string
}

// URL is the revision-pinned download URL of r.
func (r Ref) URL() string {
	return "https://huggingface.co/" + r.Repo + "/resolve/" + r.Revision + "/" + r.File
}

// SizeFunc returns the size in bytes of a pinned file.
type SizeFunc func(r Ref) (int64, error)

// labels names the embedded configs for people; unlisted configs keep their
// config name.
var labels = map[string]string{
	"english_2026-01":        "English (Jan 2026)",
	"english_2026-09":        "English (Sep 2026)",
	"english_2026-09_24l":    "English (Sep 2026), 24 layers",
	"english_drifting_26-09": "English, drifting (Sep 2026)",
	"german":                 "German",
	"german_24l":             "German, 24 layers",
}

// languages maps a config name prefix to its ISO 639-1 code.
var languages = map[string]string{
	"english": "en",
	"german":  "de",
}

// Build returns the catalog of every embedded model config. It uses the
// ungated repo, so every model synthesizes without a Hugging Face token.
func Build(size SizeFunc) (pockettts.Catalog, error) {
	catalog := pockettts.Catalog{Default: config.DefaultLanguage}

	for _, name := range modelcfg.Languages() {
		m, err := buildModel(name, size)
		if err != nil {
			return pockettts.Catalog{}, fmt.Errorf("%s: %w", name, err)
		}

		catalog.Models = append(catalog.Models, m)
	}

	return catalog, nil
}

func buildModel(name string, size SizeFunc) (pockettts.Model, error) {
	cfg, err := modelcfg.Lookup(name)
	if err != nil {
		return pockettts.Model{}, err
	}

	language, ok := languages[strings.SplitN(name, "_", 2)[0]]
	if !ok {
		return pockettts.Model{}, errors.New("unknown language prefix")
	}

	label := labels[name]
	if label == "" {
		label = name
	}

	m := pockettts.Model{
		Name:               name,
		Label:              label,
		Language:           language,
		Layers:             cfg.FlowLM.Transformer.NumLayers,
		DefaultVoice:       cfg.DefaultVoice,
		DefaultTemperature: cfg.DefaultTemperature,
		DefaultText:        cfg.DefaultText,
	}

	m.Weights, m.Tokenizer, err = modelFiles(name, size)
	if err != nil {
		return pockettts.Model{}, err
	}

	m.Voices, err = voiceFiles(name, size)
	if err != nil {
		return pockettts.Model{}, err
	}

	if _, ok := m.Voice(m.DefaultVoice); !ok {
		return pockettts.Model{}, fmt.Errorf("default voice %q is not a predefined voice", m.DefaultVoice)
	}

	return m, nil
}

// modelFiles returns the weights and tokenizer of the config name.
func modelFiles(name string, size SizeFunc) (weights, tokenizer pockettts.File, err error) {
	files, err := model.LanguageManifest(name, model.VoiceRepo)
	if err != nil {
		return weights, tokenizer, err
	}

	for _, f := range files.Files {
		local := f.LocalPath
		if local == "" {
			local = path.Base(f.Filename)
		}

		var target *pockettts.File

		switch {
		case strings.HasSuffix(f.Filename, ".safetensors"):
			target, local = &weights, "model.safetensors"
		case strings.Contains(path.Base(f.Filename), "tokenizer"):
			target = &tokenizer
		default:
			return weights, tokenizer, fmt.Errorf("unexpected model file %s", f.Filename)
		}

		*target, err = file(files, f, name+"/"+local, size)
		if err != nil {
			return weights, tokenizer, err
		}
	}

	if weights.URL == "" || tokenizer.URL == "" {
		return weights, tokenizer, errors.New("model manifest lacks weights or tokenizer")
	}

	return weights, tokenizer, nil
}

// voiceFiles returns the predefined voices of the config name.
func voiceFiles(name string, size SizeFunc) ([]pockettts.VoiceFile, error) {
	voices, err := model.VoiceManifestForLanguage(name, nil)
	if err != nil {
		return nil, err
	}

	out := make([]pockettts.VoiceFile, 0, len(voices.Files))

	for _, f := range voices.Files {
		id := strings.TrimSuffix(f.LocalPath, ".safetensors")

		vf, err := file(voices, f, name+"/voices/"+id+".safetensors", size)
		if err != nil {
			return nil, err
		}

		out = append(out, pockettts.VoiceFile{ID: id, File: vf})
	}

	return out, nil
}

func file(m model.Manifest, f model.ModelFile, local string, size SizeFunc) (pockettts.File, error) {
	repo := f.Repo
	if repo == "" {
		repo = m.Repo
	}

	if f.SHA256 == "" {
		return pockettts.File{}, fmt.Errorf("%s has no pinned checksum", f.Filename)
	}

	ref := Ref{Repo: repo, Revision: f.Revision, File: f.Filename}

	n, err := size(ref)
	if err != nil {
		return pockettts.File{}, fmt.Errorf("size of %s: %w", f.Filename, err)
	}

	return pockettts.File{URL: ref.URL(), SHA256: strings.ToLower(f.SHA256), Size: n, Path: local}, nil
}
