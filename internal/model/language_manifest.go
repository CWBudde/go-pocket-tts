package model

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path"
	"slices"
	"strings"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
)

// FlatLayoutLanguage keeps the layout from before per-language models:
// models/tts_b6369a24.safetensors, models/tokenizer.model and voices/*.safetensors
// (config.DefaultLanguage). Every other language downloads
// languages/<lang>/… into its own directory.
const FlatLayoutLanguage = "english_2026-01"

// voiceLicense is the license of VoiceRepo, recorded for each predefined voice.
const voiceLicense = "CC-BY-4.0"

//go:generate go run ./internal/genchecksums -out language_checksums.json

//go:embed language_checksums.json
var languageChecksumsJSON []byte

// languageChecksums maps ChecksumKey to the SHA256 of every ungated file a
// per-language download fetches.
var languageChecksums = mustParseChecksums(languageChecksumsJSON)

func mustParseChecksums(data []byte) map[string]string {
	var sums map[string]string

	err := json.Unmarshal(data, &sums)
	if err != nil {
		panic(fmt.Sprintf("model: parse language_checksums.json: %v", err))
	}

	return sums
}

// ChecksumKey identifies a file at a revision in language_checksums.json.
func ChecksumKey(repo, file, revision string) string {
	return repo + "/" + file + "@" + revision
}

// ParseHFRef splits an upstream "hf://<org>/<repo>/<path>@<revision>"
// reference, as used by the model configs. Downloads must be pinned, so a
// reference without a revision is an error.
func ParseHFRef(ref string) (repo, file, revision string, err error) {
	rest, ok := strings.CutPrefix(ref, "hf://")
	if !ok {
		return "", "", "", fmt.Errorf("%q is not an hf:// reference", ref)
	}

	at := strings.LastIndex(rest, "@")
	if at < 0 || at == len(rest)-1 {
		return "", "", "", fmt.Errorf("%q has no pinned revision", ref)
	}

	rest, revision = rest[:at], rest[at+1:]

	parts := strings.SplitN(rest, "/", 3)
	if len(parts) != 3 || parts[0] == "" || parts[1] == "" || parts[2] == "" {
		return "", "", "", fmt.Errorf("%q has no file path", ref)
	}

	return parts[0] + "/" + parts[1], parts[2], revision, nil
}

// LanguageManifest returns the model files of language from repo (GatedRepo
// or VoiceRepo). FlatLayoutLanguage (or "") keeps PinnedManifest; every other
// language fetches languages/<lang>/model.safetensors and the tokenizer at
// the revisions its embedded model config pins, saved as model.safetensors
// and under the tokenizer's file name (tokenizer.json for every embedded
// config).
func LanguageManifest(language, repo string) (Manifest, error) {
	if language == "" || language == FlatLayoutLanguage {
		return PinnedManifest(repo)
	}

	cfg, err := modelcfg.Lookup(language)
	if err != nil {
		return Manifest{}, err
	}

	var weightsRef string

	switch repo {
	case GatedRepo:
		weightsRef = cfg.WeightsPath
	case VoiceRepo:
		weightsRef = cfg.WeightsPathWithoutVoiceCloning
	default:
		return Manifest{}, fmt.Errorf("no %s download from repo %q (use %s or %s)", language, repo, GatedRepo, VoiceRepo)
	}

	weights, err := pinnedFile(weightsRef, "model.safetensors")
	if err != nil {
		return Manifest{}, fmt.Errorf("%s weights: %w", language, err)
	}

	// The tokenizer always comes from the ungated repo, also for gated weights.
	tokenizer, err := pinnedFile(cfg.FlowLM.LookupTable.TokenizerPath, "")
	if err != nil {
		return Manifest{}, fmt.Errorf("%s tokenizer: %w", language, err)
	}

	return Manifest{Repo: repo, Files: []ModelFile{weights, tokenizer}}, nil
}

// pinnedFile turns a config reference into a ModelFile with its checksum from
// language_checksums.json, saved as localPath (the file's base name when
// empty). Gated files have none there (the tree API masks them); their
// checksum comes from HF metadata at download time.
func pinnedFile(ref, localPath string) (ModelFile, error) {
	repo, file, revision, err := ParseHFRef(ref)
	if err != nil {
		return ModelFile{}, err
	}

	if localPath == "" {
		localPath = path.Base(file)
	}

	sha := languageChecksums[ChecksumKey(repo, file, revision)]
	if sha == "" && repo != GatedRepo {
		return ModelFile{}, fmt.Errorf("no pinned checksum for %s; run go generate ./internal/model/", ChecksumKey(repo, file, revision))
	}

	return ModelFile{Repo: repo, Filename: file, Revision: revision, SHA256: sha, LocalPath: localPath}, nil
}

// VoiceManifestForLanguage returns the predefined voices of language,
// restricted to voices when that is not empty. FlatLayoutLanguage (or "")
// keeps VoiceManifest; every other language fetches
// languages/<lang>/embeddings/<id>.safetensors at the voices revision, saved
// as <id>.safetensors.
func VoiceManifestForLanguage(language string, voices []string) (Manifest, error) {
	m := VoiceManifest()

	if language != "" && language != FlatLayoutLanguage {
		cfg, err := modelcfg.Lookup(language)
		if err != nil {
			return Manifest{}, err
		}

		m = Manifest{Repo: VoiceRepo, Files: languageVoices(language, cfg.VoicesRevision)}
		if len(m.Files) == 0 {
			return Manifest{}, fmt.Errorf("no predefined voices pinned for %s; run go generate ./internal/model/", language)
		}
	}

	return filterVoices(m, voices)
}

// WriteVoiceIndex adds the downloaded voices ids (<id>.safetensors next to
// the manifest) to the voice manifest at path, the file the runtime reads
// (paths.voice_manifest). Existing entries with other ids are kept, so
// repeated downloads accumulate.
func WriteVoiceIndex(path string, ids []string) error {
	var index voiceIndex

	data, err := os.ReadFile(path)

	switch {
	case err == nil:
		err = json.Unmarshal(data, &index)
		if err != nil {
			return fmt.Errorf("decode voice manifest %s: %w", path, err)
		}
	case !errors.Is(err, os.ErrNotExist):
		return fmt.Errorf("read voice manifest: %w", err)
	}

	for _, id := range ids {
		entry := voiceIndexEntry{ID: id, Path: id + ".safetensors", License: voiceLicense}

		i := slices.IndexFunc(index.Voices, func(v voiceIndexEntry) bool { return v.ID == id })
		if i >= 0 {
			index.Voices[i] = entry
		} else {
			index.Voices = append(index.Voices, entry)
		}
	}

	slices.SortFunc(index.Voices, func(a, b voiceIndexEntry) int { return strings.Compare(a.ID, b.ID) })

	out, err := json.MarshalIndent(index, "", "  ")
	if err != nil {
		return fmt.Errorf("encode voice manifest: %w", err)
	}

	err = os.WriteFile(path, append(out, '\n'), 0o600)
	if err != nil {
		return fmt.Errorf("write voice manifest: %w", err)
	}

	return nil
}

func languageVoices(language, revision string) []ModelFile {
	dir := "languages/" + language + "/embeddings/"
	prefix := VoiceRepo + "/" + dir
	suffix := ".safetensors@" + revision

	var files []ModelFile

	for _, key := range slices.Sorted(maps.Keys(languageChecksums)) {
		rest, ok := strings.CutPrefix(key, prefix)
		if !ok || !strings.HasSuffix(rest, suffix) {
			continue
		}

		id := strings.TrimSuffix(rest, suffix)
		files = append(files, ModelFile{
			Repo:      VoiceRepo,
			Filename:  dir + id + ".safetensors",
			Revision:  revision,
			SHA256:    languageChecksums[key],
			LocalPath: id + ".safetensors",
		})
	}

	return files
}

func voiceID(f ModelFile) string {
	return strings.TrimSuffix(f.LocalPath, ".safetensors")
}

func filterVoices(m Manifest, voices []string) (Manifest, error) {
	if len(voices) == 0 {
		return m, nil
	}

	available := make([]string, 0, len(m.Files))
	for _, f := range m.Files {
		available = append(available, voiceID(f))
	}

	for _, v := range voices {
		if !slices.Contains(available, v) {
			return Manifest{}, fmt.Errorf("unknown voice %q (available: %s)", v, strings.Join(available, ", "))
		}
	}

	files := make([]ModelFile, 0, len(voices))
	for _, f := range m.Files {
		if slices.Contains(voices, voiceID(f)) {
			files = append(files, f)
		}
	}

	return Manifest{Repo: m.Repo, Files: files}, nil
}

type voiceIndexEntry struct {
	ID      string `json:"id"`
	Path    string `json:"path"`
	License string `json:"license"`
}

type voiceIndex struct {
	Voices []voiceIndexEntry `json:"voices"`
}
