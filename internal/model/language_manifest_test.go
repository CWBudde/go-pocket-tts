package model

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
)

const (
	germanWeightsRev   = "1e08e6a23401048648a9fdcfde2f89348215c2a7"
	germanTokenizerRev = "4e1e0a3e611c51c0b4ed8174fc10f32a54644303"
	germanGatedRev     = "3e82814a68665eec246ff649b14c71331f955c06"
)

func TestParseHFRef(t *testing.T) {
	tests := []struct {
		ref, repo, path, rev string
	}{
		{
			"hf://kyutai/pocket-tts/languages/german/model.safetensors@3e82",
			"kyutai/pocket-tts", "languages/german/model.safetensors", "3e82",
		},
		{
			"hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model@00ea",
			"kyutai/pocket-tts-without-voice-cloning", "tokenizer.model", "00ea",
		},
	}
	for _, tc := range tests {
		repo, path, rev, err := ParseHFRef(tc.ref)
		if err != nil {
			t.Fatalf("ParseHFRef(%q): %v", tc.ref, err)
		}

		if repo != tc.repo || path != tc.path || rev != tc.rev {
			t.Errorf("ParseHFRef(%q) = %q, %q, %q; want %q, %q, %q",
				tc.ref, repo, path, rev, tc.repo, tc.path, tc.rev)
		}
	}

	for _, bad := range []string{
		"kyutai/pocket-tts/model.safetensors@r",     // no scheme
		"hf://kyutai/pocket-tts/model.safetensors",  // unpinned
		"hf://kyutai/pocket-tts@r",                  // no file path
		"hf://kyutai/pocket-tts/@r",                 // empty file path
		"hf://kyutai/pocket-tts/model.safetensors@", // empty revision
	} {
		_, _, _, err := ParseHFRef(bad)
		if err == nil {
			t.Errorf("ParseHFRef(%q) = nil error; want error", bad)
		}
	}
}

func TestFlatLayoutLanguage_IsConfigDefault(t *testing.T) {
	if FlatLayoutLanguage != config.DefaultLanguage {
		t.Errorf("FlatLayoutLanguage = %q; want config.DefaultLanguage %q", FlatLayoutLanguage, config.DefaultLanguage)
	}
}

func TestLanguageManifest_FlatLanguageIsPinnedManifest(t *testing.T) {
	for _, repo := range []string{GatedRepo, VoiceRepo} {
		want, err := PinnedManifest(repo)
		if err != nil {
			t.Fatal(err)
		}

		for _, language := range []string{"", FlatLayoutLanguage} {
			got, err := LanguageManifest(language, repo)
			if err != nil {
				t.Fatalf("LanguageManifest(%q, %q): %v", language, repo, err)
			}

			if !reflect.DeepEqual(got, want) {
				t.Errorf("LanguageManifest(%q, %q) = %+v; want PinnedManifest %+v", language, repo, got, want)
			}
		}
	}
}

func TestLanguageManifest_GermanUngated(t *testing.T) {
	got, err := LanguageManifest("german", VoiceRepo)
	if err != nil {
		t.Fatal(err)
	}

	want := Manifest{
		Repo: VoiceRepo,
		Files: []ModelFile{
			{
				Repo:      VoiceRepo,
				Filename:  "languages/german/model.safetensors",
				Revision:  germanWeightsRev,
				SHA256:    "9fe42605604832349e3f495177bde56b77ae8fee73b2ed4b63236092ccde7621",
				LocalPath: "model.safetensors",
			},
			{
				Repo:      VoiceRepo,
				Filename:  "languages/german/tokenizer.json",
				Revision:  germanTokenizerRev,
				SHA256:    "2d77849811e3d1b6ae80e5b11b203f57e1847901104d8a0573b34f46f8dddb53",
				LocalPath: "tokenizer.json",
			},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("LanguageManifest(german, ungated) =\n%+v\nwant\n%+v", got, want)
	}
}

func TestLanguageManifest_GermanGated(t *testing.T) {
	got, err := LanguageManifest("german", GatedRepo)
	if err != nil {
		t.Fatal(err)
	}

	weights := got.Files[0]
	if weights.Repo != GatedRepo || weights.Revision != germanGatedRev ||
		weights.Filename != "languages/german/model.safetensors" || weights.SHA256 != "" {
		t.Errorf("gated weights = %+v; want gated repo @%s with checksum from metadata", weights, germanGatedRev)
	}

	// The gated repo has no tokenizer of its own; it stays the ungated pin.
	tokenizer := got.Files[1]
	if tokenizer.Repo != VoiceRepo || tokenizer.Revision != germanTokenizerRev || !isSHA256Hex(tokenizer.SHA256) {
		t.Errorf("gated tokenizer = %+v; want pinned ungated file", tokenizer)
	}
}

// TestLanguageManifest_AllLanguagesPinned guards language_checksums.json
// against the embedded configs: every non-flat language must resolve to fully
// pinned ungated files.
func TestLanguageManifest_AllLanguagesPinned(t *testing.T) {
	for _, language := range modelcfg.Languages() {
		if language == FlatLayoutLanguage {
			continue
		}

		m, err := LanguageManifest(language, VoiceRepo)
		if err != nil {
			t.Errorf("LanguageManifest(%q): %v", language, err)
			continue
		}

		locals := make([]string, 0, len(m.Files))
		for _, f := range m.Files {
			locals = append(locals, f.LocalPath)

			if !isSHA256Hex(f.SHA256) {
				t.Errorf("%s: %s has no pinned checksum", language, f.Filename)
			}

			// The tokenizer may be another language's: english_drifting_26-09
			// uses the english_2026-09 one.
			dir := "languages/"
			if f.LocalPath == "model.safetensors" {
				dir += language + "/"
			}

			if !strings.HasPrefix(f.Filename, dir) {
				t.Errorf("%s: %s is outside %s", language, f.Filename, dir)
			}
		}

		if !slices.Equal(locals, []string{"model.safetensors", "tokenizer.json"}) {
			t.Errorf("%s: local files = %v", language, locals)
		}

		voices, err := VoiceManifestForLanguage(language, nil)
		if err != nil {
			t.Errorf("VoiceManifestForLanguage(%q): %v", language, err)
			continue
		}

		if len(voices.Files) == 0 {
			t.Errorf("%s: no predefined voices", language)
		}
	}
}

// TestLanguageChecksums_ConfigPinsMatchVoicesRevision checks that the
// per-file revisions in the embedded configs serve the same blobs as the
// voices revision the plan names, so the manifest is "at" that revision.
func TestLanguageChecksums_ConfigPinsMatchVoicesRevision(t *testing.T) {
	for _, language := range modelcfg.Languages() {
		if language == FlatLayoutLanguage {
			continue
		}

		cfg, err := modelcfg.Lookup(language)
		if err != nil {
			t.Fatal(err)
		}

		for _, ref := range []string{cfg.WeightsPathWithoutVoiceCloning, cfg.FlowLM.LookupTable.TokenizerPath} {
			repo, path, rev, err := ParseHFRef(ref)
			if err != nil {
				t.Fatal(err)
			}

			pinned := languageChecksums[ChecksumKey(repo, path, rev)]
			atVoices := languageChecksums[ChecksumKey(repo, path, modelcfg.VoicesRevision)]

			if pinned == "" || pinned != atVoices {
				t.Errorf("%s: %s@%s = %q, @%s = %q; want equal", language, path, rev, pinned, modelcfg.VoicesRevision, atVoices)
			}
		}
	}
}

func TestLanguageManifest_Errors(t *testing.T) {
	_, err := LanguageManifest("klingon", VoiceRepo)
	if err == nil || !strings.Contains(err.Error(), "klingon") {
		t.Errorf("unknown language: err = %v", err)
	}

	_, err = LanguageManifest("german", "other/repo")
	if err == nil || !strings.Contains(err.Error(), "other/repo") {
		t.Errorf("unknown repo: err = %v", err)
	}

	_, err = LanguageManifest(FlatLayoutLanguage, "other/repo")
	if err == nil {
		t.Error("flat language with unknown repo: err = nil")
	}
}

func TestVoiceManifestForLanguage_Flat(t *testing.T) {
	for _, language := range []string{"", FlatLayoutLanguage} {
		got, err := VoiceManifestForLanguage(language, nil)
		if err != nil {
			t.Fatal(err)
		}

		if !reflect.DeepEqual(got, VoiceManifest()) {
			t.Errorf("VoiceManifestForLanguage(%q, nil) differs from VoiceManifest()", language)
		}
	}

	got, err := VoiceManifestForLanguage(FlatLayoutLanguage, []string{"alba"})
	if err != nil {
		t.Fatal(err)
	}

	if len(got.Files) != 1 || got.Files[0].LocalPath != "alba.safetensors" {
		t.Errorf("filtered flat voices = %+v; want only alba", got.Files)
	}
}

func TestVoiceManifestForLanguage_German(t *testing.T) {
	all, err := VoiceManifestForLanguage("german", nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(all.Files) != 27 {
		t.Errorf("german voices = %d; want the 27 upstream predefined voices", len(all.Files))
	}

	got, err := VoiceManifestForLanguage("german", []string{"juergen", "alba"})
	if err != nil {
		t.Fatal(err)
	}

	want := []ModelFile{
		{
			Repo:      VoiceRepo,
			Filename:  "languages/german/embeddings/alba.safetensors",
			Revision:  modelcfg.VoicesRevision,
			SHA256:    languageChecksums[ChecksumKey(VoiceRepo, "languages/german/embeddings/alba.safetensors", modelcfg.VoicesRevision)],
			LocalPath: "alba.safetensors",
		},
		{
			Repo:      VoiceRepo,
			Filename:  "languages/german/embeddings/juergen.safetensors",
			Revision:  modelcfg.VoicesRevision,
			SHA256:    "65104a25b2aa797c50b2cee555cb75deb6abba10169168b95857227a054126d1",
			LocalPath: "juergen.safetensors",
		},
	}
	if got.Repo != VoiceRepo || !reflect.DeepEqual(got.Files, want) {
		t.Errorf("filtered german voices =\n%+v\nwant\n%+v", got.Files, want)
	}

	if !isSHA256Hex(want[0].SHA256) {
		t.Errorf("alba checksum %q not pinned", want[0].SHA256)
	}

	_, err = VoiceManifestForLanguage("german", []string{"nobody"})
	if err == nil || !strings.Contains(err.Error(), "juergen") {
		t.Errorf("unknown voice: err = %v; want error listing the valid voices", err)
	}

	_, err = VoiceManifestForLanguage("klingon", nil)
	if err == nil {
		t.Error("unknown language: err = nil")
	}
}

// swapLanguageChecksums replaces the pinned table for one test.
func swapLanguageChecksums(t *testing.T, sums map[string]string) {
	t.Helper()

	orig := languageChecksums
	languageChecksums = sums

	t.Cleanup(func() { languageChecksums = orig })
}

func TestDownload_LanguageLayout(t *testing.T) {
	modelPath := "languages/german/model.safetensors"
	tokenizerPath := "languages/german/tokenizer.json"

	// The fake server answers each file with its own URL path, so the
	// checksums below are those of the paths requested.
	weightsURL := "/" + VoiceRepo + "/resolve/" + germanWeightsRev + "/" + modelPath
	tokenizerURL := "/" + VoiceRepo + "/resolve/" + germanTokenizerRev + "/" + tokenizerPath
	swapLanguageChecksums(t, map[string]string{
		ChecksumKey(VoiceRepo, modelPath, germanWeightsRev):       sha256hex([]byte(weightsURL)),
		ChecksumKey(VoiceRepo, tokenizerPath, germanTokenizerRev): sha256hex([]byte(tokenizerURL)),
	})

	var (
		mu        sync.Mutex
		requested []string
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()

		requested = append(requested, r.Method+" "+r.URL.Path)

		_, _ = w.Write([]byte(r.URL.Path))
	}))
	defer srv.Close()

	withHFTransport(t, srv.URL)

	outDir := filepath.Join(t.TempDir(), "models", "german")

	err := Download(DownloadOptions{Repo: VoiceRepo, Language: "german", OutDir: outDir})
	if err != nil {
		t.Fatal(err)
	}

	if want := []string{"GET " + weightsURL, "GET " + tokenizerURL}; !slices.Equal(requested, want) {
		t.Errorf("requests = %v; want %v", requested, want)
	}

	for name, body := range map[string]string{"model.safetensors": weightsURL, "tokenizer.json": tokenizerURL} {
		b, err := os.ReadFile(filepath.Join(outDir, name))
		if err != nil || string(b) != body {
			t.Errorf("%s = %q, %v; want %q", name, b, err, body)
		}
	}

	lock := readLockManifest(filepath.Join(outDir, "download-manifest.lock.json"))
	if rec := lock.Files[modelPath]; rec.Revision != germanWeightsRev || rec.Repo != VoiceRepo {
		t.Errorf("lock[%s] = %+v; want revision %s from %s", modelPath, rec, germanWeightsRev, VoiceRepo)
	}
}

func TestDownload_LanguageGatedUsesPerFileRepo(t *testing.T) {
	modelPath := "languages/german/model.safetensors"
	tokenizerPath := "languages/german/tokenizer.json"
	weightsURL := "/" + GatedRepo + "/resolve/" + germanGatedRev + "/" + modelPath
	tokenizerURL := "/" + VoiceRepo + "/resolve/" + germanTokenizerRev + "/" + tokenizerPath

	swapLanguageChecksums(t, map[string]string{
		ChecksumKey(VoiceRepo, tokenizerPath, germanTokenizerRev): sha256hex([]byte(tokenizerURL)),
	})

	var (
		mu        sync.Mutex
		requested []string
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()

		requested = append(requested, r.Method+" "+r.URL.Path)

		// Gated weights carry no pinned checksum: it comes from metadata.
		w.Header().Set("X-Linked-Etag", `"`+sha256hex([]byte(r.URL.Path))+`"`)

		if r.Method == http.MethodGet {
			_, _ = w.Write([]byte(r.URL.Path))
		}
	}))
	defer srv.Close()

	withHFTransport(t, srv.URL)

	err := Download(DownloadOptions{Repo: GatedRepo, Language: "german", OutDir: t.TempDir(), HFToken: "tok"})
	if err != nil {
		t.Fatal(err)
	}

	want := []string{"HEAD " + weightsURL, "GET " + weightsURL, "GET " + tokenizerURL}
	if !slices.Equal(requested, want) {
		t.Errorf("requests = %v; want %v", requested, want)
	}
}

func TestWriteVoiceIndex_MergesByID(t *testing.T) {
	// Any manifest file name works: it is the configured paths.voice_manifest.
	manifest := filepath.Join(t.TempDir(), "stimmen.json")

	err := WriteVoiceIndex(manifest, []string{"juergen"})
	if err != nil {
		t.Fatal(err)
	}

	err = WriteVoiceIndex(manifest, []string{"alba", "juergen"})
	if err != nil {
		t.Fatal(err)
	}

	vm, err := tts.NewVoiceManager(manifest)
	if err != nil {
		t.Fatal(err)
	}

	voices := vm.ListVoices()

	ids := make([]string, 0, len(voices))
	for _, v := range voices {
		ids = append(ids, v.ID)

		if v.Path != v.ID+".safetensors" || v.License != "CC-BY-4.0" {
			t.Errorf("voice %+v; want path %s.safetensors, license CC-BY-4.0", v, v.ID)
		}
	}

	if !slices.Equal(ids, []string{"alba", "juergen"}) {
		t.Errorf("voice ids = %v; want [alba juergen]", ids)
	}
}

func TestWriteVoiceIndex_KeepsForeignEntries(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manifest.json")

	err := os.WriteFile(path, []byte(`{"voices":[{"id":"mine","path":"/abs/mine.safetensors","license":"MIT"}]}`), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	err = WriteVoiceIndex(path, []string{"alba"})
	if err != nil {
		t.Fatal(err)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	var got struct {
		Voices []struct {
			ID, Path, License string
		} `json:"voices"`
	}

	err = json.Unmarshal(b, &got)
	if err != nil {
		t.Fatal(err)
	}

	if len(got.Voices) != 2 || got.Voices[1].ID != "mine" || got.Voices[1].License != "MIT" {
		t.Errorf("manifest = %s; want alba plus the untouched entry mine", b)
	}
}
