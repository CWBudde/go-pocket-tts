package webmanifest

import (
	"bytes"
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
)

var pinnedURL = regexp.MustCompile(`^https://huggingface\.co/kyutai/pocket-tts-without-voice-cloning/resolve/[0-9a-f]{40}/\S+$`)

func TestBuild_EveryConfigPinned(t *testing.T) {
	catalog, err := Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if catalog.Default != config.DefaultLanguage {
		t.Errorf("default = %q, want %q", catalog.Default, config.DefaultLanguage)
	}

	names := make([]string, 0, len(catalog.Languages))
	for _, lang := range catalog.Languages {
		names = append(names, lang.Name)
	}

	want := slices.DeleteFunc(modelcfg.Languages(), func(name string) bool { return tooLargeForBrowser[name] })
	if !slices.Equal(names, want) || len(names) != len(modelcfg.Languages())-len(tooLargeForBrowser) {
		t.Fatalf("languages = %v, want %v", names, want)
	}

	for _, lang := range catalog.Languages {
		urls := make([]string, 0, 2+len(lang.Voices))
		urls = append(urls, lang.Model, lang.Tokenizer)

		for _, v := range lang.Voices {
			urls = append(urls, v.URL)
		}

		for _, u := range urls {
			if !pinnedURL.MatchString(u) {
				t.Errorf("%s: %q is not a pinned Hugging Face URL", lang.Name, u)
			}
		}

		if !strings.HasSuffix(lang.Model, ".safetensors") {
			t.Errorf("%s: model %q is not a safetensors file", lang.Name, lang.Model)
		}

		if !strings.HasSuffix(lang.Tokenizer, "/tokenizer.json") {
			t.Errorf("%s: tokenizer %q is not a tokenizer.json", lang.Name, lang.Tokenizer)
		}

		if !slices.ContainsFunc(lang.Voices, func(v Voice) bool { return v.ID == lang.DefaultVoice }) {
			t.Errorf("%s: default voice %q is not among its %d voices", lang.Name, lang.DefaultVoice, len(lang.Voices))
		}

		if lang.Layers <= 0 || lang.DefaultTemperature <= 0 || lang.DefaultText == "" {
			t.Errorf("%s: layers %d, temperature %v, text %q", lang.Name, lang.Layers, lang.DefaultTemperature, lang.DefaultText)
		}
	}
}

func TestBuild_Values(t *testing.T) {
	catalog, err := Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	byName := map[string]Language{}
	for _, lang := range catalog.Languages {
		byName[lang.Name] = lang
	}

	// english_2026-01 keeps the flat checkpoint the CLI downloads.
	en := byName["english_2026-01"]
	if !strings.HasSuffix(en.Model, "/resolve/d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3/tts_b6369a24.safetensors") {
		t.Errorf("english_2026-01 model = %q", en.Model)
	}

	if !strings.HasSuffix(en.Tokenizer, "/resolve/00eac05ed3d16bdc3f6b5d598874019c34a89214/tokenizer.json") {
		t.Errorf("english_2026-01 tokenizer = %q", en.Tokenizer)
	}

	de := byName["german"]
	if de.DefaultVoice != "juergen" || de.Layers != 6 || !strings.HasPrefix(de.DefaultText, "Hallo Welt.") {
		t.Errorf("german = voice %q, layers %d, text %q", de.DefaultVoice, de.Layers, de.DefaultText)
	}

	if !strings.HasSuffix(de.Model, "/languages/german/model.safetensors") {
		t.Errorf("german model = %q", de.Model)
	}

	if got := byName["german_24l"].Layers; got != 24 {
		t.Errorf("german_24l layers = %d, want 24", got)
	}

	if got := byName["english_2026-09_24l"].Layers; got != 24 {
		t.Errorf("english_2026-09_24l layers = %d, want 24 (is it offered?)", got)
	}
}

// TestCatalogFileUpToDate fails when web/languages.json no longer matches the
// embedded configs; run go generate ./internal/webmanifest/.
func TestCatalogFileUpToDate(t *testing.T) {
	want, err := JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}

	got, err := os.ReadFile("../../web/languages.json")
	if err != nil {
		t.Fatalf("read web/languages.json: %v", err)
	}

	if !bytes.Equal(got, want) {
		t.Fatal("web/languages.json is stale; run go generate ./internal/webmanifest/")
	}
}
