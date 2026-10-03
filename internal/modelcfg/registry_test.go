package modelcfg

import (
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestLanguages(t *testing.T) {
	want := []string{"english_2026-01", "english_2026-09", "english_2026-09_24l", "english_drifting_26-09", "german", "german_24l"}
	if got := Languages(); !slices.Equal(got, want) {
		t.Errorf("Languages() = %v; want %v", got, want)
	}
}

func TestLookup_EmbeddedConfigsParse(t *testing.T) {
	for _, name := range Languages() {
		cfg, err := Lookup(name)
		if err != nil {
			t.Errorf("Lookup(%q): %v", name, err)

			continue
		}

		if cfg.VoicesRevision != "1e08e6a23401048648a9fdcfde2f89348215c2a7" {
			t.Errorf("Lookup(%q).VoicesRevision = %q", name, cfg.VoicesRevision)
		}
	}
}

// Upstream: get_default_voice_for_language matches by substring.
func TestLookup_DefaultVoice(t *testing.T) {
	for name, want := range map[string]string{
		"german":                 "juergen",
		"german_24l":             "juergen",
		"english_2026-01":        "alba",
		"english_2026-09":        "alba",
		"english_2026-09_24l":    "alba",
		"english_drifting_26-09": "alba",
	} {
		cfg, err := Lookup(name)
		if err != nil {
			t.Fatalf("Lookup(%q): %v", name, err)
		}

		if cfg.DefaultVoice != want {
			t.Errorf("Lookup(%q).DefaultVoice = %q; want %q", name, cfg.DefaultVoice, want)
		}
	}
}

func TestLookup_DriftingSamplerHead(t *testing.T) {
	cfg, err := Lookup("english_drifting_26-09")
	if err != nil {
		t.Fatal(err)
	}

	if cfg.FlowLM.Flow.Type != FlowTypeDrifting || cfg.NumTimeConds() != 0 {
		t.Errorf("Flow.Type = %q, NumTimeConds() = %d; want %q and 0", cfg.FlowLM.Flow.Type, cfg.NumTimeConds(), FlowTypeDrifting)
	}

	if cfg.FlowLM.LookupTable.Tokenizer != "sentencepiece" {
		t.Errorf("LookupTable.Tokenizer = %q; want sentencepiece (tokenizer.model until tokenizer.json loads)", cfg.FlowLM.LookupTable.Tokenizer)
	}
}

func TestLookup_24LayerVariant(t *testing.T) {
	cfg, err := Lookup("english_2026-09_24l")
	if err != nil {
		t.Fatal(err)
	}

	if cfg.FlowLM.Transformer.NumLayers != 24 {
		t.Errorf("NumLayers = %d; want 24", cfg.FlowLM.Transformer.NumLayers)
	}
}

// Until the Phase 5 tokenizer.json loader exists, the embedded configs point at
// the SentencePiece .model sibling.
func TestLookup_EmbeddedTokenizerIsSentencePiece(t *testing.T) {
	for _, name := range Languages() {
		cfg, err := Lookup(name)
		if err != nil {
			t.Fatal(err)
		}

		lt := cfg.FlowLM.LookupTable
		if lt.Tokenizer != TokenizerSentencePiece {
			t.Errorf("%s: Tokenizer = %q; want %q", name, lt.Tokenizer, TokenizerSentencePiece)
		}

		path, _, _ := strings.Cut(lt.TokenizerPath, "@")
		if !strings.HasSuffix(path, "/tokenizer.model") {
			t.Errorf("%s: TokenizerPath = %q; want a tokenizer.model path", name, lt.TokenizerPath)
		}
	}
}

func TestLookup_UnknownLanguage(t *testing.T) {
	_, err := Lookup("klingon")
	if err == nil || !strings.Contains(err.Error(), "klingon") || !strings.Contains(err.Error(), "german") {
		t.Fatalf("Lookup(klingon) error = %v; want an error naming it and the valid languages", err)
	}
}

func TestLookup_ReturnsFreshCopy(t *testing.T) {
	first, err := Lookup("german")
	if err != nil {
		t.Fatal(err)
	}

	first.ReplaceCharacters["x"] = "y"
	first.Mimi.SEANet.Ratios[0] = 99

	second, err := Lookup("german")
	if err != nil {
		t.Fatal(err)
	}

	if _, ok := second.ReplaceCharacters["x"]; ok || second.Mimi.SEANet.Ratios[0] == 99 {
		t.Error("mutating a Lookup result changed the next Lookup result")
	}
}

// Guards against transcription drift: the embedded config equals the verbatim
// upstream copy in testdata apart from the tokenizer rewrite.
func TestLookup_EmbeddedMatchesUpstreamCopy(t *testing.T) {
	got, err := Lookup("english_2026-01")
	if err != nil {
		t.Fatal(err)
	}

	want := loadTestdata(t, "english_2026-01")
	want.FlowLM.LookupTable.Tokenizer = TokenizerSentencePiece
	want.FlowLM.LookupTable.TokenizerPath = strings.Replace(want.FlowLM.LookupTable.TokenizerPath,
		"/tokenizer.json@", "/tokenizer.model@", 1)
	want.DefaultVoice = got.DefaultVoice
	want.VoicesRevision = got.VoicesRevision

	if !reflect.DeepEqual(got, want) {
		t.Errorf("embedded english_2026-01 differs from testdata:\n got %+v\nwant %+v", got, want)
	}
}

// Upstream: with a config or checkpoint instead of a language, the default
// voice is alba.
func TestLoadCustom_UsesFallbackVoice(t *testing.T) {
	cfg, err := LoadCustom("testdata/german.yaml")
	if err != nil {
		t.Fatal(err)
	}

	if cfg.DefaultVoice != "alba" || cfg.VoicesRevision != VoicesRevision {
		t.Errorf("DefaultVoice = %q, VoicesRevision = %q; want alba and %q",
			cfg.DefaultVoice, cfg.VoicesRevision, VoicesRevision)
	}

	if !cfg.FlowLM.InsertBOSBeforeVoice {
		t.Error("LoadCustom did not return the german config")
	}
}

func TestLoadCustom_MissingFile(t *testing.T) {
	_, err := LoadCustom("testdata/nope.yaml")
	if err == nil {
		t.Fatal("LoadCustom on a missing file succeeded; want an error")
	}
}
