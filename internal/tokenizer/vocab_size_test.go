package tokenizer

import (
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
)

type vocabSizer interface{ VocabSize() int }

func vocabSizeOf(t *testing.T, tok Tokenizer) int {
	t.Helper()

	s, ok := tok.(vocabSizer)
	if !ok {
		t.Fatalf("%T has no VocabSize method", tok)
	}

	return s.VocabSize()
}

// TestLoad_VocabSize checks the upstream n_bins assertion (JsonTokenizer /
// SentencePieceTokenizer: nbins == vocab size) for both formats and both
// loaders.
func TestLoad_VocabSize(t *testing.T) {
	// A duplicate piece counts: tokenizers' get_vocab_size() is the length of
	// model.vocab, not the number of distinct pieces (checked with 0.23.2).
	dup := jsonBase()
	jsonModel(dup)["vocab"] = append(jsonModel(dup)["vocab"].([]any), []any{"a", -5.0})

	models := map[string]struct {
		file string
		data []byte
		size int
	}{
		"proto":          {"tok.model", testModel{pieces: basePieces()}.encode(), len(basePieces())},
		"proto+bytes":    {"tok.model", testModel{pieces: append(basePieces(), bytePieces()...), byteFallback: true}.encode(), len(basePieces()) + 256},
		"json":           {"tok.json", marshalJSONModel(t, jsonBase()), jsonBaseVocabSize},
		"json+bytes":     {"tok.json", marshalJSONModel(t, jsonWithBytes()), jsonBaseVocabSize + 256},
		"json+duplicate": {"tok.json", marshalJSONModel(t, dup), jsonBaseVocabSize + 1},
	}

	for name, m := range models {
		path := filepath.Join(t.TempDir(), m.file)
		writeTestFile(t, path, m.data)

		loaders := map[string]func(nBins int) (Tokenizer, error){
			"Load":      func(n int) (Tokenizer, error) { return Load(path, n) },
			"LoadBytes": func(n int) (Tokenizer, error) { return LoadBytes(m.data, n) },
		}

		for loader, load := range loaders {
			t.Run(name+"/"+loader, func(t *testing.T) {
				for _, nBins := range []int{m.size, 0} {
					tok, err := load(nBins)
					if err != nil {
						t.Fatalf("n_bins=%d: %v", nBins, err)
					}

					if got := vocabSizeOf(t, tok); got != m.size {
						t.Errorf("VocabSize() = %d, want %d", got, m.size)
					}
				}

				for _, nBins := range []int{m.size - 1, m.size + 1} {
					_, err := load(nBins)
					if !errors.Is(err, ErrVocabSize) {
						t.Fatalf("n_bins=%d: got %v, want ErrVocabSize", nBins, err)
					}

					for _, n := range []int{m.size, nBins} {
						if !strings.Contains(err.Error(), strconv.Itoa(n)) {
							t.Errorf("error %q does not mention %d", err, n)
						}
					}
				}
			})
		}
	}
}

// TestVocabSize_RealModels checks that the local tokenizers in both formats
// pass the n_bins check against their embedded model configs.
func TestVocabSize_RealModels(t *testing.T) {
	for _, c := range []struct{ language, dir string }{
		{"english_2026-01", "models"},
		{"german", "models/german"},
	} {
		cfg, err := modelcfg.Lookup(c.language)
		if err != nil {
			t.Fatalf("modelcfg.Lookup(%s): %v", c.language, err)
		}

		nBins := cfg.FlowLM.LookupTable.NBins

		for _, file := range []string{"tokenizer.model", "tokenizer.json"} {
			t.Run(c.language+"/"+file, func(t *testing.T) {
				path := filepath.Join("..", "..", filepath.FromSlash(c.dir), file)

				_, err := os.Stat(path)
				if err != nil {
					t.Skipf("%s not available: %v", path, err)
				}

				tok, err := Load(path, nBins)
				if err != nil {
					t.Fatalf("Load(%s, %d): %v", path, nBins, err)
				}

				if got := vocabSizeOf(t, tok); got != nBins {
					t.Errorf("VocabSize() = %d, want n_bins %d", got, nBins)
				}
			})
		}
	}
}
