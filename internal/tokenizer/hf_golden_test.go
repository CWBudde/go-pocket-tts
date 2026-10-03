package tokenizer

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

// hfGolden mirrors testdata/hf_golden.json: per shipped tokenizer.json, the
// output of Python Hugging Face tokenizers (ids and token strings) for a fixed
// set of inputs. HF offsets are not usable as surfaces, so there are none.
type hfGolden struct {
	Generator  string                       `json:"generator"`
	Tokenizers map[string]hfGoldenTokenizer `json:"tokenizers"`
}

type hfGoldenTokenizer struct {
	Model  string         `json:"model"`
	Sha256 string         `json:"sha256"`
	Cases  []hfGoldenCase `json:"cases"`
}

type hfGoldenCase struct {
	Text   string   `json:"text"`
	IDs    []int64  `json:"ids"`
	Tokens []string `json:"tokens"`
}

func loadHFGolden(t *testing.T) hfGolden {
	t.Helper()

	data, err := os.ReadFile(filepath.Join("testdata", "hf_golden.json"))
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}

	var g hfGolden

	err = json.Unmarshal(data, &g)
	if err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	if len(g.Tokenizers) == 0 {
		t.Fatal("golden file lists no tokenizers")
	}

	return g
}

// readGoldenModel reads a model file named relative to the repo root (tests
// run in internal/tokenizer). It skips when the file is absent or is not the
// exact file the goldens were generated from.
func readGoldenModel(t *testing.T, model, wantSha256 string) (string, []byte) {
	t.Helper()

	path := filepath.Join("..", "..", filepath.FromSlash(model))

	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("%s not found; skipping golden comparison", model)
	}

	if err != nil {
		t.Fatalf("read %s: %v", model, err)
	}

	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != wantSha256 {
		t.Skipf("%s has sha256 %s, goldens were generated from %s", model, got, wantSha256)
	}

	return path, data
}

// hfGoldenTokenizers loads a tokenizer.json through every JSON entry point.
func hfGoldenTokenizers(t *testing.T, gt hfGoldenTokenizer) map[string]Tokenizer {
	t.Helper()

	path, data := readGoldenModel(t, gt.Model, gt.Sha256)

	fromPath, err := NewJSONTokenizer(path)
	if err != nil {
		t.Fatalf("NewJSONTokenizer(%s): %v", gt.Model, err)
	}

	fromBytes, err := NewJSONTokenizerFromBytes(data)
	if err != nil {
		t.Fatalf("NewJSONTokenizerFromBytes(%s): %v", gt.Model, err)
	}

	loaded, err := Load(path, 0)
	if err != nil {
		t.Fatalf("Load(%s): %v", gt.Model, err)
	}

	loadedBytes, err := LoadBytes(data, 0)
	if err != nil {
		t.Fatalf("LoadBytes(%s): %v", gt.Model, err)
	}

	return map[string]Tokenizer{
		"NewJSONTokenizer":          fromPath,
		"NewJSONTokenizerFromBytes": fromBytes,
		"Load":                      loaded,
		"LoadBytes":                 loadedBytes,
	}
}

// TestEncode_MatchesHFGolden is the parity test against Python Hugging Face
// tokenizers (upstream pocket-tts's default backend) for tokenizer.json files.
func TestEncode_MatchesHFGolden(t *testing.T) {
	g := loadHFGolden(t)

	for lang, gt := range g.Tokenizers {
		t.Run(lang, func(t *testing.T) {
			if len(gt.Cases) == 0 {
				t.Fatal("no golden cases")
			}

			for ctor, tok := range hfGoldenTokenizers(t, gt) {
				for _, c := range gt.Cases {
					checkHFGoldenCase(t, ctor, tok, c)
				}
			}
		})
	}
}

func checkHFGoldenCase(t *testing.T, ctor string, tok Tokenizer, c hfGoldenCase) {
	t.Helper()

	ids, err := tok.Encode(c.Text)
	if err != nil {
		t.Fatalf("%s: Encode(%q): %v", ctor, c.Text, err)
	}

	if !slices.Equal(ids, c.IDs) {
		t.Errorf("%s: Encode(%q)\n got %v\nwant %v", ctor, c.Text, ids, c.IDs)
	}

	pieces, err := tok.EncodePieces(c.Text)
	if err != nil {
		t.Fatalf("%s: EncodePieces(%q): %v", ctor, c.Text, err)
	}

	if got := pieceIDs(pieces); !slices.Equal(got, ids) {
		t.Errorf("%s: EncodePieces(%q) ids %v, Encode ids %v", ctor, c.Text, got, ids)
	}

	for i, p := range pieces {
		if !utf8.ValidString(p.Text) {
			t.Errorf("%s: EncodePieces(%q) piece %d has invalid UTF-8 text %q", ctor, c.Text, i, p.Text)
		}
	}

	// Pieces that are not byte-fallback pieces cover exactly their token
	// string (special tokens included).
	if len(pieces) == len(c.Tokens) {
		for i, name := range c.Tokens {
			if !isBytePieceName(name) && pieces[i].Text != name {
				t.Errorf("%s: EncodePieces(%q) piece %d text %q, token %q", ctor, c.Text, i, pieces[i].Text, name)
			}
		}
	}
}

// tokenizerPairs lists tokenizer.model/tokenizer.json pairs to cross-check:
// the ones the sp goldens name (models/...), plus every <dir>/<lang>/ pair
// under $POCKETTTS_TOKENIZER_PAIRS_DIR when set.
func tokenizerPairs(t *testing.T, g spGolden) map[string][2]string {
	t.Helper()

	pairs := make(map[string][2]string)

	for lang, gt := range g.Tokenizers {
		proto := filepath.Join("..", "..", filepath.FromSlash(gt.Model))
		pairs[lang] = [2]string{proto, strings.TrimSuffix(proto, filepath.Ext(proto)) + ".json"}
	}

	if dir := os.Getenv("POCKETTTS_TOKENIZER_PAIRS_DIR"); dir != "" {
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("read $POCKETTTS_TOKENIZER_PAIRS_DIR: %v", err)
		}

		for _, e := range entries {
			if e.IsDir() {
				sub := filepath.Join(dir, e.Name())
				pairs["pairs/"+e.Name()] = [2]string{
					filepath.Join(sub, "tokenizer.model"),
					filepath.Join(sub, "tokenizer.json"),
				}
			}
		}
	}

	return pairs
}

// TestJSONMatchesSentencePiece checks that a tokenizer.json and the
// tokenizer.model it was converted from produce the same ids on ordinary text
// (the two only differ on literal special-token and <0xXX> strings).
func TestJSONMatchesSentencePiece(t *testing.T) {
	g := loadSPGolden(t)

	var texts []string

	for _, gt := range g.Tokenizers {
		for _, c := range gt.Cases {
			texts = append(texts, c.Text)
		}
	}

	texts = append(texts, piecesCorpus...)

	for name, pair := range tokenizerPairs(t, g) {
		t.Run(name, func(t *testing.T) {
			for _, p := range pair {
				_, err := os.Stat(p)
				if err != nil {
					t.Skipf("%s not available: %v", p, err)
				}
			}

			proto, err := Load(pair[0], 0)
			if err != nil {
				t.Fatalf("Load(%s): %v", pair[0], err)
			}

			hf, err := Load(pair[1], 0)
			if err != nil {
				t.Fatalf("Load(%s): %v", pair[1], err)
			}

			for _, text := range texts {
				want, err := proto.Encode(text)
				if err != nil {
					t.Fatalf("proto Encode(%q): %v", text, err)
				}

				got, err := hf.Encode(text)
				if err != nil {
					t.Fatalf("json Encode(%q): %v", text, err)
				}

				if !slices.Equal(got, want) {
					t.Errorf("Encode(%q)\njson  %v\nproto %v", text, got, want)
				}
			}
		})
	}
}
