package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

// spGolden mirrors testdata/sp_golden.json: per shipped tokenizer, the
// output of Python sentencepiece (ids, piece strings and per-piece surfaces
// of the original input) for a fixed set of inputs.
type spGolden struct {
	Generator  string                       `json:"generator"`
	Tokenizers map[string]spGoldenTokenizer `json:"tokenizers"`
}

type spGoldenTokenizer struct {
	Model  string         `json:"model"`
	Sha256 string         `json:"sha256"`
	Cases  []spGoldenCase `json:"cases"`
}

type spGoldenCase struct {
	Text     string   `json:"text"`
	IDs      []int64  `json:"ids"`
	Pieces   []string `json:"pieces"`
	Surfaces []string `json:"surfaces"`
}

func loadSPGolden(t *testing.T) spGolden {
	t.Helper()

	data, err := os.ReadFile(filepath.Join("testdata", "sp_golden.json"))
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}

	var g spGolden

	err = json.Unmarshal(data, &g)
	if err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	if len(g.Tokenizers) == 0 {
		t.Fatal("golden file lists no tokenizers")
	}

	return g
}

// goldenModelTokenizers loads the model a golden entry was generated from
// through both constructors. It skips when the model is absent or is not the
// exact file the goldens were generated with.
func goldenModelTokenizers(t *testing.T, gt spGoldenTokenizer) map[string]Tokenizer {
	t.Helper()

	path, data := readGoldenModel(t, gt.Model, gt.Sha256)

	fromPath, err := NewSentencePieceTokenizer(path)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer(%s): %v", gt.Model, err)
	}

	fromBytes, err := NewSentencePieceTokenizerFromBytes(data)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizerFromBytes(%s): %v", gt.Model, err)
	}

	return map[string]Tokenizer{"path": fromPath, "bytes": fromBytes}
}

// comparableSurfaces maps our piece Texts onto sentencepiece's surface
// convention: the dummy-prefix '▁' of the first piece is dropped and every
// '▁' becomes a space.
func comparableSurfaces(pieces []Piece) []string {
	out := make([]string, len(pieces))
	for i, p := range pieces {
		text := p.Text
		if i == 0 {
			text = strings.TrimPrefix(text, string(spSep))
		}

		out[i] = strings.ReplaceAll(text, string(spSep), " ")
	}

	return out
}

func isBytePieceName(s string) bool {
	return len(s) == 6 && strings.HasPrefix(s, "<0x") && strings.HasSuffix(s, ">")
}

// TestEncode_MatchesSentencePieceGolden is the parity test against Python
// sentencepiece: identical ids, and piece surfaces that line up with
// sentencepiece's per-piece surfaces.
func TestEncode_MatchesSentencePieceGolden(t *testing.T) {
	g := loadSPGolden(t)

	for lang, gt := range g.Tokenizers {
		t.Run(lang, func(t *testing.T) {
			if len(gt.Cases) == 0 {
				t.Fatal("no golden cases")
			}

			for ctor, tok := range goldenModelTokenizers(t, gt) {
				for _, c := range gt.Cases {
					checkGoldenCase(t, ctor, tok, c)
				}
			}
		})
	}
}

func checkGoldenCase(t *testing.T, ctor string, tok Tokenizer, c spGoldenCase) {
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

	if got := pieceIDs(pieces); !slices.Equal(got, c.IDs) {
		t.Errorf("%s: EncodePieces(%q) ids\n got %v\nwant %v", ctor, c.Text, got, c.IDs)
	}

	if got := comparableSurfaces(pieces); !slices.Equal(got, mapSpaces(c.Surfaces)) {
		t.Errorf("%s: EncodePieces(%q) surfaces\n got %q\nwant %q", ctor, c.Text, got, mapSpaces(c.Surfaces))
	}

	// Pieces that are not byte-fallback pieces cover exactly their vocab
	// string.
	if len(pieces) == len(c.Pieces) {
		for i, name := range c.Pieces {
			if !isBytePieceName(name) && pieces[i].Text != name {
				t.Errorf("%s: EncodePieces(%q) piece %d text %q, vocab piece %q", ctor, c.Text, i, pieces[i].Text, name)
			}
		}
	}

	// Texts concatenate to the normalized input.
	if c.Text != "" && utf8.ValidString(c.Text) {
		want := string(spSep) + strings.ReplaceAll(c.Text, " ", string(spSep))
		if got := pieceSurface(pieces); got != want {
			t.Errorf("%s: surface(%q) = %q, want %q", ctor, c.Text, got, want)
		}
	}
}

func mapSpaces(surfaces []string) []string {
	out := make([]string, len(surfaces))
	for i, s := range surfaces {
		out[i] = strings.ReplaceAll(s, string(spSep), " ")
	}

	return out
}

// TestEncodePieces_ByteFallback uses the shipped English tokenizer; its byte
// pieces <0x00>..<0xFF> are ids 4..259.
func TestEncodePieces_ByteFallback(t *testing.T) {
	cases := map[string][]Piece{
		// € (E2 82 AC) is not in the vocab: three byte pieces, the last one
		// carrying the character.
		"€": {{ID: 260, Text: "▁"}, {ID: 230, Text: ""}, {ID: 134, Text: ""}, {ID: 176, Text: "€"}},
		// Each unknown character falls back on its own; runs are not merged.
		"a😀😀b": {
			{ID: 267, Text: "▁a"},
			{ID: 244, Text: ""},
			{ID: 163, Text: ""},
			{ID: 156, Text: ""},
			{ID: 132, Text: "😀"},
			{ID: 244, Text: ""},
			{ID: 163, Text: ""},
			{ID: 156, Text: ""},
			{ID: 132, Text: "😀"},
			{ID: 512, Text: "b"},
		},
		// Invalid UTF-8 becomes U+FFFD (EF BF BD) before encoding.
		"a\xffb": {{ID: 267, Text: "▁a"}, {ID: 243, Text: ""}, {ID: 195, Text: ""}, {ID: 193, Text: "�"}, {ID: 512, Text: "b"}},
		// Tab is not whitespace-escaped: a single byte piece.
		"a\tb": {{ID: 267, Text: "▁a"}, {ID: 13, Text: "\t"}, {ID: 512, Text: "b"}},
	}

	for name, tok := range allTokenizers(t) {
		for text, want := range cases {
			got, err := tok.EncodePieces(text)
			if err != nil {
				t.Fatalf("%s: EncodePieces(%q): %v", name, text, err)
			}

			if !slices.Equal(got, want) {
				t.Errorf("%s: EncodePieces(%q)\n got %v\nwant %v", name, text, got, want)
			}

			for _, p := range got {
				if !utf8.ValidString(p.Text) {
					t.Errorf("%s: EncodePieces(%q) piece %d has invalid UTF-8 text %q", name, text, p.ID, p.Text)
				}
			}
		}
	}
}
