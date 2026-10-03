package tokenizer

import (
	"os"
	"slices"
	"strings"
	"testing"

	gosp "github.com/vikesh-raj/go-sentencepiece-encoder/sentencepiece"
)

// piecesCorpus covers the inputs the sentence splitter cares about: plain
// text, surrounding/inner whitespace, sentence and clause punctuation,
// decimals, characters outside the vocab (single and consecutive), and NFKC
// normalisation.
var piecesCorpus = []string{
	"Hello world.",
	"  Hello  world.  ",
	"        hello",
	"Hi! Really... ok? Fine.",
	"a,b; c: d",
	"The temperature is 98.6°F today.",
	"Pi is 3.14, roughly.",
	"a😀b",
	"a😀😀b",
	"😀🎉",
	"x\ty\nz",
	"ﬁne",
	"▁x",
	"Dr. Smith's café—opened in 1999—serves crème brûlée; it's \"world-famous\" (or so they say)!",
	"",
}

// allTokenizers returns the path-loaded and the bytes-loaded tokenizer so
// every test exercises both constructors.
func allTokenizers(t *testing.T) map[string]Tokenizer {
	t.Helper()

	path := modelPath(t)

	fromPath, err := NewSentencePieceTokenizer(path)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read model: %v", err)
	}

	fromBytes, err := NewSentencePieceTokenizerFromBytes(data)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizerFromBytes: %v", err)
	}

	return map[string]Tokenizer{"path": fromPath, "bytes": fromBytes}
}

func pieceIDs(pieces []Piece) []int64 {
	ids := make([]int64, len(pieces))
	for i, p := range pieces {
		ids[i] = p.ID
	}

	return ids
}

func pieceSurface(pieces []Piece) string {
	var b strings.Builder
	for _, p := range pieces {
		b.WriteString(p.Text)
	}

	return b.String()
}

func TestEncodePieces_IDsMatchEncode(t *testing.T) {
	for name, tok := range allTokenizers(t) {
		for _, text := range piecesCorpus {
			ids, err := tok.Encode(text)
			if err != nil {
				t.Fatalf("%s: Encode(%q): %v", name, text, err)
			}

			pieces, err := tok.EncodePieces(text)
			if err != nil {
				t.Fatalf("%s: EncodePieces(%q): %v", name, text, err)
			}

			if got := pieceIDs(pieces); !equalInt64(got, ids) {
				t.Errorf("%s: EncodePieces(%q) IDs = %v, Encode = %v", name, text, got, ids)
			}
		}
	}
}

// TestEncodePieces_SurfaceConcat pins the piece surface convention observed
// from the reference library: the normalised (NFKC, whitespace-folded) input
// with every whitespace rune replaced by '▁' and a dummy-prefix '▁' prepended
// unless the input already starts with '▁'. Leading/trailing whitespace is
// kept, not stripped.
func TestEncodePieces_SurfaceConcat(t *testing.T) {
	cases := map[string]string{
		"Hello world.":                     "▁Hello▁world.",
		"  Hello  world.  ":                "▁▁▁Hello▁▁world.▁▁",
		"        hello":                    "▁▁▁▁▁▁▁▁▁hello",
		"Hi! Really... ok? Fine.":          "▁Hi!▁Really...▁ok?▁Fine.",
		"a,b; c: d":                        "▁a,b;▁c:▁d",
		"The temperature is 98.6°F today.": "▁The▁temperature▁is▁98.6°F▁today.",
		"Pi is 3.14, roughly.":             "▁Pi▁is▁3.14,▁roughly.",
		"a😀b":                              "▁a😀b",
		"a😀😀b":                             "▁a😀😀b",
		"😀🎉":                               "▁😀🎉",
		"x\ty\nz":                          "▁x▁y▁z",
		"ﬁne":                              "▁fine",
		"▁x":                               "▁x",
	}

	for name, tok := range allTokenizers(t) {
		for text, want := range cases {
			pieces, err := tok.EncodePieces(text)
			if err != nil {
				t.Fatalf("%s: EncodePieces(%q): %v", name, text, err)
			}

			if got := pieceSurface(pieces); got != want {
				t.Errorf("%s: surface(%q) = %q, want %q", name, text, got, want)
			}
		}
	}
}

func TestEncodePieces_HelloWorld(t *testing.T) {
	want := []Piece{{ID: 2994, Text: "▁Hello"}, {ID: 578, Text: "▁world"}, {ID: 263, Text: "."}}

	for name, tok := range allTokenizers(t) {
		got, err := tok.EncodePieces("Hello world.")
		if err != nil {
			t.Fatalf("%s: EncodePieces: %v", name, err)
		}

		if !slices.Equal(got, want) {
			t.Errorf("%s: EncodePieces(%q) = %v, want %v", name, "Hello world.", got, want)
		}
	}
}

func TestEncodePieces_UnknownKeepsSurface(t *testing.T) {
	cases := map[string][]Piece{
		// single unknown character
		"a😀b": {{ID: 267, Text: "▁a"}, {ID: 0, Text: "😀"}, {ID: 512, Text: "b"}},
		// consecutive unknowns merge into one unk ID but keep the whole run
		"a😀😀b": {{ID: 267, Text: "▁a"}, {ID: 0, Text: "😀😀"}, {ID: 512, Text: "b"}},
		"😀🎉":   {{ID: 260, Text: "▁"}, {ID: 0, Text: "😀🎉"}},
		"6°F":  {{ID: 260, Text: "▁"}, {ID: 543, Text: "6"}, {ID: 0, Text: "°"}, {ID: 1217, Text: "F"}},
	}

	for name, tok := range allTokenizers(t) {
		for text, want := range cases {
			got, err := tok.EncodePieces(text)
			if err != nil {
				t.Fatalf("%s: EncodePieces(%q): %v", name, text, err)
			}

			if !slices.Equal(got, want) {
				t.Errorf("%s: EncodePieces(%q) = %v, want %v", name, text, got, want)
			}
		}
	}
}

func TestEncodePieces_Empty(t *testing.T) {
	for name, tok := range allTokenizers(t) {
		got, err := tok.EncodePieces("")
		if err != nil {
			t.Fatalf("%s: EncodePieces(\"\") should not error: %v", name, err)
		}

		if got == nil || len(got) != 0 {
			t.Errorf("%s: EncodePieces(\"\") = %#v, want empty non-nil slice", name, got)
		}
	}
}

// TestEncodePieces_MatchesReferenceLibrary cross-checks the in-package
// trie/Viterbi port against github.com/vikesh-raj/go-sentencepiece-encoder.
// IDs must match for every input. Piece texts must match too, except where
// the library drops the surface of consecutive unknown characters (its
// sliceToTokens extends a copy of the previous token, so only the first
// unknown rune of a run survives).
func TestEncodePieces_MatchesReferenceLibrary(t *testing.T) {
	path := modelPath(t)

	ref, err := gosp.NewSentencepieceFromFile(path, false)
	if err != nil {
		t.Fatalf("load reference library: %v", err)
	}

	consecutiveUnknown := map[string]bool{"a😀😀b": true, "😀🎉": true}

	for name, tok := range allTokenizers(t) {
		for _, text := range piecesCorpus {
			if text == "" {
				continue
			}

			pieces, err := tok.EncodePieces(text)
			if err != nil {
				t.Fatalf("%s: EncodePieces(%q): %v", name, text, err)
			}

			refTokens := ref.Tokenize(text)
			if len(refTokens) != len(pieces) {
				t.Errorf("%s: %q: %d pieces, reference has %d", name, text, len(pieces), len(refTokens))

				continue
			}

			for i, rt := range refTokens {
				if int64(rt.ID) != pieces[i].ID {
					t.Errorf("%s: %q piece %d: ID %d, reference %d", name, text, i, pieces[i].ID, rt.ID)
				}

				if !consecutiveUnknown[text] && rt.Text != pieces[i].Text {
					t.Errorf("%s: %q piece %d: text %q, reference %q", name, text, i, pieces[i].Text, rt.Text)
				}
			}
		}
	}
}
