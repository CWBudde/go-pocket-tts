// Package texttest provides a model-free tokenizer for tests of the text
// preparation pipeline.
package texttest

import (
	"hash/fnv"
	"strings"
	"unicode"

	"github.com/cwbudde/go-pocket-tts/internal/tokenizer"
)

// wordStart is the SentencePiece meta symbol that marks a word start.
const wordStart = "▁"

// Tokenizer splits text the way SentencePiece splits English prose, without a
// model: a dummy-prefix '▁' marks every word start, a run of letters and
// digits is one piece, and every other rune is a piece of its own. A word
// that starts with such a rune gets a standalone '▁' piece, so ".!...?"
// encodes as "▁", ".", "!", ".", ".", ".", "?". Ids are derived from the
// piece surface, so equal pieces always get equal ids.
type Tokenizer struct{}

// Encode returns the ids of EncodePieces(text).
func (Tokenizer) Encode(text string) ([]int64, error) {
	pieces, err := Tokenizer{}.EncodePieces(text)
	if err != nil {
		return nil, err
	}

	ids := make([]int64, len(pieces))
	for i, p := range pieces {
		ids[i] = p.ID
	}

	return ids, nil
}

// EncodePieces splits text into SentencePiece-like pieces.
func (Tokenizer) EncodePieces(text string) ([]tokenizer.Piece, error) {
	if text == "" {
		return nil, nil
	}

	var (
		pieces  []tokenizer.Piece
		run     strings.Builder
		pending bool // a '▁' not yet attached to a piece
	)

	emit := func(s string) {
		pieces = append(pieces, tokenizer.Piece{ID: PieceID(s), Text: s})
	}

	flushRun := func() {
		if run.Len() > 0 {
			emit(run.String())
			run.Reset()
		}
	}

	for _, r := range wordStart + strings.ReplaceAll(text, " ", wordStart) {
		switch {
		case string(r) == wordStart:
			flushRun()

			if pending {
				emit(wordStart)
			}

			pending = true
		case unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsMark(r):
			if pending {
				run.WriteString(wordStart)

				pending = false
			}

			run.WriteRune(r)
		default:
			flushRun()

			if pending {
				emit(wordStart)

				pending = false
			}

			emit(string(r))
		}
	}

	flushRun()

	if pending {
		emit(wordStart)
	}

	return pieces, nil
}

// PieceID is the id Tokenizer assigns to a piece surface.
func PieceID(piece string) int64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(piece))

	return int64(h.Sum64() >> 1)
}
