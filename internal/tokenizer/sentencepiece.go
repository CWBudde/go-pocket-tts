package tokenizer

import (
	"errors"
	"fmt"
	"os"
)

// ErrEmptyPath is returned when a tokenizer is loaded from an empty path.
var ErrEmptyPath = errors.New("tokenizer model path must not be empty")

// SentencePieceTokenizer implements Tokenizer using a pure-Go UNIGRAM model,
// loaded from a SentencePiece model or a Hugging Face tokenizer.json.
type SentencePieceTokenizer struct {
	model *spModel
}

// NewSentencePieceTokenizer loads a SentencePiece model from the given path.
func NewSentencePieceTokenizer(modelPath string) (*SentencePieceTokenizer, error) {
	if modelPath == "" {
		return nil, ErrEmptyPath
	}

	data, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load sentencepiece model %q: %w", modelPath, err)
	}

	model, err := newSpModel(data)
	if err != nil {
		return nil, fmt.Errorf("load sentencepiece model %q: %w", modelPath, err)
	}

	return &SentencePieceTokenizer{model: model}, nil
}

// Encode tokenizes text and returns SentencePiece token IDs as int64.
// It is derived from EncodePieces, so both always agree.
func (t *SentencePieceTokenizer) Encode(text string) ([]int64, error) {
	pieces, err := t.EncodePieces(text)
	if err != nil {
		return nil, err
	}

	ids := make([]int64, len(pieces))
	for i, p := range pieces {
		ids[i] = p.ID
	}

	return ids, nil
}

// EncodePieces tokenizes text and returns each token with its source surface.
func (t *SentencePieceTokenizer) EncodePieces(text string) ([]Piece, error) {
	if text == "" {
		return []Piece{}, nil
	}

	return t.model.pieces(text), nil
}
