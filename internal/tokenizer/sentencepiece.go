package tokenizer

import (
	"errors"
	"fmt"

	gosp "github.com/vikesh-raj/go-sentencepiece-encoder/sentencepiece"
)

// ErrEmptyPath is returned when NewSentencePieceTokenizer is called with an empty path.
var ErrEmptyPath = errors.New("tokenizer model path must not be empty")

// SentencePieceTokenizer implements Tokenizer using a pure-Go UNIGRAM SentencePiece model.
type SentencePieceTokenizer struct {
	proc gosp.Sentencepiece
}

// NewSentencePieceTokenizer loads a SentencePiece model from the given path.
func NewSentencePieceTokenizer(modelPath string) (*SentencePieceTokenizer, error) {
	if modelPath == "" {
		return nil, ErrEmptyPath
	}

	proc, err := gosp.NewSentencepieceFromFile(modelPath, false)
	if err != nil {
		return nil, fmt.Errorf("load sentencepiece model %q: %w", modelPath, err)
	}

	return &SentencePieceTokenizer{proc: proc}, nil
}

// Encode tokenizes text and returns SentencePiece token IDs as int64.
func (t *SentencePieceTokenizer) Encode(text string) ([]int64, error) {
	if text == "" {
		return []int64{}, nil
	}

	ids := t.proc.TokenizeToIDs(text)

	result := make([]int64, len(ids))
	for i, id := range ids {
		result[i] = int64(id)
	}

	return result, nil
}
