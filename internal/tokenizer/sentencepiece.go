package tokenizer

import (
	"errors"
	"fmt"
	"os"

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

// NewSentencePieceTokenizerFromBytes loads a SentencePiece model from raw bytes.
// It writes the data to a temporary file and delegates to NewSentencePieceTokenizer,
// which is necessary because the upstream library only exposes a file-path API.
// This is the primary entry point for WASM builds where there is no real filesystem.
func NewSentencePieceTokenizerFromBytes(data []byte) (*SentencePieceTokenizer, error) {
	if len(data) == 0 {
		return nil, errors.New("tokenizer model data must not be empty")
	}

	f, err := os.CreateTemp("", "sp-*.model")
	if err != nil {
		return nil, fmt.Errorf("create temp sentencepiece file: %w", err)
	}

	defer func() { _ = os.Remove(f.Name()) }() // best-effort temp file cleanup

	_, err = f.Write(data)
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("write tokenizer model bytes: %w", err)
	}

	path := f.Name()

	err = f.Close()
	if err != nil {
		return nil, fmt.Errorf("close tokenizer temp file: %w", err)
	}

	return NewSentencePieceTokenizer(path)
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
