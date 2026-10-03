package tokenizer

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// NewJSONTokenizer loads a Hugging Face tokenizer.json (Unigram model) from
// the given path. It encodes like Python tokenizers does with that file.
func NewJSONTokenizer(path string) (*SentencePieceTokenizer, error) {
	if path == "" {
		return nil, ErrEmptyPath
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer.json %q: %w", path, err)
	}

	model, err := newSpModelFromJSON(data)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer.json %q: %w", path, err)
	}

	return &SentencePieceTokenizer{model: model}, nil
}

// ErrVocabSize is returned when a tokenizer's vocab size differs from the
// model config's n_bins (upstream asserts the same when it loads one).
var ErrVocabSize = errors.New("tokenizer vocab size does not match n_bins")

// checkVocabSize returns tok unless nBins > 0 and differs from its vocab size.
func checkVocabSize(tok Tokenizer, nBins int) (Tokenizer, error) {
	if nBins <= 0 {
		return tok, nil
	}

	sp, ok := tok.(*SentencePieceTokenizer)
	if !ok {
		return nil, fmt.Errorf("%w: %T has no vocab size", ErrVocabSize, tok)
	}

	if size := sp.VocabSize(); size != nBins {
		return nil, fmt.Errorf("%w: tokenizer has vocab size=%d but n_bins=%d", ErrVocabSize, size, nBins)
	}

	return tok, nil
}

// Load loads a tokenizer from path, picking the format by file extension:
// ".json" (any case) is a Hugging Face tokenizer.json, anything else a
// SentencePiece model (tokenizer.model). Like upstream, its vocab size must
// equal nBins (the model config's flow_lm.lookup_table.n_bins), else the
// error wraps ErrVocabSize; nBins <= 0 skips the check, for callers without a
// model config.
func Load(path string, nBins int) (Tokenizer, error) {
	if path == "" {
		return nil, ErrEmptyPath
	}

	var (
		tok *SentencePieceTokenizer
		err error
	)

	if strings.EqualFold(filepath.Ext(path), ".json") {
		tok, err = NewJSONTokenizer(path)
	} else {
		tok, err = NewSentencePieceTokenizer(path)
	}

	if err != nil {
		return nil, err
	}

	checked, err := checkVocabSize(tok, nBins)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer %q: %w", path, err)
	}

	return checked, nil
}

// LoadBytes loads a tokenizer from raw bytes, for callers without a file name
// (js/wasm), and checks its vocab size against nBins like Load does.
func LoadBytes(data []byte, nBins int) (Tokenizer, error) {
	tok, err := loadBytes(data)
	if err != nil {
		return nil, err
	}

	return checkVocabSize(tok, nBins)
}

// loadBytes picks the format: data whose first non-whitespace byte is '{' is
// a Hugging Face tokenizer.json, anything else a SentencePiece model. A
// SentencePiece model whose first piece message happens to be 123 bytes long
// also starts with "\n{", so such data that is not valid JSON is tried as a
// SentencePiece model before the JSON error is reported.
func loadBytes(data []byte) (Tokenizer, error) {
	trimmed := bytes.TrimLeft(data, " \t\r\n")
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return NewSentencePieceTokenizerFromBytes(data)
	}

	if !json.Valid(data) {
		tok, err := NewSentencePieceTokenizerFromBytes(data)
		if err == nil {
			return tok, nil
		}
	}

	return NewJSONTokenizerFromBytes(data)
}
