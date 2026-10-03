package tokenizer

import (
	"bytes"
	"encoding/json"
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

// Load loads a tokenizer from path, picking the format by file extension:
// ".json" (any case) is a Hugging Face tokenizer.json, anything else a
// SentencePiece model (tokenizer.model).
func Load(path string) (Tokenizer, error) {
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

	return tok, nil
}

// LoadBytes loads a tokenizer from raw bytes, for callers without a file name
// (js/wasm): data whose first non-whitespace byte is '{' is a Hugging Face
// tokenizer.json, anything else a SentencePiece model. A SentencePiece model
// whose first piece message happens to be 123 bytes long also starts with
// "\n{", so such data that is not valid JSON is tried as a SentencePiece model
// before the JSON error is reported.
func LoadBytes(data []byte) (Tokenizer, error) {
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
