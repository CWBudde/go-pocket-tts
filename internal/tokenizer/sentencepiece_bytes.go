//go:build !(js && wasm)

package tokenizer

import (
	"errors"
	"fmt"
	"os"
)

// NewSentencePieceTokenizerFromBytes loads a SentencePiece model from raw bytes.
// On native platforms it writes a temporary file and delegates to
// NewSentencePieceTokenizer, since the upstream library only exposes a file-path API.
func NewSentencePieceTokenizerFromBytes(data []byte) (Tokenizer, error) {
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
