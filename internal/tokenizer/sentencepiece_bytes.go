package tokenizer

// NewSentencePieceTokenizerFromBytes loads a SentencePiece model from raw
// bytes without touching the filesystem, so it also works in js/wasm builds.
func NewSentencePieceTokenizerFromBytes(data []byte) (Tokenizer, error) {
	model, err := newSpModel(data)
	if err != nil {
		return nil, err
	}

	return &SentencePieceTokenizer{model: model}, nil
}
