package tokenizer

// NewSentencePieceTokenizerFromBytes loads a SentencePiece model from raw
// bytes without touching the filesystem, so it also works in js/wasm builds.
func NewSentencePieceTokenizerFromBytes(data []byte) (Tokenizer, error) {
	trie, err := newSpTrie(data)
	if err != nil {
		return nil, err
	}

	return &SentencePieceTokenizer{trie: trie}, nil
}
