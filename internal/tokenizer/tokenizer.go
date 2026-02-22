// Package tokenizer provides text tokenization for the PocketTTS engine.
// The primary implementation uses SentencePiece BPE tokenization matching
// the reference Python/Rust implementations exactly.
package tokenizer

// Tokenizer encodes text into SentencePiece token IDs.
type Tokenizer interface {
	// Encode tokenizes text and returns SentencePiece token IDs.
	Encode(text string) ([]int64, error)
}
