// Package tokenizer provides text tokenization for the PocketTTS engine.
// The primary implementation uses SentencePiece UNIGRAM tokenization matching
// the reference Python/Rust implementations exactly.
package tokenizer

// Piece is one token of an encoding and the text it covers.
type Piece struct {
	ID int64
	// Text is the source surface of the piece after normalization (NFKC,
	// whitespace folded to '▁'); '▁' (U+2581) marks a word start, as in the
	// vocab. The first piece carries the dummy-prefix '▁' unless the input
	// already starts with '▁'. Unknown pieces keep the characters they cover.
	Text string
}

// Tokenizer encodes text into SentencePiece token IDs.
type Tokenizer interface {
	// Encode tokenizes text and returns SentencePiece token IDs.
	Encode(text string) ([]int64, error)
	// EncodePieces tokenizes text and returns each token with its source
	// surface. The IDs are identical to Encode(text).
	EncodePieces(text string) ([]Piece, error)
}
