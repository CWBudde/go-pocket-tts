// Package tokenizer provides text tokenization for the PocketTTS engine.
// It implements UNIGRAM tokenization in pure Go from either a SentencePiece
// model (tokenizer.model, ids match Python sentencepiece) or a Hugging Face
// tokenizer.json (ids match Python tokenizers, upstream pocket-tts's default
// backend). Both formats share one trie and Viterbi; Load and LoadBytes pick
// the format from the file name or the bytes.
package tokenizer

// Piece is one token of an encoding and the text it covers.
type Piece struct {
	ID int64
	// Text is the source surface of the piece after normalization. The
	// shipped models use sentencepiece's identity normalizer: only ' ' is
	// replaced by '▁' (U+2581), which marks a word start as in the vocab;
	// every other character (tabs, newlines, NFKC-decomposable ones) is kept.
	// The first piece always carries the dummy-prefix '▁', even when the
	// input starts with '▁'. An unknown character is byte-fallback encoded as
	// one <0xXX> piece per UTF-8 byte: all but the last have an empty Text,
	// the last carries the whole character. So every Text is valid UTF-8 and
	// the Texts concatenate to "▁" + input with ' ' → '▁' (invalid UTF-8
	// bytes become U+FFFD).
	//
	// Tokenizers loaded from tokenizer.json follow Hugging Face tokenizers:
	// added tokens (<s>, </s>, ...) are cut out of the raw input first and
	// become one piece whose Text is the token itself (e.g. "<s>"); every
	// other segment gets its own leading '▁'. So the Texts concatenate to the
	// input with each non-token segment mapped to "▁" + segment with ' ' →
	// '▁'. A literal "<0xXX>" in the input matches that vocab piece and has
	// Text "<0x41>" etc., and a run of unknown characters is byte-encoded
	// as a whole but still attributes each character to its last byte piece.
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
