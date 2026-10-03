package text

import (
	"errors"
	"fmt"
	"math"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/cwbudde/go-pocket-tts/internal/tokenizer"
)

// Tokenizer is the minimal interface required by PrepareChunks.
// It is satisfied by tokenizer.Tokenizer from the tokenizer package.
type Tokenizer interface {
	Encode(text string) ([]int64, error)
	// EncodePieces returns the tokens of text with their surfaces; the IDs
	// are identical to Encode(text). The splitter decodes token spans from it.
	EncodePieces(text string) ([]tokenizer.Piece, error)
}

// ChunkMetadata holds a preprocessed text chunk and its generation parameters.
type ChunkMetadata struct {
	Text      string  // preprocessed chunk text
	TokenIDs  []int64 // SentencePiece token IDs
	NumTokens int     // len(TokenIDs)
	NumWords  int     // word count (for FramesAfterEOS)
}

const DefaultMimiFrameRate = 12.5

// MaxFrames returns the maximum number of latent frames for this chunk.
// Formula: ceil((num_tokens / 3 + 2) × 12.5)
// Matches the reference: ceil(gen_len_sec × frame_rate)
// where gen_len_sec = token_count / 3.0 + 2.0 and frame_rate = 12.5.
func (c ChunkMetadata) MaxFrames() float64 {
	return float64(EstimateMaxFrames(c.NumTokens, DefaultMimiFrameRate))
}

// EstimateMaxFrames mirrors upstream TTSModel._estimate_max_gen_len:
// ceil((token_count / 3.0 + 2.0) * mimi.frame_rate).
func EstimateMaxFrames(tokenCount int, frameRate float64) int {
	if tokenCount < 0 {
		tokenCount = 0
	}

	if frameRate <= 0 || math.IsNaN(frameRate) || math.IsInf(frameRate, 0) {
		frameRate = DefaultMimiFrameRate
	}

	return int(math.Ceil((float64(tokenCount)/3.0 + 2.0) * frameRate))
}

// FramesAfterEOS returns the number of extra frames to generate after EOS is
// detected. The base value is 3 for ≤4-word chunks or 1 otherwise, plus 2
// additional frames matching the reference implementation.
func (c ChunkMetadata) FramesAfterEOS() int {
	if c.NumWords <= 4 {
		return 5
	}

	return 3
}

// Upstream text_chunking.py: _TERMINAL_PUNCTUATION, _WEAK_PUNCTUATION, _CLOSERS.
const (
	terminalPunctuation = ".!?…"
	weakPunctuation     = ",;:-–—"
	closers             = "\"'”’)]»"
)

// strayPunctuation matches a sentence mark followed by a comma, semicolon or
// colon, which deleting quotes leaves behind ('"Hi?", she said').
var strayPunctuation = regexp.MustCompile(`([.!?…])\s*[,;:]`)

// PrepareText ports upstream prepare_text_prompt:
//  1. Trim; apply opts.ReplaceCharacters, collapse whitespace and drop a
//     comma/semicolon/colon after a sentence mark. Empty → ErrEmptyText.
//  2. Normalize newlines → spaces, collapse repeated spaces.
//  3. opts.RemoveSemicolons: ';' → ','.
//  4. Count words (returned; it drives the frames_after_eos guess).
//  5. opts.CapitalizeFirst: upper-case the first character.
//  6. opts.AppendTerminalPunctuation: see ensureTerminalPunctuation.
//  7. opts.PadShortInputs: pad with 8 leading spaces when < 5 words.
func PrepareText(input string, opts Options) (string, int, error) {
	s := strings.TrimSpace(input)

	if len(opts.ReplaceCharacters) > 0 {
		s = strings.Join(strings.Fields(replaceCharacters(s, opts.ReplaceCharacters)), " ")
		s = strayPunctuation.ReplaceAllString(s, "$1")
	}

	if s == "" {
		return "", 0, ErrEmptyText
	}

	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	// Go collapses every run of spaces; upstream does a single "  " → " " pass.
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}

	if opts.RemoveSemicolons {
		s = strings.ReplaceAll(s, ";", ",")
	}

	words := len(splitWords(s))

	if opts.CapitalizeFirst {
		r, size := utf8.DecodeRuneInString(s)
		if r != utf8.RuneError {
			s = string(unicode.ToUpper(r)) + s[size:]
		}
	}

	if opts.AppendTerminalPunctuation {
		s = ensureTerminalPunctuation(s)
	}

	if opts.PadShortInputs && len(splitWords(s)) < 5 {
		s = "        " + s
	}

	return s, words, nil
}

// replaceCharacters is Python's str.translate with a one-character key table.
func replaceCharacters(s string, table map[rune]string) string {
	var sb strings.Builder

	sb.Grow(len(s))

	for _, r := range s {
		if to, ok := table[r]; ok {
			sb.WriteString(to)
		} else {
			sb.WriteRune(r)
		}
	}

	return sb.String()
}

// ensureTerminalPunctuation ports upstream _ensure_terminal_punctuation. Text
// that ends with sentence-final punctuation, possibly followed by closing
// quotes or brackets, is left alone. A trailing comma, colon or dash is
// replaced by a period placed before the closers. Anything else gets a period
// appended after them.
func ensureTerminalPunctuation(s string) string {
	core := strings.TrimRight(s, closers+" ")
	trailing := strings.TrimSpace(s[len(core):])

	if core == "" {
		return s
	}

	last, _ := utf8.DecodeLastRuneInString(core)

	switch {
	case strings.ContainsRune(terminalPunctuation, last):
		return s
	case strings.ContainsRune(weakPunctuation, last):
		return strings.TrimRight(core, weakPunctuation+" ") + "." + trailing
	default:
		return s + "."
	}
}

// PrepareChunks splits text into chunks of at most maxTokens tokens like
// upstream split_into_best_sentences (see splitIntoBestSentences) and, like
// upstream generate_audio_stream, prepares and tokenizes every chunk again.
// Each returned ChunkMetadata holds the prepared chunk text, its token IDs and
// its word count.
func PrepareChunks(input string, tok Tokenizer, maxTokens int, opts Options) ([]ChunkMetadata, error) {
	if strings.TrimSpace(input) == "" {
		return nil, errors.New("input text is empty")
	}

	texts, err := splitIntoBestSentences(tok, input, maxTokens, opts)
	if err != nil {
		return nil, err
	}

	chunks := make([]ChunkMetadata, 0, len(texts))

	for _, chunk := range texts {
		prepared, ids, words, err := encodePrepared(tok, chunk, opts)
		if err != nil {
			return nil, err
		}

		chunks = append(chunks, ChunkMetadata{
			Text:      prepared,
			TokenIDs:  ids,
			NumTokens: len(ids),
			NumWords:  words,
		})
	}

	return chunks, nil
}

// encodePrepared runs PrepareText on s and tokenizes the result.
func encodePrepared(tok Tokenizer, s string, opts Options) (string, []int64, int, error) {
	prepared, words, err := PrepareText(s, opts)
	if err != nil {
		return "", nil, 0, err
	}

	ids, err := tok.Encode(prepared)
	if err != nil {
		return "", nil, 0, fmt.Errorf("encode %q: %w", prepared, err)
	}

	return prepared, ids, words, nil
}

// splitWords splits text into non-empty word tokens on whitespace boundaries.
func splitWords(s string) []string {
	return strings.FieldsFunc(s, unicode.IsSpace)
}
