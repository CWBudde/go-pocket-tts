package text

import (
	"fmt"
	"math"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Tokenizer is the minimal interface required by PrepareChunks.
// It is satisfied by tokenizer.Tokenizer from the tokenizer package.
type Tokenizer interface {
	Encode(text string) ([]int64, error)
}

// ChunkMetadata holds a preprocessed text chunk and its generation parameters.
type ChunkMetadata struct {
	Text      string  // preprocessed chunk text
	TokenIDs  []int64 // SentencePiece token IDs
	NumTokens int     // len(TokenIDs)
	NumWords  int     // word count (for FramesAfterEOS)
}

// MaxFrames returns the maximum number of latent frames for this chunk.
// Formula: ceil(num_tokens/3 + 2) × 12.5
func (c ChunkMetadata) MaxFrames() float64 {
	return math.Ceil(float64(c.NumTokens)/3.0+2.0) * 12.5
}

// FramesAfterEOS returns the number of extra frames to generate after EOS is
// detected: 3 for ≤4-word chunks, 1 otherwise.
func (c ChunkMetadata) FramesAfterEOS() int {
	if c.NumWords <= 4 {
		return 3
	}
	return 1
}

// PrepareText applies the reference text preprocessing:
//  1. Normalize newlines → spaces, collapse repeated spaces.
//  2. Capitalize the first letter.
//  3. Add a trailing period if the last character is alphanumeric.
//  4. Pad with 8 leading spaces when the word count is < 5.
func PrepareText(input string) string {
	// Step 1: normalize whitespace (newlines → spaces, collapse doubles).
	s := strings.ReplaceAll(input, "\r\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	// Collapse multiple spaces to single.
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}
	s = strings.TrimSpace(s)

	// Step 2: capitalize first letter.
	if s != "" {
		r, size := utf8.DecodeRuneInString(s)
		if r != utf8.RuneError {
			s = string(unicode.ToUpper(r)) + s[size:]
		}
	}

	// Step 3: add trailing period if last char is alphanumeric.
	if s != "" {
		last, _ := utf8.DecodeLastRuneInString(s)
		if unicode.IsLetter(last) || unicode.IsDigit(last) {
			s += "."
		}
	}

	// Step 4: pad with 8 leading spaces when < 5 words.
	if len(splitWords(s)) < 5 {
		s = "        " + s
	}

	return s
}

// PrepareChunks tokenizes and splits text into ≤maxTokens chunks, applying
// all reference preprocessing steps. Each returned ChunkMetadata includes the
// processed text, token IDs, word count, and generation parameters.
func PrepareChunks(input string, tok Tokenizer, maxTokens int) ([]ChunkMetadata, error) {
	if strings.TrimSpace(input) == "" {
		return nil, fmt.Errorf("input text is empty")
	}

	// Split into sentences first, then apply PrepareText per chunk.
	// We group sentences greedily into ≤maxTokens buckets.
	sentences := splitSentences(input)
	if len(sentences) == 0 {
		sentences = []string{input}
	}

	var chunks []ChunkMetadata
	var pending []string // sentences accumulated into current chunk

	flush := func() error {
		if len(pending) == 0 {
			return nil
		}
		joined := strings.Join(pending, " ")
		prepared := PrepareText(joined)
		ids, err := tok.Encode(prepared)
		if err != nil {
			return fmt.Errorf("encode %q: %w", prepared, err)
		}
		chunks = append(chunks, ChunkMetadata{
			Text:      prepared,
			TokenIDs:  ids,
			NumTokens: len(ids),
			NumWords:  len(splitWords(joined)),
		})
		pending = pending[:0]
		return nil
	}

	for _, sent := range sentences {
		prepared := PrepareText(sent)
		ids, err := tok.Encode(prepared)
		if err != nil {
			return nil, fmt.Errorf("encode sentence %q: %w", sent, err)
		}

		// Count tokens that would result if we add this sentence to pending.
		pendingTokens := 0
		if len(pending) > 0 {
			joined := PrepareText(strings.Join(append(pending, sent), " "))
			tentativeIDs, err := tok.Encode(joined)
			if err != nil {
				return nil, fmt.Errorf("encode combined chunk: %w", err)
			}
			pendingTokens = len(tentativeIDs)
		} else {
			pendingTokens = len(ids)
		}

		if len(pending) > 0 && pendingTokens > maxTokens {
			// Current sentence would exceed budget — flush and start fresh.
			if err := flush(); err != nil {
				return nil, err
			}
		}
		pending = append(pending, sent)
	}
	if err := flush(); err != nil {
		return nil, err
	}

	return chunks, nil
}

// splitWords splits text into non-empty word tokens on whitespace boundaries.
func splitWords(s string) []string {
	return strings.FieldsFunc(s, unicode.IsSpace)
}
