package text

import "strings"

// ChunkBySentence splits text into chunks at sentence boundaries (., !, ?),
// grouping consecutive sentences together while staying within maxChars per chunk.
// If maxChars is 0, no splitting is performed.
// Sentences that individually exceed maxChars are kept intact as a single chunk.
func ChunkBySentence(text string, maxChars int) []string {
	if maxChars <= 0 {
		return []string{text}
	}

	// Split into sentences by scanning for terminators.
	sentences := splitSentences(text)
	if len(sentences) <= 1 {
		return []string{text}
	}

	var chunks []string
	var current strings.Builder

	for _, s := range sentences {
		if current.Len() == 0 {
			current.WriteString(s)
			continue
		}
		// Would appending this sentence (with a space separator) exceed the limit?
		if current.Len()+1+len(s) > maxChars {
			chunks = append(chunks, current.String())
			current.Reset()
			current.WriteString(s)
		} else {
			current.WriteByte(' ')
			current.WriteString(s)
		}
	}
	if current.Len() > 0 {
		chunks = append(chunks, current.String())
	}

	return chunks
}

// splitSentences splits text on sentence-ending punctuation (., !, ?),
// keeping the terminator attached to its sentence.
// Empty segments are dropped.
func splitSentences(text string) []string {
	var sentences []string
	start := 0

	for i, r := range text {
		if r == '.' || r == '!' || r == '?' {
			s := strings.TrimSpace(text[start : i+1])
			if s != "" {
				sentences = append(sentences, s)
			}
			start = i + 1
		}
	}

	// Trailing text after the last terminator (if any).
	if start < len(text) {
		s := strings.TrimSpace(text[start:])
		if s != "" {
			sentences = append(sentences, s)
		}
	}

	return sentences
}
