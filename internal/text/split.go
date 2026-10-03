package text

import (
	"fmt"
	"log/slog"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/cwbudde/go-pocket-tts/internal/tokenizer"
)

// Port of upstream text_chunking.py split_into_best_sentences: the text is
// split on token boundaries, not characters, so it reproduces the chunks the
// reference produces for the same tokenizer.

const (
	// sentenceMarks is encoded and, minus its first token (the dummy-prefix
	// '▁'), gives the tokens that end a sentence.
	sentenceMarks = ".!...?"
	// clauseMarks gives the tokens an oversized sentence is sub-split on.
	clauseMarks = ",;:"
	// wordStart is the SentencePiece meta symbol that marks a word start.
	wordStart = "▁"
	// warnPreviewLen is how many characters of an oversized chunk are logged.
	warnPreviewLen = 50
)

// segment is a decoded token span and its token count.
type segment struct {
	numTokens int
	text      string
}

// spanText decodes pieces like SentencePiece decode: '▁' becomes a space and
// the dummy-prefix space at the start is dropped.
func spanText(pieces []tokenizer.Piece) string {
	var sb strings.Builder
	for _, p := range pieces {
		sb.WriteString(p.Text)
	}

	s := strings.ReplaceAll(sb.String(), wordStart, " ")

	return strings.TrimPrefix(s, " ")
}

func pieceIDs(pieces []tokenizer.Piece) []int64 {
	ids := make([]int64, len(pieces))
	for i, p := range pieces {
		ids[i] = p.ID
	}

	return ids
}

// boundaryTokens encodes marks and drops the first token, which is the
// dummy-prefix '▁' (upstream: `_, *tokens = tokenizer(marks)[0].tolist()`).
func boundaryTokens(tok Tokenizer, marks string) ([]int64, error) {
	pieces, err := tok.EncodePieces(marks)
	if err != nil {
		return nil, fmt.Errorf("encode %q: %w", marks, err)
	}

	if len(pieces) == 0 {
		return nil, nil
	}

	return pieceIDs(pieces[1:]), nil
}

// isDecimalPeriodBoundary reports whether pieces[start:] begins right after a
// decimal period: the decoded prefix ends with a digit and '.', and the
// decoded suffix starts with a digit (upstream _is_decimal_period_boundary).
func isDecimalPeriodBoundary(pieces []tokenizer.Piece, start int) bool {
	prefix := []rune(spanText(pieces[:start]))
	suffix := spanText(pieces[start:])
	first, _ := utf8.DecodeRuneInString(suffix)

	return len(prefix) >= 2 &&
		prefix[len(prefix)-1] == '.' &&
		isPyDigit(prefix[len(prefix)-2]) &&
		suffix != "" &&
		isPyDigit(first)
}

// pyDigitNotDecimal holds the characters Python str.isdigit accepts beyond the
// decimal digits (Numeric_Type=Digit: superscripts, circled digits, ...).
// Generated with Python: isdigit() and not isdecimal(); identical for
// unicodedata 15.0 and 16.0.
var pyDigitNotDecimal = &unicode.RangeTable{
	R16: []unicode.Range16{
		{Lo: 0x00B2, Hi: 0x00B3, Stride: 1},
		{Lo: 0x00B9, Hi: 0x00B9, Stride: 1},
		{Lo: 0x1369, Hi: 0x1371, Stride: 1},
		{Lo: 0x19DA, Hi: 0x19DA, Stride: 1},
		{Lo: 0x2070, Hi: 0x2070, Stride: 1},
		{Lo: 0x2074, Hi: 0x2079, Stride: 1},
		{Lo: 0x2080, Hi: 0x2089, Stride: 1},
		{Lo: 0x2460, Hi: 0x2468, Stride: 1},
		{Lo: 0x2474, Hi: 0x247C, Stride: 1},
		{Lo: 0x2488, Hi: 0x2490, Stride: 1},
		{Lo: 0x24EA, Hi: 0x24EA, Stride: 1},
		{Lo: 0x24F5, Hi: 0x24FD, Stride: 1},
		{Lo: 0x24FF, Hi: 0x24FF, Stride: 1},
		{Lo: 0x2776, Hi: 0x277E, Stride: 1},
		{Lo: 0x2780, Hi: 0x2788, Stride: 1},
		{Lo: 0x278A, Hi: 0x2792, Stride: 1},
	},
	R32: []unicode.Range32{
		{Lo: 0x10A40, Hi: 0x10A43, Stride: 1},
		{Lo: 0x10E60, Hi: 0x10E68, Stride: 1},
		{Lo: 0x11052, Hi: 0x1105A, Stride: 1},
		{Lo: 0x1F100, Hi: 0x1F10A, Stride: 1},
	},
	LatinOffset: 2,
}

// isPyDigit reports whether Python str.isdigit accepts r.
func isPyDigit(r rune) bool {
	return unicode.IsDigit(r) || unicode.Is(pyDigitNotDecimal, r)
}

// findBoundaryIndices returns the token indices the text is cut at: 0, the
// index of the first non-boundary token after every run of boundary tokens,
// and len(ids). With skipDecimalPeriods, a cut right after a decimal period
// is skipped; pieces must then be the pieces of ids. Ports upstream
// _find_boundary_indices.
func findBoundaryIndices(ids, boundary []int64, pieces []tokenizer.Piece, skipDecimalPeriods bool) []int {
	isBoundary := make(map[int64]bool, len(boundary))
	for _, id := range boundary {
		isBoundary[id] = true
	}

	indices := []int{0}
	afterBoundary := false

	for i, id := range ids {
		switch {
		case isBoundary[id]:
			afterBoundary = true
		case afterBoundary:
			afterBoundary = false

			if skipDecimalPeriods && pieces != nil && isDecimalPeriodBoundary(pieces, i) {
				continue
			}

			indices = append(indices, i)
		}
	}

	return append(indices, len(ids))
}

// segmentsFromBoundaries decodes the spans between consecutive boundary
// indices (upstream _segments_from_boundaries).
func segmentsFromBoundaries(pieces []tokenizer.Piece, indices []int) []segment {
	segments := make([]segment, 0, len(indices)-1)
	for i := range len(indices) - 1 {
		start, end := indices[i], indices[i+1]
		segments = append(segments, segment{numTokens: end - start, text: spanText(pieces[start:end])})
	}

	return segments
}

// splitIntoBestSentences ports upstream split_into_best_sentences: prepare the
// whole text, split it into sentences on sentence-end tokens (not on decimal
// periods), sub-split sentences over maxTokens on comma, semicolon and colon
// tokens, and group the segments greedily into chunks of at most maxTokens
// tokens. A segment that does not fit on its own still becomes a chunk; such
// chunks are logged.
func splitIntoBestSentences(tok Tokenizer, input string, maxTokens int, opts Options) ([]string, error) {
	prepared, _, err := PrepareText(input, opts)
	if err != nil {
		return nil, err
	}

	pieces, err := tok.EncodePieces(strings.TrimSpace(prepared))
	if err != nil {
		return nil, fmt.Errorf("encode text: %w", err)
	}

	sentenceEnds, err := boundaryTokens(tok, sentenceMarks)
	if err != nil {
		return nil, err
	}

	indices := findBoundaryIndices(pieceIDs(pieces), sentenceEnds, pieces, true)

	segments, err := splitOversized(tok, segmentsFromBoundaries(pieces, indices), maxTokens)
	if err != nil {
		return nil, err
	}

	chunks := groupSegments(segments, maxTokens)

	err = warnOversized(tok, chunks, maxTokens)
	if err != nil {
		return nil, err
	}

	return chunks, nil
}

// splitOversized sub-splits each sentence over maxTokens on clause tokens,
// when that yields more than one segment, so long sentences without a period
// do not make the model skip words.
func splitOversized(tok Tokenizer, sentences []segment, maxTokens int) ([]segment, error) {
	clauseEnds, err := boundaryTokens(tok, clauseMarks)
	if err != nil {
		return nil, err
	}

	refined := make([]segment, 0, len(sentences))

	for _, s := range sentences {
		if s.numTokens <= maxTokens {
			refined = append(refined, s)

			continue
		}

		pieces, err := tok.EncodePieces(strings.TrimSpace(s.text))
		if err != nil {
			return nil, fmt.Errorf("encode sentence: %w", err)
		}

		sub := segmentsFromBoundaries(pieces, findBoundaryIndices(pieceIDs(pieces), clauseEnds, nil, false))
		if len(sub) > 1 {
			refined = append(refined, sub...)
		} else {
			refined = append(refined, s)
		}
	}

	return refined, nil
}

// groupSegments joins consecutive segments with a space while their summed
// token counts stay within maxTokens.
func groupSegments(segments []segment, maxTokens int) []string {
	var (
		chunks  []string
		current string
		tokens  int
	)

	for _, s := range segments {
		switch {
		case current == "":
			current, tokens = s.text, s.numTokens
		case tokens+s.numTokens > maxTokens:
			chunks = append(chunks, strings.TrimSpace(current))
			current, tokens = s.text, s.numTokens
		default:
			current += " " + s.text
			tokens += s.numTokens
		}
	}

	if current != "" {
		chunks = append(chunks, strings.TrimSpace(current))
	}

	return chunks
}

// warnOversized logs every chunk that encodes to more than maxTokens tokens.
func warnOversized(tok Tokenizer, chunks []string, maxTokens int) error {
	for _, chunk := range chunks {
		ids, err := tok.Encode(strings.TrimSpace(chunk))
		if err != nil {
			return fmt.Errorf("encode chunk: %w", err)
		}

		if len(ids) > maxTokens {
			slog.Warn("text chunk exceeds the token budget, generation may skip words",
				"tokens", len(ids), "max", maxTokens, "chunk", preview(chunk))
		}
	}

	return nil
}

// preview returns the first warnPreviewLen characters of s.
func preview(s string) string {
	r := []rune(s)
	if len(r) <= warnPreviewLen {
		return s
	}

	return string(r[:warnPreviewLen]) + "..."
}
