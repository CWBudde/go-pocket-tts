package text

import (
	"bytes"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/text/texttest"
	"github.com/cwbudde/go-pocket-tts/internal/tokenizer"
)

// Ports of upstream text_chunking.py (split_into_best_sentences and helpers).
// texttest.Tokenizer splits punctuation into pieces of its own, like
// SentencePiece does, so the token-boundary logic runs without model files.

func pieces(t *testing.T, s string) []tokenizer.Piece {
	t.Helper()

	p, err := texttest.Tokenizer{}.EncodePieces(s)
	if err != nil {
		t.Fatalf("EncodePieces(%q): %v", s, err)
	}

	return p
}

// split runs splitIntoBestSentences with upstream's test keyword defaults
// (no padding, no semicolon removal) and checks that no text is lost.
func split(t *testing.T, input string, maxTokens int) []string {
	t.Helper()

	opts := upstreamOptions()

	chunks, err := splitIntoBestSentences(texttest.Tokenizer{}, input, maxTokens, opts)
	if err != nil {
		t.Fatalf("splitIntoBestSentences(%q, %d): %v", input, maxTokens, err)
	}

	whole := strings.TrimSpace(mustPrepare(t, input, opts))
	if joined := strings.Join(chunks, " "); joined != whole {
		t.Errorf("chunks joined = %q, want the prepared text %q", joined, whole)
	}

	return chunks
}

func assertChunks(t *testing.T, got, want []string) {
	t.Helper()

	if !slices.Equal(got, want) {
		t.Errorf("chunks = %q\nwant     %q", got, want)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func TestFindBoundaryIndices(t *testing.T) {
	const b = 9 // the boundary token

	for _, tc := range []struct {
		name string
		ids  []int64
		want []int
	}{
		{"empty", nil, []int{0, 0}},
		{"no boundary", []int64{1, 2, 3}, []int{0, 3}},
		{"cut after boundary", []int64{1, 2, b, 3, 4, b}, []int{0, 3, 6}},
		{"run of boundaries is one cut", []int64{1, b, b, b, 2}, []int{0, 4, 5}},
		{"leading boundary", []int64{b, 1}, []int{0, 1, 2}},
		{"trailing run", []int64{1, b, b}, []int{0, 3}},
		{"two boundary ids", []int64{1, b, 2, 8, 3}, []int{0, 2, 4, 5}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := findBoundaryIndices(tc.ids, []int64{b, 8}, nil, false)
			if !slices.Equal(got, tc.want) {
				t.Errorf("findBoundaryIndices(%v) = %v, want %v", tc.ids, got, tc.want)
			}
		})
	}
}

func TestSpanText(t *testing.T) {
	for _, tc := range []struct {
		pieces []string
		want   string
	}{
		{nil, ""},
		{[]string{"▁Pi", "▁is"}, "Pi is"},
		{[]string{"▁", "."}, "."},
		{[]string{"14", "."}, "14."},
		{[]string{"▁", "▁", "▁Hi"}, "  Hi"},
	} {
		ps := make([]tokenizer.Piece, len(tc.pieces))
		for i, s := range tc.pieces {
			ps[i] = tokenizer.Piece{Text: s}
		}

		if got := spanText(ps); got != tc.want {
			t.Errorf("spanText(%q) = %q, want %q", tc.pieces, got, tc.want)
		}
	}
}

func TestSpanTextRoundTrip(t *testing.T) {
	for _, s := range []string{"Hello, world.", "Pi is 3.14!", "  padded", "98.6°F"} {
		if got := spanText(pieces(t, s)); got != s {
			t.Errorf("spanText(EncodePieces(%q)) = %q", s, got)
		}
	}
}

func TestIsDecimalPeriodBoundary(t *testing.T) {
	for _, tc := range []struct {
		text  string
		start int
		want  bool
	}{
		{"Pi is 3.14", 4, true},           // ▁Pi ▁is ▁3 . | 14
		{"Version 2. Three", 3, false},    // ▁Version ▁2 . | ▁Three
		{"Version 2. 3 apples", 3, true},  // upstream quirk: decode drops the space
		{"End. 5 apples", 2, false},       // no digit before the period
		{".5", 2, false},                  // prefix "." is shorter than two characters
		{"٣.١٤", 2, true},                 // Arabic-Indic digits are digits too
		{"Pi is 3.14", 3, false},          // prefix "Pi is 3" does not end with a period
		{"Pi is 3.", 4, false},            // empty suffix
		{"It is 3.x", 4, false},           // suffix does not start with a digit
		{"It is 3. 14 and 15", 3, false},  // prefix "It is 3" (start before the period)
		{"It is 3. 14 and 15", 4, true},   // ▁It ▁is ▁3 . | ▁14
		{"Version 2.0 is out.", 3, true},  // ▁Version ▁2 . | 0
		{"Version 2.0 is out.", 7, false}, // end of text
	} {
		ps := pieces(t, tc.text)
		if got := isDecimalPeriodBoundary(ps, tc.start); got != tc.want {
			t.Errorf("isDecimalPeriodBoundary(%q, %d) = %v, want %v", tc.text, tc.start, got, tc.want)
		}
	}
}

func TestSegmentsFromBoundaries(t *testing.T) {
	ps := pieces(t, "Hi there. Bye.")

	ids := make([]int64, len(ps))
	for i, p := range ps {
		ids[i] = p.ID
	}

	got := segmentsFromBoundaries(ps, findBoundaryIndices(ids, []int64{texttest.PieceID(".")}, nil, false))

	want := []segment{{numTokens: 3, text: "Hi there."}, {numTokens: 2, text: "Bye."}}
	if !slices.Equal(got, want) {
		t.Errorf("segmentsFromBoundaries = %+v, want %+v", got, want)
	}
}

// ---------------------------------------------------------------------------
// splitIntoBestSentences — upstream tests/test_split_sentences.py
// ---------------------------------------------------------------------------

func TestSplit_ShortTextSingleChunk(t *testing.T) {
	assertChunks(t, split(t, "Hello world.", 50), []string{"Hello world."})
}

func TestSplit_MultipleSentencesSplit(t *testing.T) {
	// Every sentence is 4 tokens (▁First ▁sentence ▁here .), so two fit in 10.
	got := split(t, "First sentence here. Second sentence here. Third sentence here. Fourth sentence here.", 10)
	assertChunks(t, got, []string{
		"First sentence here. Second sentence here.",
		"Third sentence here. Fourth sentence here.",
	})
}

const taleOfTwoCities = "It was the best of times, it was the worst of times, " +
	"it was the age of wisdom, it was the age of foolishness, " +
	"it was the epoch of belief, it was the epoch of incredulity, " +
	"it was the season of Light, it was the season of Darkness, " +
	"it was the spring of hope, it was the winter of despair"

func TestSplit_LongSentenceWithCommasIsSplit(t *testing.T) {
	// One 70-token sentence; each comma clause is 7 tokens, so seven fit in 50.
	got := split(t, taleOfTwoCities, 50)
	assertChunks(t, got, []string{
		"It was the best of times, it was the worst of times, it was the age of wisdom, " +
			"it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, " +
			"it was the season of Light,",
		"it was the season of Darkness, it was the spring of hope, it was the winter of despair.",
	})

	rejoined := strings.ToLower(strings.Join(got, " "))
	for _, phrase := range []string{"best of times", "worst of times", "age of foolishness", "winter of despair"} {
		if !strings.Contains(rejoined, phrase) {
			t.Errorf("%q lost after splitting", phrase)
		}
	}
}

func TestSplit_LongSentenceWithCommasRespectsMaxTokens(t *testing.T) {
	const maxTokens = 20

	input := "It was the best of times, it was the worst of times, " +
		"it was the age of wisdom, it was the age of foolishness, " +
		"it was the epoch of belief, it was the epoch of incredulity"

	for _, chunk := range split(t, input, maxTokens) {
		ids, err := texttest.Tokenizer{}.Encode(strings.TrimSpace(chunk))
		if err != nil {
			t.Fatal(err)
		}

		if len(ids) > maxTokens {
			t.Errorf("chunk %q has %d tokens, want ≤ %d", chunk, len(ids), maxTokens)
		}
	}
}

func TestSplit_MixedSentencesAndCommas(t *testing.T) {
	input := "Short sentence. " +
		"This is a very long sentence with many clauses, separated by commas, " +
		"that goes on and on, and on some more, without any periods at all, " +
		"until it finally reaches a period. " +
		"Another short one."

	// Segments: 3 | 10 4 6 5 6 7 (the 38-token sentence, sub-split) | 4.
	assertChunks(t, split(t, input, 20), []string{
		"Short sentence. This is a very long sentence with many clauses, separated by commas,",
		"that goes on and on, and on some more, without any periods at all,",
		"until it finally reaches a period. Another short one.",
	})
}

func TestSplit_NoSplitPointsStaysSingleChunk(t *testing.T) {
	got := split(t, "one two three four five six seven eight nine ten eleven twelve", 5)
	assertChunks(t, got, []string{"One two three four five six seven eight nine ten eleven twelve."})
}

func TestSplit_SemicolonsAndColonsAlsoSplit(t *testing.T) {
	input := "First clause here; second clause here; third clause here; " +
		"fourth clause here: fifth clause here; sixth clause here"

	assertChunks(t, split(t, input, 15), []string{
		"First clause here; second clause here; third clause here;",
		"fourth clause here: fifth clause here; sixth clause here.",
	})
}

func TestSplit_ShortSentenceNotAffectedByCommaSplitting(t *testing.T) {
	assertChunks(t, split(t, "Hello, world.", 50), []string{"Hello, world."})
}

func TestSplit_EmptyStringErrors(t *testing.T) {
	_, err := splitIntoBestSentences(texttest.Tokenizer{}, "", 50, upstreamOptions())
	if !errors.Is(err, ErrEmptyText) {
		t.Errorf("err = %v, want ErrEmptyText", err)
	}
}

func TestSplit_OversizedClauseWithoutCommasStillReturns(t *testing.T) {
	words := make([]string, 20)
	for i := range words {
		words[i] = fmt.Sprintf("word%d", i)
	}

	input := strings.Join(words, " ")

	got := split(t, input, 5)
	if len(got) != 1 {
		t.Fatalf("chunks = %q, want 1", got)
	}

	if strings.TrimRight(strings.ToLower(got[0]), ".") != input {
		t.Errorf("chunk = %q, want %q (capitalized, with a period)", got[0], input)
	}
}

// ---------------------------------------------------------------------------
// Decimal periods (upstream #162 / #217)
// ---------------------------------------------------------------------------

func TestSplit_DecimalsAreNotSplitOnPeriod(t *testing.T) {
	input := "The average human body temperature is 98.6°F, which is a common decimal used in medicine."
	assertChunks(t, split(t, input, 50), []string{input})
}

func TestSplit_MultipleDecimalsInOneSentence(t *testing.T) {
	// No sentence boundary and no comma: the 12 tokens stay one chunk even
	// over budget, and neither decimal is cut.
	for _, maxTokens := range []int{50, 3} {
		assertChunks(t, split(t, "Pi is 3.14 and e is 2.718.", maxTokens), []string{"Pi is 3.14 and e is 2.718."})
	}
}

func TestSplit_DecimalFollowedBySentenceBoundary(t *testing.T) {
	input := "The average human body temperature is 98.6°F, " +
		"which is a common decimal used in medicine. " +
		"Pi is 3.14 and e is 2.718."

	// Segments: 21 (over budget → 12 + 9 on the comma) | 12.
	assertChunks(t, split(t, input, 20), []string{
		"The average human body temperature is 98.6°F,",
		"which is a common decimal used in medicine.",
		"Pi is 3.14 and e is 2.718.",
	})
}

func TestSplit_SentencePeriodAfterDecimalStillSplits(t *testing.T) {
	const input = "Version 2.0 is out. Pi is 3.14."

	assertChunks(t, split(t, input, 50), []string{input})
	// "Version 2.0 is out." is 7 tokens, "Pi is 3.14." is 6.
	assertChunks(t, split(t, input, 7), []string{"Version 2.0 is out.", "Pi is 3.14."})
}

// Upstream decodes the suffix on its own, which drops the space after the
// period, so a sentence ending in a digit followed by one starting with a
// digit is treated as a decimal. The port keeps that behaviour.
func TestSplit_DigitPeriodSpaceDigitIsNotSplit(t *testing.T) {
	assertChunks(t, split(t, "Version 2. 3 apples.", 3), []string{"Version 2. 3 apples."})
	assertChunks(t, split(t, "Version 2. Three apples.", 3), []string{"Version 2.", "Three apples."})
}

// ---------------------------------------------------------------------------
// Boundary tokens and grouping
// ---------------------------------------------------------------------------

func TestSplit_QuestionAndExclamationMarks(t *testing.T) {
	assertChunks(t, split(t, "Really? Yes! Fine.", 2), []string{"Really?", "Yes!", "Fine."})
}

func TestSplit_RunOfBoundaryTokensIsOneCut(t *testing.T) {
	assertChunks(t, split(t, "Wait... what?! Fine.", 2), []string{"Wait...", "what?!", "Fine."})
}

func TestSplit_CommaClausesOnlyForOversizedSentences(t *testing.T) {
	// Segments: 3 | 10 | 12. The 10-token sentence fits and keeps its commas
	// (sub-split, "Red, green," would join "Go now."); the 12-token one does
	// not and is split into 2 2 6 2 on them.
	input := "Go now. Red, green, and blue are colors here. Cats, dogs, and birds are animals too, ok."
	assertChunks(t, split(t, input, 10), []string{
		"Go now.",
		"Red, green, and blue are colors here.",
		"Cats, dogs, and birds are animals too,",
		"ok.",
	})
}

// Grouping adds up the segment token counts, like upstream; it does not
// re-encode the prepared (padded) chunk.
func TestPrepareChunks_GroupsBySegmentTokenCounts(t *testing.T) {
	chunks, err := PrepareChunks("Hi. Yo.", texttest.Tokenizer{}, 4, DefaultOptions())
	if err != nil {
		t.Fatalf("PrepareChunks: %v", err)
	}

	if len(chunks) != 1 || chunks[0].Text != "        Hi. Yo." {
		t.Errorf("chunks = %+v, want one padded chunk %q", chunks, "        Hi. Yo.")
	}
}

func TestSplit_WarnsAboutOversizedChunk(t *testing.T) {
	var buf bytes.Buffer

	prev := slog.Default()

	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(prev) })

	split(t, "Short one.", 5)

	if buf.Len() != 0 {
		t.Errorf("unexpected log output for a chunk within budget: %s", buf.String())
	}

	split(t, "one two three four five six seven", 5)

	if out := buf.String(); !strings.Contains(out, "level=WARN") || !strings.Contains(out, "tokens=8") {
		t.Errorf("log output = %q, want a warning about the 8-token chunk", out)
	}
}

// ---------------------------------------------------------------------------
// PrepareChunks on top of the splitter
// ---------------------------------------------------------------------------

func TestPrepareChunks_DecimalsStayInOneChunk(t *testing.T) {
	chunks, err := PrepareChunks("Version 2.0 is out. Pi is 3.14.", texttest.Tokenizer{}, 50, DefaultOptions())
	if err != nil {
		t.Fatalf("PrepareChunks: %v", err)
	}

	if len(chunks) != 1 || chunks[0].Text != "Version 2.0 is out. Pi is 3.14." {
		t.Errorf("chunks = %+v, want one chunk %q", chunks, "Version 2.0 is out. Pi is 3.14.")
	}
}

// Like upstream generate_audio_stream, every chunk is prepared again: a chunk
// that starts with a lower-case clause is capitalized.
func TestPrepareChunks_PreparesEachChunk(t *testing.T) {
	chunks, err := PrepareChunks(taleOfTwoCities, texttest.Tokenizer{}, 50, DefaultOptions())
	if err != nil {
		t.Fatalf("PrepareChunks: %v", err)
	}

	if len(chunks) != 2 {
		t.Fatalf("got %d chunks, want 2", len(chunks))
	}

	const want = "It was the season of Darkness, it was the spring of hope, it was the winter of despair."
	if chunks[1].Text != want {
		t.Errorf("chunks[1].Text = %q, want %q", chunks[1].Text, want)
	}

	ids, _ := texttest.Tokenizer{}.Encode(want)
	if !slices.Equal(chunks[1].TokenIDs, ids) || chunks[1].NumTokens != len(ids) {
		t.Errorf("chunks[1] token ids do not match the encoded chunk text")
	}
}
