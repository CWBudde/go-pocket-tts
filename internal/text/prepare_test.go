package text

import (
	"strings"
	"testing"
)

// stubTokenizer is a minimal Tokenizer for testing that counts words as tokens.
type stubTokenizer struct{}

func (s *stubTokenizer) Encode(text string) ([]int64, error) {
	// Split naively on spaces; each non-empty word = 1 token.
	words := splitWords(text)

	ids := make([]int64, len(words))
	for i := range ids {
		ids[i] = int64(i + 1)
	}

	return ids, nil
}

// ---------------------------------------------------------------------------
// PrepareText
// ---------------------------------------------------------------------------

func TestPrepareText_CapitalizesFirstLetter(t *testing.T) {
	// "hello world." has 2 words → padded with 8 spaces; first non-space should be 'H'.
	got := PrepareText("hello world.")

	trimmed := strings.TrimLeft(got, " ")
	if len(trimmed) == 0 || trimmed[0] != 'H' {
		t.Errorf("PrepareText(%q) = %q, want first letter capitalized (after stripping leading spaces)", "hello world.", got)
	}
}

func TestPrepareText_AlreadyCapitalized(t *testing.T) {
	got := PrepareText("Hello world.")

	trimmed := strings.TrimLeft(got, " ")
	if len(trimmed) == 0 || trimmed[0] != 'H' {
		t.Errorf("PrepareText(%q) = %q, want H after stripping leading spaces", "Hello world.", got)
	}
}

func TestPrepareText_AddsPeriodWhenMissing(t *testing.T) {
	got := PrepareText("hello world")
	if len(got) == 0 || got[len(got)-1] != '.' {
		t.Errorf("PrepareText(%q) = %q, want trailing period", "hello world", got)
	}
}

func TestPrepareText_DoesNotAddPeriodWhenPunctPresent(t *testing.T) {
	cases := []struct {
		input     string
		lastPunct byte
	}{
		{"Hello world.", '.'},
		{"Hello world!", '!'},
		{"Hello world?", '?'},
	}
	for _, c := range cases {
		got := PrepareText(c.input)
		if got[len(got)-1] != c.lastPunct {
			t.Errorf("PrepareText(%q) = %q, last char = %q, want %q", c.input, got, got[len(got)-1], c.lastPunct)
		}
	}
}

func TestPrepareText_PadsShortInput(t *testing.T) {
	// "hi" is 1 word (< 5) → padded with 8 leading spaces.
	got := PrepareText("hi")
	if len(got) < 8 || got[:8] != "        " {
		t.Errorf("PrepareText(%q) = %q, want 8 leading spaces for short input", "hi", got)
	}
}

func TestPrepareText_DoesNotPadFiveWordInput(t *testing.T) {
	// Exactly 5 words → no leading space padding.
	got := PrepareText("one two three four five")
	if len(got) > 0 && got[0] == ' ' {
		t.Errorf("PrepareText(%q) = %q, should not start with space for 5-word input", "one two three four five", got)
	}
}

func TestPrepareText_NormalizesNewlines(t *testing.T) {
	got := PrepareText("hello\nworld")
	if strings.ContainsRune(got, '\n') {
		t.Errorf("PrepareText(%q) = %q, newlines should be replaced with spaces", "hello\nworld", got)
	}
}

func TestPrepareText_CollapsesDoubleSpaces(t *testing.T) {
	// After collapsing "hello  world" → "hello world", then PrepareText processes it.
	// The leading padding (8 spaces) is fine; no double-space should appear *within* the content.
	got := PrepareText("hello  world")
	// Strip the leading 8-space pad (if present) before checking.
	inner := strings.TrimLeft(got, " ")
	if strings.Contains(inner, "  ") {
		t.Errorf("PrepareText(%q) = %q, double spaces in content should be collapsed", "hello  world", got)
	}
}

func TestPrepareText_ExactlyFourWords_IsPadded(t *testing.T) {
	// 4 words < 5 → padding applied.
	got := PrepareText("one two three four.")
	if len(got) < 8 || got[:8] != "        " {
		t.Errorf("PrepareText(%q) = %q, want 8 leading spaces for 4-word input", "one two three four.", got)
	}
}

// ---------------------------------------------------------------------------
// ChunkMetadata
// ---------------------------------------------------------------------------

func TestChunkMetadata_MaxFrames(t *testing.T) {
	// Reference formula: ceil((token_count / 3.0 + 2.0) * 12.5)
	// Matches Python: ceil(gen_len_sec * frame_rate) where gen_len_sec = n/3+2, frame_rate = 12.5.
	tests := []struct {
		numTokens int
		want      float64
	}{
		// 3 tokens → ceil((3/3 + 2) * 12.5) = ceil(3 * 12.5) = ceil(37.5) = 38
		{3, 38},
		// 4 tokens → ceil((4/3 + 2) * 12.5) = ceil(3.333 * 12.5) = ceil(41.667) = 42
		{4, 42},
		// 9 tokens → ceil((9/3 + 2) * 12.5) = ceil(5 * 12.5) = ceil(62.5) = 63
		{9, 63},
		// 10 tokens → ceil((10/3 + 2) * 12.5) = ceil(5.333 * 12.5) = ceil(66.667) = 67
		{10, 67},
		// 14 tokens → ceil((14/3 + 2) * 12.5) = ceil(6.667 * 12.5) = ceil(83.333) = 84
		{14, 84},
		// 50 tokens → ceil((50/3 + 2) * 12.5) = ceil(18.667 * 12.5) = ceil(233.333) = 234
		{50, 234},
	}
	for _, tt := range tests {
		c := ChunkMetadata{NumTokens: tt.numTokens}
		if got := c.MaxFrames(); got != tt.want {
			t.Errorf("MaxFrames() with %d tokens = %v, want %v", tt.numTokens, got, tt.want)
		}
	}
}

func TestChunkMetadata_FramesAfterEOS_ShortInput(t *testing.T) {
	// ≤ 4 words → base 3 + 2 = 5 (matching reference implementation)
	if got := (ChunkMetadata{NumWords: 4}).FramesAfterEOS(); got != 5 {
		t.Errorf("FramesAfterEOS() = %d, want 5", got)
	}

	if got := (ChunkMetadata{NumWords: 1}).FramesAfterEOS(); got != 5 {
		t.Errorf("FramesAfterEOS() = %d, want 5", got)
	}
}

func TestChunkMetadata_FramesAfterEOS_LongInput(t *testing.T) {
	// > 4 words → base 1 + 2 = 3 (matching reference implementation)
	if got := (ChunkMetadata{NumWords: 5}).FramesAfterEOS(); got != 3 {
		t.Errorf("FramesAfterEOS() = %d, want 3", got)
	}
}

// ---------------------------------------------------------------------------
// PrepareChunks
// ---------------------------------------------------------------------------

func TestPrepareChunks_SingleChunkShortText(t *testing.T) {
	tok := &stubTokenizer{}

	chunks, err := PrepareChunks("hello world.", tok, 50)
	if err != nil {
		t.Fatalf("PrepareChunks error: %v", err)
	}

	if len(chunks) != 1 {
		t.Fatalf("PrepareChunks returned %d chunks, want 1", len(chunks))
	}
}

func TestPrepareChunks_MetadataPopulated(t *testing.T) {
	tok := &stubTokenizer{}

	chunks, err := PrepareChunks("hello world.", tok, 50)
	if err != nil {
		t.Fatalf("PrepareChunks error: %v", err)
	}

	c := chunks[0]
	if c.NumTokens <= 0 {
		t.Errorf("NumTokens = %d, want > 0", c.NumTokens)
	}

	if c.NumWords <= 0 {
		t.Errorf("NumWords = %d, want > 0", c.NumWords)
	}

	if c.MaxFrames() <= 0 {
		t.Errorf("MaxFrames() = %v, want > 0", c.MaxFrames())
	}
}

func TestPrepareChunks_ReturnsTokenIDs(t *testing.T) {
	tok := &stubTokenizer{}

	chunks, err := PrepareChunks("hello world.", tok, 50)
	if err != nil {
		t.Fatalf("PrepareChunks error: %v", err)
	}

	if len(chunks[0].TokenIDs) == 0 {
		t.Error("TokenIDs should be non-empty")
	}
}

func TestPrepareChunks_SplitsLongText(t *testing.T) {
	// stubTokenizer: 1 token per space-delimited word.
	// Two sentences each with 2 words → 2 tokens each, but after PrepareText
	// they get padding words. Force split with maxTokens=3.
	tok := &stubTokenizer{}

	chunks, err := PrepareChunks("First sentence. Second sentence.", tok, 3)
	if err != nil {
		t.Fatalf("PrepareChunks error: %v", err)
	}

	if len(chunks) < 2 {
		t.Errorf("PrepareChunks returned %d chunks, want ≥2 for small maxTokens with multi-sentence input", len(chunks))
	}
}

func TestPrepareChunks_EmptyTextError(t *testing.T) {
	tok := &stubTokenizer{}

	_, err := PrepareChunks("", tok, 50)
	if err == nil {
		t.Error("PrepareChunks(\"\") should return error")
	}
}

func TestPrepareChunks_WhitespaceOnlyError(t *testing.T) {
	tok := &stubTokenizer{}

	_, err := PrepareChunks("   \n\t  ", tok, 50)
	if err == nil {
		t.Error("PrepareChunks(whitespace) should return error")
	}
}

// ---------------------------------------------------------------------------
// PrepareText — whitespace collapse edge cases
// ---------------------------------------------------------------------------

func TestPrepareText_CollapseTripleSpaces(t *testing.T) {
	// Go collapses all runs of multiple spaces (unlike Python which only does
	// a single replace("  ", " ") pass). Verify Go fully collapses ≥ 3 spaces.
	got := PrepareText("hello   world   test")

	inner := strings.TrimLeft(got, " ")
	if strings.Contains(inner, "  ") {
		t.Errorf("PrepareText with triple spaces: %q still contains double spaces in content", got)
	}
}

func TestPrepareText_MixedNewlinesAndSpaces(t *testing.T) {
	got := PrepareText("hello\r\nworld\n\ntest")

	inner := strings.TrimLeft(got, " ")
	if strings.ContainsAny(inner, "\r\n") {
		t.Errorf("PrepareText(%q) still contains newlines: %q", "hello\r\nworld\n\ntest", got)
	}

	if strings.Contains(inner, "  ") {
		t.Errorf("PrepareText with newlines should collapse to single spaces: %q", got)
	}
}

func TestPrepareText_DigitFirstChar(t *testing.T) {
	// First char is a digit — ToUpper is a no-op, should not crash.
	got := PrepareText("3 cats")

	inner := strings.TrimLeft(got, " ")
	if inner[0] != '3' {
		t.Errorf("PrepareText(%q) = %q, expected to start with '3'", "3 cats", inner)
	}
}

func TestPrepareText_PunctuationFirstChar(t *testing.T) {
	// Non-letter, non-digit first char.
	got := PrepareText("...hello world test one two")
	if got[0] == ' ' {
		t.Errorf("PrepareText starting with ... should not be padded (5+ words)")
	}
}

// ---------------------------------------------------------------------------
// splitSentences — edge cases
// ---------------------------------------------------------------------------

func TestSplitSentences_Ellipsis(t *testing.T) {
	// "Hello... world" — each dot triggers a split in the current implementation.
	// Document the current behavior (character-level splitting).
	got := splitSentences("Hello... world")
	// Current behavior: splits at each dot, so we get fragments.
	// After TrimSpace and empty-string filtering, "Hello." is captured at first dot,
	// then "." at second dot, "." at third dot, and " world" as trailing text.
	// The empty strings from "." are filtered out.
	nonEmpty := 0

	for _, s := range got {
		if strings.TrimSpace(s) != "" {
			nonEmpty++
		}
	}

	if nonEmpty < 1 {
		t.Errorf("splitSentences(%q) produced no non-empty sentences, got %v", "Hello... world", got)
	}
	// Verify all fragments are non-empty (TrimSpace filtering in splitSentences).
	for i, s := range got {
		if strings.TrimSpace(s) == "" {
			t.Errorf("splitSentences(%q)[%d] = %q, should not contain empty segments", "Hello... world", i, s)
		}
	}
}

func TestSplitSentences_CombinedPunctuation(t *testing.T) {
	// "Hello?! World" — splits at ? and at !, producing fragments.
	got := splitSentences("Hello?! World")
	if len(got) < 2 {
		t.Errorf("splitSentences(%q) = %v, want ≥2 parts", "Hello?! World", got)
	}
	// First part should contain "Hello?"
	if !strings.Contains(got[0], "Hello") {
		t.Errorf("splitSentences(%q)[0] = %q, want to contain 'Hello'", "Hello?! World", got[0])
	}
}

func TestSplitSentences_Abbreviation(t *testing.T) {
	// "Dr. Smith said hello." — the current character-level splitter splits at "Dr.".
	// Document the current (imperfect) behavior.
	got := splitSentences("Dr. Smith said hello.")
	// Should produce at least "Dr." as one segment.
	if len(got) < 2 {
		t.Errorf("splitSentences(%q) = %v, expected ≥2 parts (character-level split at Dr.)", "Dr. Smith said hello.", got)
	}
}

func TestSplitSentences_NoPunctuation(t *testing.T) {
	got := splitSentences("hello world no punctuation")
	if len(got) != 1 || got[0] != "hello world no punctuation" {
		t.Errorf("splitSentences without punctuation = %v, want single element", got)
	}
}

func TestSplitSentences_BasicTwoSentences(t *testing.T) {
	got := splitSentences("First sentence. Second sentence.")
	if len(got) != 2 {
		t.Fatalf("splitSentences = %v, want 2 sentences", got)
	}

	if got[0] != "First sentence." {
		t.Errorf("sentence[0] = %q, want %q", got[0], "First sentence.")
	}

	if got[1] != "Second sentence." {
		t.Errorf("sentence[1] = %q, want %q", got[1], "Second sentence.")
	}
}

// ---------------------------------------------------------------------------
// PrepareChunks — budget accounting and NumWords consistency
// ---------------------------------------------------------------------------

func TestPrepareChunks_NumWordsFromRawText(t *testing.T) {
	// NumWords should reflect the raw (un-prepared) joined text, not the
	// PrepareText output. Verify that short-text padding does not inflate NumWords.
	tok := &stubTokenizer{}

	chunks, err := PrepareChunks("Hi.", tok, 50)
	if err != nil {
		t.Fatalf("PrepareChunks error: %v", err)
	}

	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	// "Hi." is 1 word — NumWords should be 1, not 9 (which it would be if
	// the 8-space padding words were counted).
	if chunks[0].NumWords != 1 {
		t.Errorf("NumWords = %d, want 1 (raw word count of %q)", chunks[0].NumWords, "Hi.")
	}
}

func TestPrepareChunks_NumWordsMultiSentence(t *testing.T) {
	tok := &stubTokenizer{}

	chunks, err := PrepareChunks("First sentence. Second sentence.", tok, 50)
	if err != nil {
		t.Fatalf("PrepareChunks error: %v", err)
	}

	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	// "First sentence. Second sentence." joined = 4 words.
	// But after splitSentences: "First sentence." + "Second sentence." → joined = "First sentence. Second sentence." → 4 words
	if chunks[0].NumWords != 4 {
		t.Errorf("NumWords = %d, want 4", chunks[0].NumWords)
	}
}

func TestPrepareChunks_ChunkTextIsPrepared(t *testing.T) {
	// The final chunk Text should be the output of PrepareText on the joined
	// raw sentences, not raw text.
	tok := &stubTokenizer{}

	chunks, err := PrepareChunks("hello world", tok, 50)
	if err != nil {
		t.Fatalf("PrepareChunks error: %v", err)
	}

	c := chunks[0]
	// Should be capitalized, have trailing period, and be padded (2 words < 5).
	if !strings.HasPrefix(c.Text, "        ") {
		t.Errorf("chunk Text = %q, want 8-space padding for short text", c.Text)
	}

	trimmed := strings.TrimLeft(c.Text, " ")
	if trimmed[0] != 'H' {
		t.Errorf("chunk Text = %q, want capitalized after padding", c.Text)
	}

	if c.Text[len(c.Text)-1] != '.' {
		t.Errorf("chunk Text = %q, want trailing period", c.Text)
	}
}

func TestPrepareChunks_FramesAfterEOS_MatchesWordCount(t *testing.T) {
	tok := &stubTokenizer{}

	// Short text: "Hi." → 1 word ≤ 4 → FramesAfterEOS = 5
	chunks, err := PrepareChunks("Hi.", tok, 50)
	if err != nil {
		t.Fatal(err)
	}

	if chunks[0].FramesAfterEOS() != 5 {
		t.Errorf("FramesAfterEOS() = %d for 1-word chunk, want 5", chunks[0].FramesAfterEOS())
	}

	// Long text: "One two three four five." → 5 words > 4 → FramesAfterEOS = 3
	chunks, err = PrepareChunks("One two three four five.", tok, 50)
	if err != nil {
		t.Fatal(err)
	}

	if chunks[0].FramesAfterEOS() != 3 {
		t.Errorf("FramesAfterEOS() = %d for 5-word chunk, want 3", chunks[0].FramesAfterEOS())
	}

	// Exactly 4 words: "One two three four." → 4 words ≤ 4 → FramesAfterEOS = 5
	chunks, err = PrepareChunks("One two three four.", tok, 50)
	if err != nil {
		t.Fatal(err)
	}

	if chunks[0].FramesAfterEOS() != 5 {
		t.Errorf("FramesAfterEOS() = %d for 4-word chunk, want 5", chunks[0].FramesAfterEOS())
	}
}

func TestPrepareChunks_TokenCountMatchesPreparedText(t *testing.T) {
	// Verify that NumTokens matches what the tokenizer produces for the
	// final prepared chunk Text (not some intermediate representation).
	tok := &stubTokenizer{}

	chunks, err := PrepareChunks("Hello world.", tok, 50)
	if err != nil {
		t.Fatal(err)
	}

	c := chunks[0]

	directIDs, _ := tok.Encode(c.Text)
	if c.NumTokens != len(directIDs) {
		t.Errorf("NumTokens = %d, but re-encoding chunk Text gives %d tokens", c.NumTokens, len(directIDs))
	}
}
