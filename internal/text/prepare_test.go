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
	// 3 tokens → ceil(3/3 + 2) × 12.5 = ceil(3) × 12.5 = 3 × 12.5 = 37.5
	c := ChunkMetadata{NumTokens: 3}
	if got, want := c.MaxFrames(), 37.5; got != want {
		t.Errorf("MaxFrames() = %v, want %v", got, want)
	}
}

func TestChunkMetadata_MaxFrames_NonDivisible(t *testing.T) {
	// 4 tokens → ceil(4/3 + 2) × 12.5 = ceil(3.333) × 12.5 = 4 × 12.5 = 50
	c := ChunkMetadata{NumTokens: 4}
	if got, want := c.MaxFrames(), 50.0; got != want {
		t.Errorf("MaxFrames() = %v, want %v", got, want)
	}
}

func TestChunkMetadata_FramesAfterEOS_ShortInput(t *testing.T) {
	// ≤ 4 words → 3
	if got := (ChunkMetadata{NumWords: 4}).FramesAfterEOS(); got != 3 {
		t.Errorf("FramesAfterEOS() = %d, want 3", got)
	}
	if got := (ChunkMetadata{NumWords: 1}).FramesAfterEOS(); got != 3 {
		t.Errorf("FramesAfterEOS() = %d, want 3", got)
	}
}

func TestChunkMetadata_FramesAfterEOS_LongInput(t *testing.T) {
	// > 4 words → 1
	if got := (ChunkMetadata{NumWords: 5}).FramesAfterEOS(); got != 1 {
		t.Errorf("FramesAfterEOS() = %d, want 1", got)
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
