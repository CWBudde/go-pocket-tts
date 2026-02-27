package tokenizer

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// modelPath returns the path to the real tokenizer model, skipping if absent.
func modelPath(t *testing.T) string {
	t.Helper()
	// Walk up from the package dir to find models/tokenizer.model.
	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatalf("abs path: %v", err)
	}

	for {
		candidate := filepath.Join(dir, "models", "tokenizer.model")

		_, err = os.Stat(candidate)
		if err == nil {
			return candidate
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}

		dir = parent
	}

	t.Skip("models/tokenizer.model not found; skipping tokenizer tests")

	return ""
}

// ---------------------------------------------------------------------------
// NewSentencePieceTokenizer
// ---------------------------------------------------------------------------

func TestNewSentencePieceTokenizer_ValidModel(t *testing.T) {
	path := modelPath(t)

	tok, err := NewSentencePieceTokenizer(path)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer(%q): %v", path, err)
	}

	if tok == nil {
		t.Fatal("expected non-nil tokenizer")
	}
}

func TestNewSentencePieceTokenizer_MissingFile(t *testing.T) {
	_, err := NewSentencePieceTokenizer("/nonexistent/tokenizer.model")
	if err == nil {
		t.Fatal("expected error for missing model file")
	}
}

func TestNewSentencePieceTokenizer_EmptyPath(t *testing.T) {
	_, err := NewSentencePieceTokenizer("")
	if err == nil {
		t.Fatal("expected error for empty path")
	}

	if !errors.Is(err, ErrEmptyPath) {
		t.Errorf("expected ErrEmptyPath, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Encode — correctness against Python reference output
//
// Ground truth from: python3 -c "import sentencepiece as spm; sp = spm.SentencePieceProcessor();
//   sp.Load('models/tokenizer.model'); print(sp.Encode(text, out_type=int))"
// ---------------------------------------------------------------------------

func TestEncode_Hello(t *testing.T) {
	path := modelPath(t)

	tok, err := NewSentencePieceTokenizer(path)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer: %v", err)
	}

	// Python reference: sp.Encode("hello") -> [1876, 393]
	got, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	want := []int64{1876, 393}
	if !equalInt64(got, want) {
		t.Errorf("Encode(%q) = %v, want %v", "hello", got, want)
	}
}

func TestEncode_HelloWorldDot(t *testing.T) {
	path := modelPath(t)

	tok, err := NewSentencePieceTokenizer(path)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer: %v", err)
	}

	// Python reference: sp.Encode("Hello world.") -> [2994, 578, 263]
	got, err := tok.Encode("Hello world.")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	want := []int64{2994, 578, 263}
	if !equalInt64(got, want) {
		t.Errorf("Encode(%q) = %v, want %v", "Hello world.", got, want)
	}
}

func TestEncode_WithLeadingSpaces(t *testing.T) {
	path := modelPath(t)

	tok, err := NewSentencePieceTokenizer(path)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer: %v", err)
	}

	// Python reference: sp.Encode("        hello") -> [260, 260, 260, 260, 260, 260, 260, 260, 1876, 393]
	// This is the 8-space padding used by PocketTTS text preprocessing.
	got, err := tok.Encode("        hello")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	want := []int64{260, 260, 260, 260, 260, 260, 260, 260, 1876, 393}
	if !equalInt64(got, want) {
		t.Errorf("Encode(%q) = %v, want %v", "        hello", got, want)
	}
}

func TestEncode_ReturnsNonEmpty(t *testing.T) {
	path := modelPath(t)

	tok, err := NewSentencePieceTokenizer(path)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer: %v", err)
	}

	got, err := tok.Encode("Test sentence.")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	// Python reference: [602, 552, 1472, 599, 263]
	want := []int64{602, 552, 1472, 599, 263}
	if !equalInt64(got, want) {
		t.Errorf("Encode(%q) = %v, want %v", "Test sentence.", got, want)
	}
}

func TestEncode_EmptyString(t *testing.T) {
	path := modelPath(t)

	tok, err := NewSentencePieceTokenizer(path)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer: %v", err)
	}

	got, err := tok.Encode("")
	if err != nil {
		t.Fatalf("Encode(\"\") should not error: %v", err)
	}

	if len(got) != 0 {
		t.Errorf("Encode(\"\") = %v, want empty slice", got)
	}
}

func TestEncode_TokenIDsInRange(t *testing.T) {
	path := modelPath(t)

	tok, err := NewSentencePieceTokenizer(path)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer: %v", err)
	}

	// All token IDs must be in [0, 3999] for vocab size 4000.
	ids, err := tok.Encode("The quick brown fox jumps over the lazy dog.")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	if len(ids) == 0 {
		t.Fatal("Encode returned empty result")
	}

	for i, id := range ids {
		if id < 0 || id >= 4000 {
			t.Errorf("token[%d] = %d out of vocab range [0, 4000)", i, id)
		}
	}
}

func TestEncode_ImplementsInterface(t *testing.T) {
	path := modelPath(t)

	tok, err := NewSentencePieceTokenizer(path)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer: %v", err)
	}
	// Verify SentencePieceTokenizer implements Tokenizer interface.
	var _ Tokenizer = tok
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func equalInt64(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}
