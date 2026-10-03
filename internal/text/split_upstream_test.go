package text

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/tokenizer"
)

// splitGolden is a testdata/split_upstream*.json file: upstream
// split_into_best_sentences run with Python sentencepiece on Model.
type splitGolden struct {
	Model           string `json:"model"`
	TokenizerSHA256 string `json:"tokenizer_sha256"`
	Cases           []struct {
		Name      string   `json:"name"`
		Text      string   `json:"text"`
		MaxTokens int      `json:"max_tokens"`
		Chunks    []string `json:"chunks"`
	} `json:"cases"`
}

// realTokenizer loads model (relative to the repo root) and checks it is the
// model the goldens were generated with; it skips when absent.
func realTokenizer(t *testing.T, model, wantSHA256 string) *tokenizer.SentencePieceTokenizer {
	t.Helper()

	path, err := filepath.Abs(filepath.Join("..", "..", filepath.FromSlash(model)))
	if err != nil {
		t.Fatalf("abs path: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("%s not available: %v", model, err)
	}

	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != wantSHA256 {
		t.Skipf("%s sha256 %s; goldens were generated with %s", model, got, wantSHA256)
	}

	tok, err := tokenizer.NewSentencePieceTokenizer(path)
	if err != nil {
		t.Fatalf("NewSentencePieceTokenizer: %v", err)
	}

	return tok
}

// Upstream tests/test_split_sentences.py texts (plus a few extras, and German
// text on the German tokenizer) through the real Python splitter. Options match
// those tests: no padding, no semicolon removal, upstream defaults otherwise.
func TestSplitIntoBestSentences_UpstreamGolden(t *testing.T) {
	for _, file := range []string{"split_upstream.json", "split_upstream_german.json"} {
		t.Run(file, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("testdata", file))
			if err != nil {
				t.Fatal(err)
			}

			var golden splitGolden

			err = json.Unmarshal(data, &golden)
			if err != nil {
				t.Fatalf("parse goldens: %v", err)
			}

			tok := realTokenizer(t, golden.Model, golden.TokenizerSHA256)

			for _, tc := range golden.Cases {
				t.Run(tc.Name, func(t *testing.T) {
					got, err := splitIntoBestSentences(tok, tc.Text, tc.MaxTokens, upstreamOptions())
					if err != nil {
						t.Fatalf("splitIntoBestSentences: %v", err)
					}

					if !slices.Equal(got, tc.Chunks) {
						t.Errorf("chunks (max %d)\n got: %q\nwant: %q", tc.MaxTokens, got, tc.Chunks)
					}
				})
			}
		})
	}
}
