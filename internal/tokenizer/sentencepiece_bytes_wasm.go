//go:build js && wasm

// Package tokenizer provides a filesystem-free SentencePiece implementation for
// js/wasm builds, where os.CreateTemp is not available. The algorithm (trie +
// Viterbi DP) is reproduced from the upstream
// github.com/vikesh-raj/go-sentencepiece-encoder library so that tokenisation
// results are byte-for-byte identical to those produced on native platforms.

package tokenizer

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	gosp "github.com/vikesh-raj/go-sentencepiece-encoder/sentencepiece"
	"golang.org/x/text/unicode/norm"
	"google.golang.org/protobuf/proto"
)

// NewSentencePieceTokenizerFromBytes loads a SentencePiece model from raw bytes
// without touching the filesystem. It is the js/wasm counterpart of the native
// implementation in sentencepiece_bytes.go.
func NewSentencePieceTokenizerFromBytes(data []byte) (Tokenizer, error) {
	if len(data) == 0 {
		return nil, errors.New("tokenizer model data must not be empty")
	}

	var model gosp.ModelProto
	if err := proto.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("unmarshal sentencepiece model: %w", err)
	}

	t := &spTrie{
		root:         newSpNode(),
		controlWords: make(map[string]int32),
	}

	for i, piece := range model.GetPieces() {
		switch piece.GetType() {
		case gosp.ModelProto_SentencePiece_NORMAL, gosp.ModelProto_SentencePiece_USER_DEFINED:
			t.insert(piece.GetPiece(), piece.GetScore(), int32(i))
		case gosp.ModelProto_SentencePiece_UNKNOWN:
			t.unknown = int32(i)
		case gosp.ModelProto_SentencePiece_CONTROL:
			t.controlWords[piece.GetPiece()] = int32(i)
		}
	}

	return t, nil
}

// ── trie node ────────────────────────────────────────────────────────────────

type spNode struct {
	score    float32
	index    int32
	level    int
	end      bool
	children map[rune]*spNode
}

func newSpNode() *spNode {
	return &spNode{children: make(map[rune]*spNode)}
}

// ── tokenizer ────────────────────────────────────────────────────────────────

const (
	spMinScore float32 = -math.MaxFloat32
	spSep      rune    = 0x2581 // ▁  (LOWER ONE EIGHTH BLOCK) — SentencePiece word-start marker
)

type spTrie struct {
	root         *spNode
	unknown      int32
	controlWords map[string]int32
}

// Encode implements Tokenizer.
func (t *spTrie) Encode(text string) ([]int64, error) {
	if text == "" {
		return []int64{}, nil
	}

	ids := t.tokenizeToIDs(text)
	out := make([]int64, len(ids))
	for i, id := range ids {
		out[i] = int64(id)
	}

	return out, nil
}

func (t *spTrie) insert(word string, score float32, index int32) {
	_, size := utf8.DecodeLastRuneInString(word)
	charCount := len(word)
	node := t.root

	for i, r := range word {
		child, ok := node.children[r]
		if !ok {
			child = newSpNode()
			child.level = node.level + 1
		}

		if i == charCount-size {
			child.end = true
			child.score = score
			child.index = index
		}

		node.children[r] = child
		node = child
	}
}

func (t *spTrie) tokenizeToIDs(text string) []int32 {
	text = spNormalize(text)
	runes := spToRunes(text)
	spReplaceWhitespace(runes)
	slices := t.viterbiForward(runes)
	slices = t.viterbiBackward(slices)

	ids := make([]int32, 0, len(slices))
	prevUnknown := false

	for _, s := range slices {
		if prevUnknown && s.spIdx == t.unknown {
			// merge consecutive unknowns (same behaviour as upstream)
		} else {
			ids = append(ids, s.spIdx)
		}

		prevUnknown = s.spIdx == t.unknown
	}

	return ids
}

type spSlice struct {
	score float32
	spIdx int32
	start int
	end   int
}

func (t *spTrie) commonPrefixSearch(runes []rune) []*spNode {
	var out []*spNode

	node := t.root
	for _, r := range runes {
		child, ok := node.children[r]
		if !ok {
			break
		}

		if child.end {
			out = append(out, child)
		}

		node = child
	}

	return out
}

func (t *spTrie) viterbiForward(runes []rune) []spSlice {
	n := len(runes) + 1
	scores := make([]float32, n)
	slices := make([]spSlice, n)

	for i := range scores {
		scores[i] = spMinScore
		slices[i].start = -1
		slices[i].spIdx = t.unknown
	}

	scores[0] = 0.0

	for i := range runes {
		matches := t.commonPrefixSearch(runes[i:])
		for _, node := range matches {
			localScore := scores[i] + node.score
			end := i + node.level
			if localScore > scores[end] {
				slices[end] = spSlice{score: localScore, spIdx: node.index, start: i, end: end}
				scores[end] = localScore
			}
		}

		if scores[i+1] <= spMinScore {
			slices[i+1] = spSlice{score: spMinScore, spIdx: t.unknown, start: i, end: i + 1}
			scores[i+1] = 0.0
		}
	}

	return slices
}

func (t *spTrie) viterbiBackward(slices []spSlice) []spSlice {
	last := len(slices) - 1
	best := make([]spSlice, len(slices))
	i := last
	idx := last

	for ; i >= 0; i-- {
		s := slices[idx]
		if s.start == -1 {
			i++
			break
		}

		best[i] = s
		idx = s.start
	}

	return best[i : last+1]
}

// ── normalization (mirrors upstream normalize.go) ─────────────────────────

var spControlChars = []rune{
	0x007F, 0x00AD, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604, 0x0605, 0x061C, 0x06DD, 0x070F,
	0x08E2, 0x180E, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D,
	0x202E, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0x2066, 0x2067, 0x2068, 0x2069, 0x206A,
	0x206B, 0x206C, 0x206D, 0x206E, 0x206F, 0xFEFF, 0xFFF9, 0xFFFA, 0xFFFB, 0x110BD,
	0x110CD, 0x13430, 0x13431, 0x13432, 0x13433, 0x13434, 0x13435, 0x13436, 0x13437,
	0x13438, 0x1BCA0, 0x1BCA1, 0x1BCA2, 0x1BCA3, 0x1D173, 0x1D174, 0x1D175, 0x1D176,
	0x1D177, 0x1D178, 0x1D179, 0x1D17A, 0xE0001,
}

func spIsControlChar(c rune) bool {
	return slices.Contains(spControlChars, c)
}

func spIsControl(c rune) bool {
	if c == ' ' || c == '\n' || c == '\r' || c == '\t' {
		return false
	}

	if c <= 0x001F ||
		(c >= 0x0080 && c <= 0x009F) ||
		(c >= 0xE0020 && c <= 0xE007F) ||
		(c >= 0xE000 && c <= 0xF8FF) ||
		(c >= 0xF0000 && c <= 0xFFFFD) ||
		(c >= 0x100000 && c <= 0x10FFFD) ||
		(c >= 0xD800 && c <= 0xDB7F) ||
		(c >= 0xDB80 && c <= 0xDBFF) ||
		(c >= 0xDC00 && c <= 0xDFFF) ||
		spIsControlChar(c) {
		return true
	}

	return false
}

func spNormalize(s string) string {
	mapped := strings.Map(func(r rune) rune {
		if spIsControl(r) || r == 0 {
			return -1
		}

		if unicode.IsSpace(r) {
			return ' '
		}

		return r
	}, s)

	return norm.NFKC.String(mapped)
}

func spToRunes(text string) []rune {
	runes := make([]rune, 0, len(text)+1)

	first, _ := utf8.DecodeRuneInString(text)
	if first != spSep {
		runes = append(runes, spSep)
	}

	for _, r := range text {
		runes = append(runes, r)
	}

	return runes
}

func spReplaceWhitespace(runes []rune) {
	for i, r := range runes {
		if unicode.IsSpace(r) {
			runes[i] = spSep
		}
	}
}
