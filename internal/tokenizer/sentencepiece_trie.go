package tokenizer

// Filesystem-free SentencePiece UNIGRAM encoder shared by all platforms. The
// algorithm (trie + Viterbi DP) and the normalisation are reproduced from
// github.com/vikesh-raj/go-sentencepiece-encoder so token IDs are identical to
// that library (cross-checked in tests). Unlike the library, piece surfaces
// of consecutive unknown characters are kept in full: the library extends a
// copy of the previous token when merging unknowns and so drops all but the
// first unknown rune of a run.

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

// errEmptyModelData is returned when a model is loaded from zero bytes.
var errEmptyModelData = errors.New("tokenizer model data must not be empty")

const (
	spMinScore float32 = -math.MaxFloat32
	spSep      rune    = 0x2581 // ▁ (LOWER ONE EIGHTH BLOCK) — SentencePiece word-start marker
)

// ── trie ─────────────────────────────────────────────────────────────────────

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

type spTrie struct {
	root    *spNode
	unknown int32
}

// newSpTrie builds the vocabulary trie from a serialized SentencePiece
// ModelProto. Control pieces (<s>, </s>, ...) never take part in encoding.
func newSpTrie(data []byte) (*spTrie, error) {
	if len(data) == 0 {
		return nil, errEmptyModelData
	}

	var model gosp.ModelProto

	err := proto.Unmarshal(data, &model)
	if err != nil {
		return nil, fmt.Errorf("unmarshal sentencepiece model: %w", err)
	}

	t := &spTrie{root: newSpNode()}

	for i, piece := range model.GetPieces() {
		switch piece.GetType() {
		case gosp.ModelProto_SentencePiece_NORMAL, gosp.ModelProto_SentencePiece_USER_DEFINED:
			t.insert(piece.GetPiece(), piece.GetScore(), int32(i))
		case gosp.ModelProto_SentencePiece_UNKNOWN:
			t.unknown = int32(i)
		}
	}

	return t, nil
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

// ── encoding ─────────────────────────────────────────────────────────────────

// spSpan is a token as a half-open rune range of the normalized rune array.
type spSpan struct {
	id    int32
	start int
	end   int
}

// pieces encodes text into pieces. Consecutive unknown slices are merged
// into one unknown piece covering the whole run (same IDs as upstream).
func (t *spTrie) pieces(text string) []Piece {
	runes := spToRunes(spNormalize(text))
	spReplaceWhitespace(runes)
	best := t.viterbiBackward(t.viterbiForward(runes))

	spans := make([]spSpan, 0, len(best))
	prevUnknown := false

	for _, s := range best {
		isUnknown := s.spIdx == t.unknown
		if prevUnknown && isUnknown {
			spans[len(spans)-1].end = s.end
		} else {
			spans = append(spans, spSpan{id: s.spIdx, start: s.start, end: s.end})
		}

		prevUnknown = isUnknown
	}

	out := make([]Piece, len(spans))
	for i, sp := range spans {
		out[i] = Piece{ID: int64(sp.id), Text: string(runes[sp.start:sp.end])}
	}

	return out
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
	lattice := make([]spSlice, n)

	for i := range scores {
		scores[i] = spMinScore
		lattice[i].start = -1
		lattice[i].spIdx = t.unknown
	}

	scores[0] = 0.0

	for i := range runes {
		for _, node := range t.commonPrefixSearch(runes[i:]) {
			localScore := scores[i] + node.score

			end := i + node.level
			if localScore > scores[end] {
				lattice[end] = spSlice{score: localScore, spIdx: node.index, start: i, end: end}
				scores[end] = localScore
			}
		}

		if scores[i+1] <= spMinScore {
			lattice[i+1] = spSlice{score: spMinScore, spIdx: t.unknown, start: i, end: i + 1}
			scores[i+1] = 0.0
		}
	}

	return lattice
}

func (t *spTrie) viterbiBackward(lattice []spSlice) []spSlice {
	last := len(lattice) - 1
	best := make([]spSlice, len(lattice))
	i := last
	idx := last

	for ; i >= 0; i-- {
		s := lattice[idx]
		if s.start == -1 {
			i++

			break
		}

		best[i] = s
		idx = s.start
	}

	return best[i : last+1]
}

// ── normalization (mirrors upstream normalize.go) ───────────────────────────

var spControlChars = []rune{
	0x007F, 0x00AD, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604, 0x0605, 0x061C, 0x06DD, 0x070F,
	0x08E2, 0x180E, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D,
	0x202E, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0x2066, 0x2067, 0x2068, 0x2069, 0x206A,
	0x206B, 0x206C, 0x206D, 0x206E, 0x206F, 0xFEFF, 0xFFF9, 0xFFFA, 0xFFFB, 0x110BD,
	0x110CD, 0x13430, 0x13431, 0x13432, 0x13433, 0x13434, 0x13435, 0x13436, 0x13437,
	0x13438, 0x1BCA0, 0x1BCA1, 0x1BCA2, 0x1BCA3, 0x1D173, 0x1D174, 0x1D175, 0x1D176,
	0x1D177, 0x1D178, 0x1D179, 0x1D17A, 0xE0001,
}

// spControlRanges are the inclusive control ranges of upstream isControl
// (its three adjacent surrogate ranges are merged into one).
var spControlRanges = [][2]rune{
	{0x0000, 0x001F},
	{0x0080, 0x009F},
	{0xD800, 0xDFFF},
	{0xE000, 0xF8FF},
	{0xE0020, 0xE007F},
	{0xF0000, 0xFFFFD},
	{0x100000, 0x10FFFD},
}

func spIsControl(c rune) bool {
	if c == ' ' || c == '\n' || c == '\r' || c == '\t' {
		return false
	}

	for _, r := range spControlRanges {
		if c >= r[0] && c <= r[1] {
			return true
		}
	}

	return slices.Contains(spControlChars, c)
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

// spToRunes converts text to runes and prepends the dummy-prefix '▁' unless
// the text already starts with one.
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
