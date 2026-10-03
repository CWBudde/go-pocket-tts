package tokenizer

// Filesystem-free SentencePiece UNIGRAM encoder shared by all platforms. It is
// a port of sentencepiece's Normalizer (without precompiled charsmap, i.e. the
// "identity" normalizer the shipped models use), unigram Model::EncodeOptimized
// (Viterbi over a trie of the vocab) and SentencePieceProcessor's byte
// fallback, so ids match Python sentencepiece exactly (golden-tested).

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

// errEmptyModelData is returned when a model is loaded from zero bytes.
var errEmptyModelData = errors.New("tokenizer model data must not be empty")

var errUnsupportedModel = errors.New("unsupported sentencepiece model")

const (
	spSep rune = 0x2581 // ▁ (LOWER ONE EIGHTH BLOCK) — SentencePiece word-start marker

	// spUnkPenalty is sentencepiece's kUnkPenalty: an unknown character
	// scores this much below the worst vocab piece.
	spUnkPenalty float32 = 10

	// spFltMin is C's FLT_MIN, sentencepiece's initial max_score.
	spFltMin float32 = 0x1p-126
)

// ── trie ─────────────────────────────────────────────────────────────────────

type spNode struct {
	id       int32
	score    float32
	end      bool
	children map[rune]*spNode
}

func newSpNode() *spNode {
	return &spNode{children: make(map[rune]*spNode)}
}

func (n *spNode) insert(piece string, id int32, score float32) {
	node := n

	for _, r := range piece {
		child, ok := node.children[r]
		if !ok {
			child = newSpNode()
			node.children[r] = child
		}

		node = child
	}

	node.end = true
	node.id = id
	node.score = score
}

// ── model ────────────────────────────────────────────────────────────────────

// spModel is a SentencePiece UNIGRAM model ready for encoding.
type spModel struct {
	root     *spNode
	unkID    int32
	unkScore float32

	// byteIDs maps a byte to the id of its <0xXX> piece; only set with
	// byteFallback.
	byteFallback bool
	byteIDs      [256]int32

	addDummyPrefix         bool
	removeExtraWhitespaces bool
	escapeWhitespaces      bool
}

// newSpModel builds the encoder from a serialized SentencePiece ModelProto.
func newSpModel(data []byte) (*spModel, error) {
	if len(data) == 0 {
		return nil, errEmptyModelData
	}

	proto, err := parseSpModelProto(data)
	if err != nil {
		return nil, err
	}

	err = proto.checkSupported()
	if err != nil {
		return nil, err
	}

	m := &spModel{
		root:                   newSpNode(),
		byteFallback:           proto.byteFallback,
		addDummyPrefix:         proto.addDummyPrefix,
		removeExtraWhitespaces: proto.removeExtraWhitespaces,
		escapeWhitespaces:      proto.escapeWhitespaces,
	}

	err = m.buildVocab(proto.pieces)
	if err != nil {
		return nil, err
	}

	return m, nil
}

func (p *spModelProto) checkSupported() error {
	if p.modelType != spModelTypeUnigram {
		return fmt.Errorf("%w: model type %d (only UNIGRAM is supported)", errUnsupportedModel, p.modelType)
	}

	if len(p.precompiledCharsmap) > 0 {
		return fmt.Errorf("%w: normalizer %q uses a precompiled charsmap", errUnsupportedModel, p.normalizerName)
	}

	if p.treatWhitespaceAsSuffix {
		return fmt.Errorf("%w: treat_whitespace_as_suffix is set", errUnsupportedModel)
	}

	return nil
}

// buildVocab fills the trie with the NORMAL and USER_DEFINED pieces and
// resolves the unknown and byte pieces. CONTROL and UNUSED pieces never take
// part in encoding.
func (m *spModel) buildVocab(pieces []spPieceProto) error {
	minScore, maxScore := float32(math.MaxFloat32), spFltMin
	byteByName := make(map[string]int32)
	unknowns := 0

	for _, p := range pieces {
		if p.typ == spNormal {
			minScore = min(minScore, p.score)
			maxScore = max(maxScore, p.score)
		}
	}

	for i, p := range pieces {
		id := int32(i)

		switch p.typ {
		case spNormal:
			m.root.insert(p.piece, id, p.score)
		case spUserDefined:
			// sentencepiece gives user-defined symbols a bonus so they are
			// always chosen: length in bytes times max_score, minus 0.1.
			m.root.insert(p.piece, id, float32(len(p.piece))*maxScore-0.1)
		case spUnknown:
			m.unkID = id
			unknowns++
		case spByte:
			byteByName[p.piece] = id
		}
	}

	if unknowns != 1 {
		return fmt.Errorf("%w: need exactly one UNKNOWN piece, found %d", errUnsupportedModel, unknowns)
	}

	m.unkScore = minScore - spUnkPenalty

	if m.byteFallback {
		for b := range m.byteIDs {
			name := fmt.Sprintf("<0x%02X>", b)

			id, ok := byteByName[name]
			if !ok {
				return fmt.Errorf("%w: byte_fallback is set but byte piece %s is missing", errUnsupportedModel, name)
			}

			m.byteIDs[b] = id
		}
	}

	return nil
}

// ── normalization ────────────────────────────────────────────────────────────

// normalize mirrors sentencepiece's Normalizer::Normalize without a charsmap:
// characters are copied as they are (invalid UTF-8 bytes become U+FFFD), only
// ' ' (U+0020) counts as whitespace.
func (m *spModel) normalize(text string) []rune {
	src := []rune(text)

	if m.removeExtraWhitespaces {
		for len(src) > 0 && src[0] == ' ' {
			src = src[1:]
		}
	}

	if len(src) == 0 {
		return nil
	}

	space := ' '
	if m.escapeWhitespaces {
		space = spSep
	}

	out := make([]rune, 0, len(src)+1)
	if m.addDummyPrefix {
		out = append(out, space)
	}

	prevSpace := false

	for _, r := range src {
		if r != ' ' {
			out = append(out, r)
			prevSpace = false

			continue
		}

		if !prevSpace {
			out = append(out, space)
		}

		prevSpace = m.removeExtraWhitespaces
	}

	if m.removeExtraWhitespaces {
		for len(out) > 0 && out[len(out)-1] == space {
			out = out[:len(out)-1]
		}
	}

	return out
}

// ── encoding ─────────────────────────────────────────────────────────────────

// spBestPath is the best path ending at a rune position: its score and its
// last piece, which starts at start (-1: no path yet).
type spBestPath struct {
	id    int32
	score float32
	start int
}

func (b *spBestPath) relax(id int32, score float32, start int) {
	if b.start == -1 || score > b.score {
		*b = spBestPath{id: id, score: score, start: start}
	}
}

// viterbi returns the best path ending at every rune position, following
// unigram Model::EncodeOptimized: vocab pieces matching at a position are
// tried shortest first, ties keep the earlier candidate, and a position
// without a one-character vocab piece gets a one-character unknown piece.
func (m *spModel) viterbi(runes []rune) []spBestPath {
	best := make([]spBestPath, len(runes)+1)
	for i := range best {
		best[i].start = -1
	}

	best[0].start = 0

	for i := range runes {
		base := best[i].score
		hasSingle := false
		node := m.root

		for j := i; j < len(runes); j++ {
			node = node.children[runes[j]]
			if node == nil {
				break
			}

			if node.end {
				best[j+1].relax(node.id, base+node.score, i)
				hasSingle = hasSingle || j == i
			}
		}

		if !hasSingle {
			best[i+1].relax(m.unkID, base+m.unkScore, i)
		}
	}

	return best
}

// pieces encodes text. Piece texts are the normalized text they cover; with
// byte fallback, an unknown character becomes one piece per UTF-8 byte, all
// with an empty text except the last, which carries the character.
func (m *spModel) pieces(text string) []Piece {
	runes := m.normalize(text)
	if len(runes) == 0 {
		return []Piece{}
	}

	best := m.viterbi(runes)

	// Backtrack the piece boundaries from the end.
	var ends []int
	for end := len(runes); end > 0; end = best[end].start {
		ends = append(ends, end)
	}

	out := make([]Piece, 0, len(ends))

	for _, end := range slices.Backward(ends) {
		b := best[end]
		surface := string(runes[b.start:end])

		if b.id != m.unkID || !m.byteFallback {
			out = append(out, Piece{ID: int64(b.id), Text: surface})

			continue
		}

		for k := range len(surface) {
			p := Piece{ID: int64(m.byteIDs[surface[k]])}
			if k == len(surface)-1 {
				p.Text = surface
			}

			out = append(out, p)
		}
	}

	return out
}
