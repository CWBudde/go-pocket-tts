package tokenizer

// Loader for Hugging Face tokenizers' tokenizer.json with a Unigram model, the
// format the SentencePiece models are converted to and upstream configs pin
// (every shipped language also has the .model sibling). Encoding follows
// tokenizers' TokenizerImpl::encode for this pipeline (ported from tokenizers
// 0.23.2 and golden-tested against it):
//
//  1. Added tokens (normalized=false) are cut out of the raw input, exact and
//     case-sensitive, leftmost-longest (AddedVocabulary::find_matches).
//  2. Every other non-empty segment is normalized on its own: the Prepend("▁")
//     normalizer, then the Metaspace pre-tokenizer replaces ' ' by '▁',
//     applies its prepend_scheme and, with split, starts a new word at every
//     '▁' (SplitDelimiterBehavior::MergedWithNext).
//  3. Each word runs through the shared Viterbi (Unigram::encode_optimized):
//     every vocab entry is in the trie, <0xXX> and special pieces included;
//     scores are float64; the unknown score is the minimum score of the whole
//     vocab minus 10.
//  4. Consecutive unknown pieces fuse into one token (fuse_unk), which
//     Unigram::tokenize looks up in the vocab, else byte-fallback encodes,
//     else maps to unk_id.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

// hfReplacement is the only Prepend/Metaspace string supported: SentencePiece's
// word-start marker.
const hfReplacement = string(spSep)

type hfPrependScheme int

const (
	hfPrependAlways hfPrependScheme = iota
	hfPrependFirst
	hfPrependNever
)

// hfSpecial is an added token extracted from the raw input.
type hfSpecial struct {
	content string
	id      int32
}

// hfConfig holds the tokenizers pipeline around the Unigram model.
type hfConfig struct {
	// specials are the added tokens, matched leftmost-longest.
	specials []hfSpecial
	// prepend is set by a Prepend("▁") normalizer.
	prepend       bool
	prependScheme hfPrependScheme
	// split makes every '▁' start a new word (Metaspace split).
	split bool
	// vocab maps a piece to its id; for duplicate pieces the last id wins,
	// as in tokenizers.
	vocab map[string]int32
}

// ── JSON schema ──────────────────────────────────────────────────────────────

type hfTokenizerJSON struct {
	Truncation    json.RawMessage    `json:"truncation"`
	Padding       json.RawMessage    `json:"padding"`
	AddedTokens   []hfAddedTokenJSON `json:"added_tokens"`
	Normalizer    json.RawMessage    `json:"normalizer"`
	PreTokenizer  json.RawMessage    `json:"pre_tokenizer"`
	PostProcessor json.RawMessage    `json:"post_processor"`
	Model         *hfModelJSON       `json:"model"`
}

type hfAddedTokenJSON struct {
	Content    string `json:"content"`
	SingleWord bool   `json:"single_word"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
	Normalized *bool  `json:"normalized"`
}

type hfNormalizerJSON struct {
	Type        string            `json:"type"`
	Prepend     string            `json:"prepend"`
	Normalizers []json.RawMessage `json:"normalizers"`
}

type hfPreTokenizerJSON struct {
	Type           string  `json:"type"`
	Replacement    string  `json:"replacement"`
	PrependScheme  *string `json:"prepend_scheme"`
	Split          *bool   `json:"split"`
	AddPrefixSpace *bool   `json:"add_prefix_space"`
}

type hfModelJSON struct {
	Type         string         `json:"type"`
	UnkID        *int           `json:"unk_id"`
	Vocab        []hfVocabEntry `json:"vocab"`
	ByteFallback bool           `json:"byte_fallback"`
}

// hfVocabEntry is one [piece, score] pair of model.vocab.
type hfVocabEntry struct {
	piece string
	score float64
}

// UnmarshalJSON decodes a [piece, score] pair.
func (e *hfVocabEntry) UnmarshalJSON(data []byte) error {
	var pair []json.RawMessage

	err := json.Unmarshal(data, &pair)
	if err != nil {
		return fmt.Errorf("vocab entry: %w", err)
	}

	if len(pair) != 2 {
		return fmt.Errorf("vocab entry has %d elements, want [piece, score]", len(pair))
	}

	if isNullJSON(pair[0]) || isNullJSON(pair[1]) {
		return errors.New("vocab entry has a null piece or score")
	}

	err = json.Unmarshal(pair[0], &e.piece)
	if err != nil {
		return fmt.Errorf("vocab entry piece: %w", err)
	}

	err = json.Unmarshal(pair[1], &e.score)
	if err != nil {
		return fmt.Errorf("vocab entry score: %w", err)
	}

	return nil
}

func isNullJSON(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)

	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null"))
}

// ── loading ──────────────────────────────────────────────────────────────────

// NewJSONTokenizerFromBytes loads a Hugging Face tokenizer.json (Unigram
// model) from raw bytes without touching the filesystem, so it also works in
// js/wasm builds.
func NewJSONTokenizerFromBytes(data []byte) (Tokenizer, error) {
	model, err := newSpModelFromJSON(data)
	if err != nil {
		return nil, err
	}

	return &SentencePieceTokenizer{model: model}, nil
}

// newSpModelFromJSON builds the encoder from a tokenizer.json document.
func newSpModelFromJSON(data []byte) (*spModel, error) {
	if len(data) == 0 {
		return nil, errEmptyModelData
	}

	var doc hfTokenizerJSON

	err := json.Unmarshal(data, &doc)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}

	model, err := doc.checkModel()
	if err != nil {
		return nil, err
	}

	hf, err := doc.pipeline()
	if err != nil {
		return nil, err
	}

	m := &spModel{
		root:         newSpNode(),
		unkID:        int32(*model.UnkID),
		byteFallback: model.ByteFallback,
		hf:           hf,
	}

	err = m.buildJSONVocab(model.Vocab)
	if err != nil {
		return nil, err
	}

	hf.specials, err = doc.specials(hf.vocab)
	if err != nil {
		return nil, err
	}

	return m, nil
}

// checkModel validates the Unigram model section.
func (doc *hfTokenizerJSON) checkModel() (*hfModelJSON, error) {
	model := doc.Model
	if model == nil {
		return nil, fmt.Errorf("%w: tokenizer.json has no model", errUnsupportedModel)
	}

	if model.Type != "Unigram" {
		return nil, fmt.Errorf("%w: model type %q (only Unigram is supported)", errUnsupportedModel, model.Type)
	}

	if model.Vocab == nil {
		return nil, fmt.Errorf("%w: Unigram model has no vocab", errUnsupportedModel)
	}

	if len(model.Vocab) > math.MaxInt32 {
		return nil, fmt.Errorf("%w: vocab has %d entries", errUnsupportedModel, len(model.Vocab))
	}

	if model.UnkID == nil {
		return nil, fmt.Errorf("%w: unk_id is null", errUnsupportedModel)
	}

	if *model.UnkID < 0 || *model.UnkID >= len(model.Vocab) {
		return nil, fmt.Errorf("%w: unk_id %d is outside the vocab of %d entries", errUnsupportedModel, *model.UnkID, len(model.Vocab))
	}

	return model, nil
}

// pipeline validates everything around the model and returns its settings.
func (doc *hfTokenizerJSON) pipeline() (*hfConfig, error) {
	for _, f := range []struct {
		name string
		raw  json.RawMessage
	}{
		{"truncation", doc.Truncation},
		{"padding", doc.Padding},
		{"post_processor", doc.PostProcessor},
	} {
		if !isNullJSON(f.raw) {
			return nil, fmt.Errorf("%w: %s is set (only null is supported)", errUnsupportedModel, f.name)
		}
	}

	prepends, err := hfPrependCount(doc.Normalizer)
	if err != nil {
		return nil, err
	}

	if prepends > 1 {
		return nil, fmt.Errorf("%w: %d Prepend normalizers (at most one is supported)", errUnsupportedModel, prepends)
	}

	hf := &hfConfig{prepend: prepends == 1}

	err = hf.parsePreTokenizer(doc.PreTokenizer)
	if err != nil {
		return nil, err
	}

	return hf, nil
}

// hfPrependCount returns how many Prepend("▁") normalizers raw holds, which
// may be nested in Sequences, and rejects every other normalizer.
func hfPrependCount(raw json.RawMessage) (int, error) {
	if isNullJSON(raw) {
		return 0, nil
	}

	var n hfNormalizerJSON

	err := json.Unmarshal(raw, &n)
	if err != nil {
		return 0, fmt.Errorf("parse tokenizer.json normalizer: %w", err)
	}

	switch n.Type {
	case "Prepend":
		if n.Prepend != hfReplacement {
			return 0, fmt.Errorf("%w: Prepend normalizer %q (only %q is supported)", errUnsupportedModel, n.Prepend, hfReplacement)
		}

		return 1, nil
	case "Sequence":
		total := 0

		for _, sub := range n.Normalizers {
			count, err := hfPrependCount(sub)
			if err != nil {
				return 0, err
			}

			total += count
		}

		return total, nil
	default:
		return 0, fmt.Errorf("%w: normalizer %q (only Prepend %q is supported)", errUnsupportedModel, n.Type, hfReplacement)
	}
}

// parsePreTokenizer accepts a Metaspace pre-tokenizer with replacement '▁',
// with tokenizers' defaults: prepend_scheme "always" (legacy
// add_prefix_space=false means "never") and split true.
func (hf *hfConfig) parsePreTokenizer(raw json.RawMessage) error {
	if isNullJSON(raw) {
		return fmt.Errorf("%w: pre_tokenizer is missing (only Metaspace is supported)", errUnsupportedModel)
	}

	var p hfPreTokenizerJSON

	err := json.Unmarshal(raw, &p)
	if err != nil {
		return fmt.Errorf("parse tokenizer.json pre_tokenizer: %w", err)
	}

	if p.Type != "Metaspace" {
		return fmt.Errorf("%w: pre_tokenizer %q (only Metaspace is supported)", errUnsupportedModel, p.Type)
	}

	if p.Replacement != hfReplacement {
		return fmt.Errorf("%w: Metaspace replacement %q (only %q is supported)", errUnsupportedModel, p.Replacement, hfReplacement)
	}

	scheme := "always"
	if p.PrependScheme != nil {
		scheme = *p.PrependScheme
	}

	switch scheme {
	case "always":
		hf.prependScheme = hfPrependAlways
	case "first":
		hf.prependScheme = hfPrependFirst
	case "never":
		hf.prependScheme = hfPrependNever
	default:
		return fmt.Errorf("%w: Metaspace prepend_scheme %q", errUnsupportedModel, scheme)
	}

	if p.AddPrefixSpace != nil && !*p.AddPrefixSpace {
		if hf.prependScheme != hfPrependNever {
			return fmt.Errorf("%w: Metaspace add_prefix_space=false contradicts prepend_scheme %q", errUnsupportedModel, scheme)
		}
	}

	hf.split = p.Split == nil || *p.Split

	return nil
}

// specials returns the added tokens to extract. tokenizers gives an added
// token the id its content has in the model vocab (the "id" field only
// triggers a warning) and ignores empty ones.
func (doc *hfTokenizerJSON) specials(vocab map[string]int32) ([]hfSpecial, error) {
	var out []hfSpecial

	seen := make(map[string]bool)

	for _, t := range doc.AddedTokens {
		if t.Content == "" || seen[t.Content] {
			continue
		}

		seen[t.Content] = true

		if t.Normalized == nil {
			return nil, fmt.Errorf("%w: added token %q has no normalized flag", errUnsupportedModel, t.Content)
		}

		if *t.Normalized {
			return nil, fmt.Errorf("%w: added token %q is normalized", errUnsupportedModel, t.Content)
		}

		if t.LStrip || t.RStrip || t.SingleWord {
			return nil, fmt.Errorf("%w: added token %q uses lstrip, rstrip or single_word", errUnsupportedModel, t.Content)
		}

		id, ok := vocab[t.Content]
		if !ok {
			return nil, fmt.Errorf("%w: added token %q is not in the model vocab", errUnsupportedModel, t.Content)
		}

		out = append(out, hfSpecial{content: t.Content, id: id})
	}

	return out, nil
}

// buildJSONVocab puts every vocab entry into the trie, as tokenizers' Unigram
// does, and resolves the byte pieces by name.
func (m *spModel) buildJSONVocab(vocab []hfVocabEntry) error {
	minScore := math.Inf(1)
	m.hf.vocab = make(map[string]int32, len(vocab))

	for i, e := range vocab {
		id := int32(i)
		m.hf.vocab[e.piece] = id
		m.root.insert(e.piece, id, e.score)
		minScore = min(minScore, e.score)
	}

	m.unkScore = minScore - float64(spUnkPenalty)

	if m.byteFallback {
		for b := range m.byteIDs {
			name := fmt.Sprintf("<0x%02X>", b)

			id, ok := m.hf.vocab[name]
			if !ok {
				return fmt.Errorf("%w: byte_fallback is set but byte piece %s is missing", errUnsupportedModel, name)
			}

			m.byteIDs[b] = id
		}
	}

	return nil
}

// ── encoding ─────────────────────────────────────────────────────────────────

// specialAt returns the longest added token text starts with.
func (hf *hfConfig) specialAt(text string) (hfSpecial, bool) {
	var (
		best  hfSpecial
		found bool
	)

	for _, s := range hf.specials {
		if len(s.content) > len(best.content) && strings.HasPrefix(text, s.content) {
			best, found = s, true
		}
	}

	return best, found
}

// hfPieces encodes text with the tokenizers pipeline. Added tokens become one
// piece with their content as text; other pieces carry the normalized text
// they cover, as on the SentencePiece path.
func (m *spModel) hfPieces(text string) []Piece {
	out := []Piece{}
	last := 0

	for i := 0; i < len(text); {
		s, ok := m.hf.specialAt(text[i:])
		if !ok {
			i++

			continue
		}

		if last < i {
			out = m.hfSegmentPieces(out, text[last:i], last == 0)
		}

		out = append(out, Piece{ID: int64(s.id), Text: s.content})
		i += len(s.content)
		last = i
	}

	if last < len(text) {
		out = m.hfSegmentPieces(out, text[last:], last == 0)
	}

	return out
}

// hfSegmentPieces normalizes and encodes one non-empty segment between added
// tokens; atStart reports whether it starts the input.
func (m *spModel) hfSegmentPieces(out []Piece, segment string, atStart bool) []Piece {
	runes := make([]rune, 0, len(segment)+2)
	if m.hf.prepend {
		runes = append(runes, spSep)
	}

	for _, r := range segment {
		if r == ' ' {
			r = spSep
		}

		runes = append(runes, r)
	}

	addPrefix := m.hf.prependScheme == hfPrependAlways ||
		m.hf.prependScheme == hfPrependFirst && atStart
	if addPrefix && runes[0] != spSep {
		runes = append([]rune{spSep}, runes...)
	}

	start := 0

	for i := 1; i <= len(runes); i++ {
		if i == len(runes) || m.hf.split && runes[i] == spSep {
			out = m.hfWordPieces(out, runes[start:i])
			start = i
		}
	}

	return out
}

// hfWordPieces encodes one pre-tokenized word. Consecutive unknown pieces
// (including vocab matches of the unknown piece itself) fuse into one token,
// which resolves to its vocab id if it has one, else to byte pieces (each
// character still attributed to its last byte), else to a single unknown.
func (m *spModel) hfWordPieces(out []Piece, word []rune) []Piece {
	path := m.bestPath(word)

	for i := 0; i < len(path); {
		s := path[i]
		if s.id != m.unkID {
			out = append(out, Piece{ID: int64(s.id), Text: string(word[s.start:s.end])})
			i++

			continue
		}

		j := i + 1
		for j < len(path) && path[j].id == m.unkID {
			j++
		}

		run := path[i:j]
		fused := string(word[run[0].start:run[len(run)-1].end])

		id, inVocab := m.hf.vocab[fused]

		switch {
		case inVocab:
			out = append(out, Piece{ID: int64(id), Text: fused})
		case m.byteFallback:
			for _, u := range run {
				out = m.appendByteFallback(out, string(word[u.start:u.end]))
			}
		default:
			out = append(out, Piece{ID: int64(m.unkID), Text: fused})
		}

		i = j
	}

	return out
}
