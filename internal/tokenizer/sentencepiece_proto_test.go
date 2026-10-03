package tokenizer

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"
)

// testPiece is one vocab entry of a synthetic model.
type testPiece struct {
	piece string
	score float32
	typ   int // 0: omit the field (proto default NORMAL)
}

// testModel describes a synthetic ModelProto. Nil option pointers leave the
// field out so the proto2 default applies.
type testModel struct {
	pieces                 []testPiece
	modelType              int // 0: omit
	byteFallback           bool
	treatWhitespaceSuffix  bool
	charsmap               string
	addDummyPrefix         *bool
	removeExtraWhitespaces *bool
	escapeWhitespaces      *bool
}

func boolPtr(b bool) *bool { return &b }

func appendBoolField(b []byte, num protowire.Number, v bool) []byte {
	b = protowire.AppendTag(b, num, protowire.VarintType)

	return protowire.AppendVarint(b, protowire.EncodeBool(v))
}

func (m testModel) encode() []byte {
	var out []byte

	for _, p := range m.pieces {
		var sp []byte

		sp = protowire.AppendTag(sp, 1, protowire.BytesType)
		sp = protowire.AppendString(sp, p.piece)
		sp = protowire.AppendTag(sp, 2, protowire.Fixed32Type)

		sp = protowire.AppendFixed32(sp, math.Float32bits(p.score))
		if p.typ != 0 {
			sp = protowire.AppendTag(sp, 3, protowire.VarintType)
			sp = protowire.AppendVarint(sp, uint64(p.typ))
		}

		out = protowire.AppendTag(out, 1, protowire.BytesType)
		out = protowire.AppendBytes(out, sp)
	}

	var trainer []byte
	if m.modelType != 0 {
		trainer = protowire.AppendTag(trainer, 3, protowire.VarintType)
		trainer = protowire.AppendVarint(trainer, uint64(m.modelType))
	}

	trainer = appendBoolField(trainer, 24, m.treatWhitespaceSuffix)
	trainer = appendBoolField(trainer, 35, m.byteFallback)
	out = protowire.AppendTag(out, 2, protowire.BytesType)
	out = protowire.AppendBytes(out, trainer)

	var norm []byte

	norm = protowire.AppendTag(norm, 1, protowire.BytesType)
	norm = protowire.AppendString(norm, "identity")
	norm = protowire.AppendTag(norm, 2, protowire.BytesType)

	norm = protowire.AppendString(norm, m.charsmap)
	if m.addDummyPrefix != nil {
		norm = appendBoolField(norm, 3, *m.addDummyPrefix)
	}

	if m.removeExtraWhitespaces != nil {
		norm = appendBoolField(norm, 4, *m.removeExtraWhitespaces)
	}

	if m.escapeWhitespaces != nil {
		norm = appendBoolField(norm, 5, *m.escapeWhitespaces)
	}

	out = protowire.AppendTag(out, 3, protowire.BytesType)

	return protowire.AppendBytes(out, norm)
}

const (
	testNormal  = 1
	testUnknown = 2
	testControl = 3
	testUnused  = 5
	testByte    = 6
)

// basePieces is a tiny vocab: <unk>=0, <s>=1, ▁=2, a=3, b=4, ▁a=5, ab=6.
func basePieces() []testPiece {
	return []testPiece{
		{piece: "<unk>", typ: testUnknown},
		{piece: "<s>", typ: testControl},
		{piece: "▁", score: -2},
		{piece: "a", score: -3},
		{piece: "b", score: -3},
		{piece: "▁a", score: -1},
		{piece: "ab", score: -4, typ: testNormal},
	}
}

func bytePieces() []testPiece {
	out := make([]testPiece, 256)
	for i := range out {
		out[i] = testPiece{piece: fmt.Sprintf("<0x%02X>", i), typ: testByte}
	}

	return out
}

func encodeSynthetic(t *testing.T, m testModel, text string) []Piece {
	t.Helper()

	tok, err := NewSentencePieceTokenizerFromBytes(m.encode())
	if err != nil {
		t.Fatalf("load synthetic model: %v", err)
	}

	pieces, err := tok.EncodePieces(text)
	if err != nil {
		t.Fatalf("EncodePieces(%q): %v", text, err)
	}

	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode(%q): %v", text, err)
	}

	if !slices.Equal(ids, pieceIDs(pieces)) {
		t.Fatalf("Encode(%q) = %v, EncodePieces ids %v", text, ids, pieceIDs(pieces))
	}

	return pieces
}

func TestSentencePieceModel_LoadErrors(t *testing.T) {
	cases := map[string]struct {
		data []byte
		want string
	}{
		"empty":   {data: nil, want: "must not be empty"},
		"garbage": {data: []byte{0xff, 0xff, 0xff}, want: "parse sentencepiece model"},
		"bpe": {
			data: testModel{pieces: basePieces(), modelType: 2}.encode(),
			want: "only UNIGRAM",
		},
		"charsmap": {
			data: testModel{pieces: basePieces(), charsmap: "\x01\x02"}.encode(),
			want: "precompiled charsmap",
		},
		"no unk": {
			data: testModel{pieces: basePieces()[1:]}.encode(),
			want: "exactly one UNKNOWN piece",
		},
		"missing byte pieces": {
			data: testModel{pieces: append(basePieces(), bytePieces()[:255]...), byteFallback: true}.encode(),
			want: "<0xFF>",
		},
		"whitespace suffix": {
			data: testModel{pieces: basePieces(), treatWhitespaceSuffix: true}.encode(),
			want: "treat_whitespace_as_suffix",
		},
	}

	for name, c := range cases {
		_, err := NewSentencePieceTokenizerFromBytes(c.data)
		if err == nil {
			t.Errorf("%s: expected an error", name)

			continue
		}

		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %q does not mention %q", name, err, c.want)
		}
	}

	_, err := NewSentencePieceTokenizerFromBytes(nil)
	if !errors.Is(err, errEmptyModelData) {
		t.Errorf("empty data: got %v, want errEmptyModelData", err)
	}
}

func TestSentencePieceModel_Viterbi(t *testing.T) {
	m := testModel{pieces: basePieces()}

	// ▁a (-1) + b (-3) = -4 beats ▁ (-2) + ab (-4) = -6.
	got := encodeSynthetic(t, m, "ab")
	want := []Piece{{ID: 5, Text: "▁a"}, {ID: 4, Text: "b"}}

	if !slices.Equal(got, want) {
		t.Errorf("ab: got %v, want %v", got, want)
	}

	// Without byte fallback an unknown character is one <unk> piece per
	// character; runs are not merged.
	got = encodeSynthetic(t, m, "a€€")
	want = []Piece{{ID: 5, Text: "▁a"}, {ID: 0, Text: "€"}, {ID: 0, Text: "€"}}

	if !slices.Equal(got, want) {
		t.Errorf("a€€: got %v, want %v", got, want)
	}
}

func TestSentencePieceModel_UnusedPiecesSkipped(t *testing.T) {
	pieces := append(basePieces(), testPiece{piece: "▁ab", score: 0, typ: testUnused})

	got := encodeSynthetic(t, testModel{pieces: pieces}, "ab")
	want := []Piece{{ID: 5, Text: "▁a"}, {ID: 4, Text: "b"}}

	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestSentencePieceModel_ByteFallbackIDsByName(t *testing.T) {
	// Byte pieces placed in reverse order: ids must come from the piece
	// names, not from an assumed offset.
	bytes := bytePieces()
	slices.Reverse(bytes)

	m := testModel{pieces: append(basePieces(), bytes...), byteFallback: true}
	byteID := func(b byte) int64 { return int64(len(basePieces()) + 255 - int(b)) }

	got := encodeSynthetic(t, m, "a€")
	want := []Piece{
		{ID: 5, Text: "▁a"},
		{ID: byteID(0xE2), Text: ""},
		{ID: byteID(0x82), Text: ""},
		{ID: byteID(0xAC), Text: "€"},
	}

	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestSentencePieceModel_NormalizerFlags(t *testing.T) {
	cases := []struct {
		name string
		m    testModel
		text string
		want string
	}{
		{name: "defaults", m: testModel{}, text: "  a  b ", want: "▁a▁b"},
		{name: "defaults only spaces", m: testModel{}, text: "   ", want: ""},
		{
			name: "keep whitespace",
			m:    testModel{removeExtraWhitespaces: boolPtr(false)},
			text: "  a  b ", want: "▁▁▁a▁▁b▁",
		},
		{
			name: "no dummy prefix",
			m:    testModel{addDummyPrefix: boolPtr(false), removeExtraWhitespaces: boolPtr(false)},
			text: "a b", want: "a▁b",
		},
		{
			name: "no escaping",
			m:    testModel{escapeWhitespaces: boolPtr(false), removeExtraWhitespaces: boolPtr(false)},
			text: "a b", want: " a b",
		},
		{
			name: "only space is escaped",
			m:    testModel{removeExtraWhitespaces: boolPtr(false)},
			text: "a\tb ", want: "▁a\tb ",
		},
	}

	for _, c := range cases {
		c.m.pieces = basePieces()

		if got := pieceSurface(encodeSynthetic(t, c.m, c.text)); got != c.want {
			t.Errorf("%s: surface(%q) = %q, want %q", c.name, c.text, got, c.want)
		}
	}
}

// FuzzSentencePieceModel feeds arbitrary model bytes and text through the
// loader and the encoder: neither may panic, and a loaded model must keep
// Encode and EncodePieces in step.
func FuzzSentencePieceModel(f *testing.F) {
	valid := testModel{pieces: append(basePieces(), bytePieces()...), byteFallback: true}.encode()
	f.Add(valid, "Hello, world! 3.14 €\t\xff▁")
	f.Add(valid[:len(valid)/2], "a")
	f.Add(testModel{pieces: basePieces(), charsmap: "\x01"}.encode(), "")
	f.Add([]byte{0xff, 0xff, 0xff}, "x")
	f.Add([]byte{0x0a, 0x05, 0x0a, 0x01, 'a', 0x15}, "a")

	f.Fuzz(func(t *testing.T, data []byte, text string) {
		tok, err := NewSentencePieceTokenizerFromBytes(data)
		if err != nil {
			return
		}

		ids, err := tok.Encode(text)
		if err != nil {
			t.Fatalf("Encode(%q): %v", text, err)
		}

		pieces, err := tok.EncodePieces(text)
		if err != nil {
			t.Fatalf("EncodePieces(%q): %v", text, err)
		}

		if !slices.Equal(ids, pieceIDs(pieces)) {
			t.Fatalf("Encode(%q) = %v, EncodePieces ids %v", text, ids, pieceIDs(pieces))
		}
	})
}
