package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// Synthetic tokenizer.json models. The base vocab is
//
//	0 <unk>  1 <s>  2 </s>  3 ▁  4 a  5 b  6 ▁a  7 ab  8 a▁b  9 [A]  10 [A]B
//
// with <s>, </s>, [A] and [A]B as added tokens; jsonWithBytes appends the 256
// <0xXX> pieces as ids 11..266, scored 0 like the converted shipped models.
const jsonBaseVocabSize = 11

func jsonByteID(b byte) int64 { return jsonBaseVocabSize + int64(b) }

func jsonAddedToken(id int, content string, special bool) map[string]any {
	return map[string]any{
		"id": id, "content": content, "single_word": false, "lstrip": false,
		"rstrip": false, "normalized": false, "special": special,
	}
}

func jsonBase() map[string]any {
	return map[string]any{
		"version":    "1.0",
		"truncation": nil,
		"padding":    nil,
		"added_tokens": []any{
			jsonAddedToken(1, "<s>", true),
			jsonAddedToken(2, "</s>", true),
			jsonAddedToken(9, "[A]", false),
			jsonAddedToken(10, "[A]B", false),
		},
		"normalizer": map[string]any{
			"type":        "Sequence",
			"normalizers": []any{map[string]any{"type": "Prepend", "prepend": "▁"}},
		},
		"pre_tokenizer": map[string]any{
			"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true,
		},
		"post_processor": nil,
		"decoder":        nil,
		"model": map[string]any{
			"type":   "Unigram",
			"unk_id": 0,
			"vocab": []any{
				[]any{"<unk>", 0.0},
				[]any{"<s>", 0.0},
				[]any{"</s>", 0.0},
				[]any{"▁", -2.0},
				[]any{"a", -3.0},
				[]any{"b", -3.0},
				[]any{"▁a", -1.0},
				[]any{"ab", -4.0},
				[]any{"a▁b", 0.0},
				[]any{"[A]", -1.0},
				[]any{"[A]B", -1.0},
			},
			"byte_fallback": false,
		},
	}
}

func jsonModel(m map[string]any) map[string]any { return m["model"].(map[string]any) }

func jsonWithBytes() map[string]any {
	m := jsonBase()
	model := jsonModel(m)

	vocab := model["vocab"].([]any)
	for b := range 256 {
		vocab = append(vocab, []any{fmt.Sprintf("<0x%02X>", b), 0.0})
	}

	model["vocab"] = vocab
	model["byte_fallback"] = true

	return m
}

func marshalJSONModel(t testing.TB, m map[string]any) []byte {
	t.Helper()

	data, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal json model: %v", err)
	}

	return data
}

func encodeJSONSynthetic(t *testing.T, m map[string]any, text string) []Piece {
	t.Helper()

	tok, err := NewJSONTokenizerFromBytes(marshalJSONModel(t, m))
	if err != nil {
		t.Fatalf("load synthetic json: %v", err)
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

type jsonEncodeCase struct {
	text string
	want []Piece
}

func checkJSONEncodeCases(t *testing.T, name string, m map[string]any, cases []jsonEncodeCase) {
	t.Helper()

	for _, c := range cases {
		got := encodeJSONSynthetic(t, m, c.text)
		if !slices.Equal(got, c.want) {
			t.Errorf("%s: EncodePieces(%q)\n got %v\nwant %v", name, c.text, got, c.want)
		}
	}
}

func TestJSONModel_SpecialTokensAndSegments(t *testing.T) {
	checkJSONEncodeCases(t, "base", jsonBase(), []jsonEncodeCase{
		{text: "", want: []Piece{}},
		{text: " ", want: []Piece{{ID: 3, Text: "▁"}, {ID: 3, Text: "▁"}}},
		// ▁a (-1) + b (-3) = -4 beats ▁ (-2) + ab (-4) = -6.
		{text: "ab", want: []Piece{{ID: 6, Text: "▁a"}, {ID: 5, Text: "b"}}},
		// Added tokens are cut out of the raw input; every other segment gets
		// its own '▁' prefix.
		{text: "a<s>b", want: []Piece{{ID: 6, Text: "▁a"}, {ID: 1, Text: "<s>"}, {ID: 3, Text: "▁"}, {ID: 5, Text: "b"}}},
		{text: "<s><s>", want: []Piece{{ID: 1, Text: "<s>"}, {ID: 1, Text: "<s>"}}},
		{text: "</s>", want: []Piece{{ID: 2, Text: "</s>"}}},
		{
			text: "a <s> b",
			want: []Piece{
				{ID: 6, Text: "▁a"},
				{ID: 3, Text: "▁"},
				{ID: 1, Text: "<s>"},
				{ID: 3, Text: "▁"},
				{ID: 3, Text: "▁"},
				{ID: 5, Text: "b"},
			},
		},
		// Leftmost-longest matching among the added tokens.
		{text: "[A]B[A]", want: []Piece{{ID: 10, Text: "[A]B"}, {ID: 9, Text: "[A]"}}},
		// Matching is exact and case-sensitive. Without byte fallback a run of
		// unknown characters fuses into a single <unk>.
		{text: "<S>", want: []Piece{{ID: 3, Text: "▁"}, {ID: 0, Text: "<S>"}}},
		// Metaspace split: words start at every '▁', so "a▁b" (score 0) is
		// never matched across the word boundary.
		{text: "a b", want: []Piece{{ID: 6, Text: "▁a"}, {ID: 3, Text: "▁"}, {ID: 5, Text: "b"}}},
		{text: "a▁b", want: []Piece{{ID: 6, Text: "▁a"}, {ID: 3, Text: "▁"}, {ID: 5, Text: "b"}}},
	})

	noSplit := jsonBase()
	noSplit["pre_tokenizer"].(map[string]any)["split"] = false
	checkJSONEncodeCases(t, "split=false", noSplit, []jsonEncodeCase{
		{text: "a b", want: []Piece{{ID: 3, Text: "▁"}, {ID: 8, Text: "a▁b"}}},
	})
}

func TestJSONModel_PrependSchemes(t *testing.T) {
	withScheme := func(normalizer any, scheme any) map[string]any {
		m := jsonBase()
		m["normalizer"] = normalizer
		pre := m["pre_tokenizer"].(map[string]any)

		if scheme == nil {
			delete(pre, "prepend_scheme")
		} else {
			pre["prepend_scheme"] = scheme
		}

		return m
	}
	prepend := map[string]any{"type": "Prepend", "prepend": "▁"}

	// Prepend normalizer: always a fresh '▁', even before a leading space.
	checkJSONEncodeCases(t, "prepend", withScheme(prepend, "always"), []jsonEncodeCase{
		{text: " a", want: []Piece{{ID: 3, Text: "▁"}, {ID: 6, Text: "▁a"}}},
	})
	// Metaspace "always" only prepends when the segment does not already
	// start with '▁'; the default scheme is "always".
	for _, scheme := range []any{"always", nil} {
		checkJSONEncodeCases(t, fmt.Sprintf("always/%v", scheme), withScheme(nil, scheme), []jsonEncodeCase{
			{text: " a", want: []Piece{{ID: 6, Text: "▁a"}}},
			{text: "a<s>a", want: []Piece{{ID: 6, Text: "▁a"}, {ID: 1, Text: "<s>"}, {ID: 6, Text: "▁a"}}},
		})
	}

	checkJSONEncodeCases(t, "never", withScheme(nil, "never"), []jsonEncodeCase{
		{text: "a b", want: []Piece{{ID: 4, Text: "a"}, {ID: 3, Text: "▁"}, {ID: 5, Text: "b"}}},
	})
	// "first" only prepends to a segment that starts the input.
	checkJSONEncodeCases(t, "first", withScheme(nil, "first"), []jsonEncodeCase{
		{text: "a<s>a", want: []Piece{{ID: 6, Text: "▁a"}, {ID: 1, Text: "<s>"}, {ID: 4, Text: "a"}}},
	})

	// Legacy add_prefix_space=false is accepted together with "never"
	// (tokenizers rejects it with any other scheme, the default included).
	legacy := withScheme(nil, "never")
	legacy["pre_tokenizer"].(map[string]any)["add_prefix_space"] = false
	checkJSONEncodeCases(t, "add_prefix_space=false", legacy, []jsonEncodeCase{
		{text: "a", want: []Piece{{ID: 4, Text: "a"}}},
	})
}

func TestJSONModel_ByteFallback(t *testing.T) {
	checkJSONEncodeCases(t, "bytes", jsonWithBytes(), []jsonEncodeCase{
		{
			text: "a€",
			want: []Piece{{ID: 6, Text: "▁a"}, {ID: jsonByteID(0xE2)}, {ID: jsonByteID(0x82)}, {ID: jsonByteID(0xAC), Text: "€"}},
		},
		// A fused run of unknown characters still attributes each character
		// to its last byte piece.
		{
			text: "€€b",
			want: []Piece{
				{ID: 3, Text: "▁"},
				{ID: jsonByteID(0xE2)},
				{ID: jsonByteID(0x82)},
				{ID: jsonByteID(0xAC), Text: "€"},
				{ID: jsonByteID(0xE2)},
				{ID: jsonByteID(0x82)},
				{ID: jsonByteID(0xAC), Text: "€"},
				{ID: 5, Text: "b"},
			},
		},
		// The <0xXX> pieces are ordinary vocab entries in tokenizers' trie.
		{text: "<0x41>", want: []Piece{{ID: 3, Text: "▁"}, {ID: jsonByteID(0x41), Text: "<0x41>"}}},
		{text: "a<0xE2>b", want: []Piece{{ID: 6, Text: "▁a"}, {ID: jsonByteID(0xE2), Text: "<0xE2>"}, {ID: 5, Text: "b"}}},
		// Invalid UTF-8 becomes U+FFFD (EF BF BD).
		{
			text: "a\xff",
			want: []Piece{{ID: 6, Text: "▁a"}, {ID: jsonByteID(0xEF)}, {ID: jsonByteID(0xBF)}, {ID: jsonByteID(0xBD), Text: "�"}},
		},
		// tokenizers fuses every path node with the unk id, so the literal
		// <unk> piece joins the unknown € before the run is byte-encoded.
		{
			text: "€<unk>",
			want: []Piece{
				{ID: 3, Text: "▁"},
				{ID: jsonByteID(0xE2)},
				{ID: jsonByteID(0x82)},
				{ID: jsonByteID(0xAC), Text: "€"},
				{ID: jsonByteID('<')},
				{ID: jsonByteID('u')},
				{ID: jsonByteID('n')},
				{ID: jsonByteID('k')},
				{ID: jsonByteID('>'), Text: "<unk>"},
			},
		},
	})
}

// TestJSONModel_FusedUnknownRunInVocab: tokenizers looks a fused unknown run
// up in the vocab. With positive scores, unk+unk (20+20) beats the "xy" piece
// (30), and the fused "xy" then resolves to that piece.
func TestJSONModel_FusedUnknownRunInVocab(t *testing.T) {
	m := jsonBase()
	m["added_tokens"] = []any{}
	jsonModel(m)["vocab"] = []any{[]any{"<unk>", 30.0}, []any{"xy", 30.0}, []any{"▁", 30.0}}

	checkJSONEncodeCases(t, "fused", m, []jsonEncodeCase{
		{text: "xy", want: []Piece{{ID: 2, Text: "▁"}, {ID: 1, Text: "xy"}}},
	})
}

func TestJSONModel_LoadErrors(t *testing.T) {
	cases := map[string]struct {
		mutate      func(m map[string]any)
		data        string
		unsupported bool
		want        string
	}{
		"garbage":       {data: "{not json", want: "parse tokenizer.json"},
		"no model":      {data: "{}", unsupported: true, want: "model"},
		"model type":    {mutate: func(m map[string]any) { jsonModel(m)["type"] = "BPE" }, unsupported: true, want: "BPE"},
		"no vocab":      {mutate: func(m map[string]any) { delete(jsonModel(m), "vocab") }, unsupported: true, want: "vocab"},
		"unk_id range":  {mutate: func(m map[string]any) { jsonModel(m)["unk_id"] = jsonBaseVocabSize }, unsupported: true, want: "unk_id"},
		"unk_id null":   {mutate: func(m map[string]any) { jsonModel(m)["unk_id"] = nil }, unsupported: true, want: "unk_id"},
		"unk_id neg":    {mutate: func(m map[string]any) { jsonModel(m)["unk_id"] = -1 }, unsupported: true, want: "unk_id"},
		"vocab entry":   {mutate: func(m map[string]any) { jsonModel(m)["vocab"] = []any{[]any{"<unk>"}} }, want: "vocab"},
		"vocab score":   {mutate: func(m map[string]any) { jsonModel(m)["vocab"] = []any{[]any{"<unk>", "x"}} }, want: "parse tokenizer.json"},
		"null score":    {mutate: func(m map[string]any) { jsonModel(m)["vocab"] = []any{[]any{"<unk>", nil}} }, want: "null"},
		"null piece":    {mutate: func(m map[string]any) { jsonModel(m)["vocab"] = []any{[]any{nil, 0.0}} }, want: "null"},
		"missing bytes": {mutate: func(m map[string]any) { jsonModel(m)["byte_fallback"] = true }, unsupported: true, want: "<0x00>"},
		"normalizer": {
			mutate:      func(m map[string]any) { m["normalizer"] = map[string]any{"type": "NFKC"} },
			unsupported: true, want: "NFKC",
		},
		"prepend string": {
			mutate:      func(m map[string]any) { m["normalizer"] = map[string]any{"type": "Prepend", "prepend": "_"} },
			unsupported: true, want: "Prepend",
		},
		"normalizer in sequence": {
			mutate: func(m map[string]any) {
				m["normalizer"].(map[string]any)["normalizers"] = []any{
					map[string]any{"type": "Prepend", "prepend": "▁"}, map[string]any{"type": "Lowercase"},
				}
			},
			unsupported: true, want: "Lowercase",
		},
		"two prepends": {
			mutate: func(m map[string]any) {
				p := map[string]any{"type": "Prepend", "prepend": "▁"}
				m["normalizer"].(map[string]any)["normalizers"] = []any{p, p}
			},
			unsupported: true, want: "Prepend",
		},
		"no pre_tokenizer": {mutate: func(m map[string]any) { m["pre_tokenizer"] = nil }, unsupported: true, want: "pre_tokenizer"},
		"byte level": {
			mutate:      func(m map[string]any) { m["pre_tokenizer"] = map[string]any{"type": "ByteLevel"} },
			unsupported: true, want: "ByteLevel",
		},
		"replacement": {
			mutate:      func(m map[string]any) { m["pre_tokenizer"].(map[string]any)["replacement"] = "_" },
			unsupported: true, want: "replacement",
		},
		"prepend scheme": {
			mutate:      func(m map[string]any) { m["pre_tokenizer"].(map[string]any)["prepend_scheme"] = "sometimes" },
			unsupported: true, want: "prepend_scheme",
		},
		"add_prefix_space conflict": {
			mutate:      func(m map[string]any) { m["pre_tokenizer"].(map[string]any)["add_prefix_space"] = false },
			unsupported: true, want: "add_prefix_space",
		},
		"add_prefix_space default scheme": {
			mutate: func(m map[string]any) {
				pre := m["pre_tokenizer"].(map[string]any)
				delete(pre, "prepend_scheme")
				pre["add_prefix_space"] = false
			},
			unsupported: true, want: "add_prefix_space",
		},
		"post_processor": {
			mutate:      func(m map[string]any) { m["post_processor"] = map[string]any{"type": "TemplateProcessing"} },
			unsupported: true, want: "post_processor",
		},
		"truncation": {
			mutate:      func(m map[string]any) { m["truncation"] = map[string]any{"max_length": 4} },
			unsupported: true, want: "truncation",
		},
		"normalized added token": {
			mutate: func(m map[string]any) {
				tok := jsonAddedToken(1, "<s>", false)
				tok["normalized"] = true
				m["added_tokens"] = []any{tok}
			},
			unsupported: true, want: "normalized",
		},
		"lstrip added token": {
			mutate: func(m map[string]any) {
				tok := jsonAddedToken(1, "<s>", true)
				tok["lstrip"] = true
				m["added_tokens"] = []any{tok}
			},
			unsupported: true, want: "lstrip",
		},
		"added token not in vocab": {
			mutate:      func(m map[string]any) { m["added_tokens"] = []any{jsonAddedToken(11, "<mask>", true)} },
			unsupported: true, want: "<mask>",
		},
	}

	for name, c := range cases {
		data := []byte(c.data)

		if c.mutate != nil {
			m := jsonBase()
			c.mutate(m)
			data = marshalJSONModel(t, m)
		}

		_, err := NewJSONTokenizerFromBytes(data)
		if err == nil {
			t.Errorf("%s: expected an error", name)

			continue
		}

		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %q does not mention %q", name, err, c.want)
		}

		if c.unsupported != errors.Is(err, errUnsupportedModel) {
			t.Errorf("%s: errors.Is(%v, errUnsupportedModel) = %v, want %v", name, err, !c.unsupported, c.unsupported)
		}
	}

	_, err := NewJSONTokenizerFromBytes(nil)
	if !errors.Is(err, errEmptyModelData) {
		t.Errorf("empty data: got %v, want errEmptyModelData", err)
	}
}

func TestLoadBytes_SniffsFormat(t *testing.T) {
	jsonData := append([]byte(" \n\t\r"), marshalJSONModel(t, jsonBase())...)

	tok, err := LoadBytes(jsonData, 0)
	if err != nil {
		t.Fatalf("LoadBytes(json): %v", err)
	}

	// Only the JSON backend extracts added tokens.
	if ids, _ := tok.Encode("<s>"); !slices.Equal(ids, []int64{1}) {
		t.Errorf("LoadBytes(json).Encode(<s>) = %v, want [1]", ids)
	}

	protoData := testModel{pieces: basePieces()}.encode()

	tok, err = LoadBytes(protoData, 0)
	if err != nil {
		t.Fatalf("LoadBytes(proto): %v", err)
	}

	got := encodeAll(t, tok, "ab")
	want := encodeSynthetic(t, testModel{pieces: basePieces()}, "ab")

	if !slices.Equal(got, want) {
		t.Errorf("LoadBytes(proto) = %v, want %v", got, want)
	}

	_, err = LoadBytes(nil, 0)
	if !errors.Is(err, errEmptyModelData) {
		t.Errorf("LoadBytes(nil): got %v, want errEmptyModelData", err)
	}
}

func encodeAll(t *testing.T, tok Tokenizer, text string) []Piece {
	t.Helper()

	pieces, err := tok.EncodePieces(text)
	if err != nil {
		t.Fatalf("EncodePieces(%q): %v", text, err)
	}

	return pieces
}

func writeTestFile(t *testing.T, path string, data []byte) {
	t.Helper()

	err := os.WriteFile(path, data, 0o600)
	if err != nil {
		t.Fatal(err)
	}
}

func TestLoad_PicksBackendByExtension(t *testing.T) {
	dir := t.TempDir()

	jsonPath := filepath.Join(dir, "tok.JSON")
	writeTestFile(t, jsonPath, marshalJSONModel(t, jsonBase()))

	protoPath := filepath.Join(dir, "tok.model")
	writeTestFile(t, protoPath, testModel{pieces: basePieces()}.encode())

	tok, err := Load(jsonPath, 0)
	if err != nil {
		t.Fatalf("Load(%s): %v", jsonPath, err)
	}

	if ids, _ := tok.Encode("<s>"); !slices.Equal(ids, []int64{1}) {
		t.Errorf("Load(json).Encode(<s>) = %v, want [1]", ids)
	}

	tok, err = Load(protoPath, 0)
	if err != nil {
		t.Fatalf("Load(%s): %v", protoPath, err)
	}

	if got, want := encodeAll(t, tok, "ab"), encodeSynthetic(t, testModel{pieces: basePieces()}, "ab"); !slices.Equal(got, want) {
		t.Errorf("Load(proto) = %v, want %v", got, want)
	}

	// The extension decides: a SentencePiece model named .json is rejected.
	misnamed := filepath.Join(dir, "proto.json")
	writeTestFile(t, misnamed, testModel{pieces: basePieces()}.encode())

	_, err = Load(misnamed, 0)
	if err == nil {
		t.Error("Load(proto bytes named .json): expected an error")
	}

	for name, load := range map[string]func(string) error{
		"Load":             func(p string) error { _, err := Load(p, 0); return err },
		"NewJSONTokenizer": func(p string) error { _, err := NewJSONTokenizer(p); return err },
	} {
		err := load("")
		if !errors.Is(err, ErrEmptyPath) {
			t.Errorf("%s(\"\"): got %v, want ErrEmptyPath", name, err)
		}

		err = load(filepath.Join(dir, "missing.json"))
		if err == nil {
			t.Errorf("%s(missing): expected an error", name)
		}
	}
}

// FuzzJSONTokenizer feeds arbitrary tokenizer.json bytes and text through the
// loader and the encoder: neither may panic, and a loaded model must keep
// Encode and EncodePieces in step.
func FuzzJSONTokenizer(f *testing.F) {
	valid := marshalJSONModel(f, jsonWithBytes())
	f.Add(valid, "Hello, a<s>b [A]B[A] €\t\xff▁ <0x41>")
	f.Add(marshalJSONModel(f, jsonBase()), "a b  <S> </s>")
	f.Add(valid[:len(valid)/2], "a")
	f.Add([]byte("{}"), "x")
	f.Add([]byte(`{"model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0]]},"pre_tokenizer":{"type":"Metaspace","replacement":"▁"}}`), "ab c")
	f.Add([]byte{0xff, 0xff, 0xff}, "x")

	f.Fuzz(func(t *testing.T, data []byte, text string) {
		tok, err := NewJSONTokenizerFromBytes(data)
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
