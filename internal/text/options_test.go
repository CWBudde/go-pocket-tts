package text

import (
	"errors"
	"maps"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
)

// upstreamOptions matches the keyword defaults of upstream's tests:
// prepare_text_prompt(text, pad_with_spaces_for_short_inputs=False, remove_semicolons=False).
func upstreamOptions() Options {
	return Options{CapitalizeFirst: true, AppendTerminalPunctuation: true}
}

func mustPrepare(t *testing.T, input string, opts Options) string {
	t.Helper()

	got, _, err := PrepareText(input, opts)
	if err != nil {
		t.Fatalf("PrepareText(%q): %v", input, err)
	}

	return got
}

// Upstream tests/test_split_sentences.py::test_terminal_punctuation_is_added_when_missing.
func TestPrepareText_TerminalPunctuation(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"hello world", "Hello world."},
		{"it costs 42", "It costs 42."},
		{"hello world!", "Hello world!"},
		{"wait for it...", "Wait for it..."},
		{"hello world,", "Hello world."},
		{"hello world:", "Hello world."},
		{"hello world -", "Hello world."},
		{`he said "go home"`, `He said "go home".`},
		{"she whispered 'run'", "She whispered 'run'."},
		{`he said "go home."`, `He said "go home."`},
		{`he said "go home,"`, `He said "go home."`},
		{"is it (really)?", "Is it (really)?"},
		{"see the note (below)", "See the note (below)."},
		{"up by 50%", "Up by 50%."},
	} {
		if got := mustPrepare(t, tc.in, upstreamOptions()); got != tc.want {
			t.Errorf("PrepareText(%q) = %q; want %q", tc.in, got, tc.want)
		}
	}
}

// Upstream test_terminal_punctuation_can_be_disabled_for_punctuation_free_training.
func TestPrepareText_TerminalPunctuationDisabled(t *testing.T) {
	in := "नमस्ते आज आपका दिन कैसा रहा"
	if got := mustPrepare(t, in, Options{}); got != in {
		t.Errorf("PrepareText(%q) = %q; want it unchanged", in, got)
	}
}

// Upstream test_capitalization_can_be_switched_off.
func TestPrepareText_CapitalizeFirst(t *testing.T) {
	if got := mustPrepare(t, "salAm hAle SomA", upstreamOptions()); got != "SalAm hAle SomA." {
		t.Errorf("capitalize on: got %q", got)
	}

	off := upstreamOptions()
	off.CapitalizeFirst = false

	if got := mustPrepare(t, "salAm hAle SomA", off); got != "salAm hAle SomA." {
		t.Errorf("capitalize off: got %q", got)
	}
}

// Upstream test_replace_characters_rewrites_unseen_characters_before_capitalizing.
func TestPrepareText_ReplaceCharacters(t *testing.T) {
	drop := map[rune]string{'"': "", '¡': "", '¿': "", '«': "", '»': ""}

	withDrop := upstreamOptions()
	withDrop.ReplaceCharacters = drop

	withApostrophe := upstreamOptions()
	withApostrophe.ReplaceCharacters = map[rune]string{'’': "'"}
	maps.Copy(withApostrophe.ReplaceCharacters, drop)

	for _, tc := range []struct {
		in, want string
		opts     Options
	}{
		{`"¡Venid a mí, hombres!" Alzó la voz.`, "Venid a mí, hombres! Alzó la voz.", withDrop},
		{"il a dit « l’homme »", "Il a dit l'homme.", withApostrophe},
		{`"Vieni stasera?", chiese.`, "Vieni stasera? chiese.", withDrop},
		// Empty by default: configs that don't set it are unchanged.
		{`"Yes," she said.`, `"Yes," she said.`, upstreamOptions()},
	} {
		if got := mustPrepare(t, tc.in, tc.opts); got != tc.want {
			t.Errorf("PrepareText(%q) = %q; want %q", tc.in, got, tc.want)
		}
	}

	_, _, err := PrepareText(`"  "`, withDrop)
	if !errors.Is(err, ErrEmptyText) {
		t.Errorf("PrepareText of only replaced characters: err = %v; want ErrEmptyText", err)
	}
}

func TestPrepareText_RemoveSemicolons(t *testing.T) {
	opts := upstreamOptions()
	if got := mustPrepare(t, "eins; zwei", opts); got != "Eins; zwei." {
		t.Errorf("remove_semicolons off: got %q", got)
	}

	opts.RemoveSemicolons = true
	if got := mustPrepare(t, "eins; zwei", opts); got != "Eins, zwei." {
		t.Errorf("remove_semicolons on: got %q", got)
	}
}

func TestPrepareText_PadShortInputs(t *testing.T) {
	opts := upstreamOptions()
	if got := mustPrepare(t, "hi", opts); got != "Hi." {
		t.Errorf("pad off: got %q", got)
	}

	opts.PadShortInputs = true
	if got := mustPrepare(t, "hi", opts); got != "        Hi." {
		t.Errorf("pad on: got %q", got)
	}

	if got := mustPrepare(t, "one two three four five", opts); got != "One two three four five." {
		t.Errorf("pad on, 5 words: got %q", got)
	}
}

// Upstream counts words for the frames_after_eos guess after the character
// replacement and before the terminal punctuation fix-up.
func TestPrepareText_WordCount(t *testing.T) {
	german := OptionsFor(lookup(t, "german"))

	for _, tc := range []struct {
		in    string
		opts  Options
		words int
	}{
		{"hello world -", upstreamOptions(), 3},
		{"hi", DefaultOptions(), 1},
		{"„ Hallo “ Welt", german, 2},
	} {
		_, words, err := PrepareText(tc.in, tc.opts)
		if err != nil {
			t.Fatalf("PrepareText(%q): %v", tc.in, err)
		}

		if words != tc.words {
			t.Errorf("PrepareText(%q) words = %d; want %d", tc.in, words, tc.words)
		}
	}
}

// Upstream split_into_best_sentences prepares the whole text before splitting,
// so a deleted quote cannot leave a stray-comma chunk behind.
func TestPrepareChunks_ReplaceCharactersBeforeSplitting(t *testing.T) {
	opts := upstreamOptions()
	opts.ReplaceCharacters = map[rune]string{'"': ""}

	// maxTokens 2 forces a split after "Hi?" (the stub counts words).
	chunks, err := PrepareChunks(`"Hi?", she said.`, &stubTokenizer{}, 2, opts)
	if err != nil {
		t.Fatalf("PrepareChunks: %v", err)
	}

	texts := make([]string, 0, len(chunks))
	for _, c := range chunks {
		texts = append(texts, c.Text)
	}

	if len(texts) != 2 || texts[0] != "Hi?" || texts[1] != "She said." {
		t.Errorf("chunks = %q; want [\"Hi?\" \"She said.\"]", texts)
	}
}

func lookup(t *testing.T, lang string) *modelcfg.ModelConfig {
	t.Helper()

	cfg, err := modelcfg.Lookup(lang)
	if err != nil {
		t.Fatalf("modelcfg.Lookup(%q): %v", lang, err)
	}

	return cfg
}

func TestOptionsFor(t *testing.T) {
	if got := OptionsFor(nil); !optionsEqual(got, DefaultOptions()) {
		t.Errorf("OptionsFor(nil) = %+v; want DefaultOptions() %+v", got, DefaultOptions())
	}

	en := OptionsFor(lookup(t, "english_2026-01"))
	if !optionsEqual(en, DefaultOptions()) {
		t.Errorf("english_2026-01 = %+v; want DefaultOptions() %+v", en, DefaultOptions())
	}

	de := OptionsFor(lookup(t, "german"))
	if de.PadShortInputs || !de.RemoveSemicolons || !de.CapitalizeFirst || !de.AppendTerminalPunctuation {
		t.Errorf("german flags = %+v; want no pad, remove semicolons, capitalize, terminal punctuation", de)
	}

	for r, want := range map[rune]string{'„': "", '“': "", '"': "", '’': "'", '(': ""} {
		got, ok := de.ReplaceCharacters[r]
		if !ok || got != want {
			t.Errorf("german ReplaceCharacters[%q] = %q (present %v); want %q", r, got, ok, want)
		}
	}

	if got := mustPrepare(t, "„Guten Tag“; dies ist ein Test", de); got != "Guten Tag, dies ist ein Test." {
		t.Errorf("german PrepareText = %q", got)
	}
}

func optionsEqual(a, b Options) bool {
	if a.PadShortInputs != b.PadShortInputs || a.CapitalizeFirst != b.CapitalizeFirst ||
		a.RemoveSemicolons != b.RemoveSemicolons || a.AppendTerminalPunctuation != b.AppendTerminalPunctuation ||
		len(a.ReplaceCharacters) != len(b.ReplaceCharacters) {
		return false
	}

	for k, v := range a.ReplaceCharacters {
		if w, ok := b.ReplaceCharacters[k]; !ok || w != v {
			return false
		}
	}

	return true
}
