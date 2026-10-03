package text

import (
	"unicode/utf8"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
)

// Options selects the per-model text preparation steps (upstream
// prepare_text_prompt arguments, taken from the model config).
type Options struct {
	// PadShortInputs prepends 8 spaces to prompts of fewer than 5 words
	// (pad_with_spaces_for_short_inputs).
	PadShortInputs bool
	// CapitalizeFirst upper-cases the first character (capitalize_first_letter).
	CapitalizeFirst bool
	// RemoveSemicolons replaces ';' with ',' (remove_semicolons).
	RemoveSemicolons bool
	// AppendTerminalPunctuation makes the prompt end with sentence-final
	// punctuation (append_terminal_punctuation).
	AppendTerminalPunctuation bool
	// ReplaceCharacters maps characters the model never saw in training to
	// their replacement; "" deletes them (replace_characters).
	ReplaceCharacters map[rune]string
}

// DefaultOptions returns the options of the english_2026-01 model, which
// matches the text preparation used before model configs existed.
func DefaultOptions() Options {
	return Options{
		PadShortInputs:            true,
		CapitalizeFirst:           true,
		AppendTerminalPunctuation: true,
	}
}

// OptionsFor returns the text preparation options of a model config, or
// DefaultOptions when cfg is nil. modelcfg.Parse guarantees that every
// replace_characters key is a single character.
func OptionsFor(cfg *modelcfg.ModelConfig) Options {
	if cfg == nil {
		return DefaultOptions()
	}

	opts := Options{
		PadShortInputs:            cfg.PadWithSpacesForShortInputs,
		CapitalizeFirst:           cfg.CapitalizeFirstLetter,
		RemoveSemicolons:          cfg.RemoveSemicolons,
		AppendTerminalPunctuation: cfg.AppendTerminalPunctuation,
	}

	if len(cfg.ReplaceCharacters) > 0 {
		opts.ReplaceCharacters = make(map[rune]string, len(cfg.ReplaceCharacters))
		for from, to := range cfg.ReplaceCharacters {
			r, _ := utf8.DecodeRuneInString(from)
			opts.ReplaceCharacters[r] = to
		}
	}

	return opts
}
