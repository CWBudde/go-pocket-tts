package text

import "unicode"

type Preprocessor struct{}

func NewPreprocessor() *Preprocessor {
	return &Preprocessor{}
}

func (p *Preprocessor) Preprocess(input string) []int {
	tokens := make([]int, 0, len(input))
	for _, r := range input {
		if unicode.IsSpace(r) {
			continue
		}

		tokens = append(tokens, int(unicode.ToLower(r)))
	}

	return tokens
}
