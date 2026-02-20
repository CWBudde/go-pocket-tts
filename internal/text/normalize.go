package text

import (
	"errors"
	"strings"
)

// ErrEmptyText is returned when the input text is empty or whitespace-only.
var ErrEmptyText = errors.New("text is empty")

// Normalize prepares raw input text for synthesis.
// It trims surrounding whitespace, normalizes line endings to \n,
// and rejects empty or whitespace-only input.
func Normalize(s string) (string, error) {
	// Normalize line endings: CRLF → LF, then bare CR → LF.
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")

	s = strings.TrimSpace(s)

	if s == "" {
		return "", ErrEmptyText
	}

	return s, nil
}
