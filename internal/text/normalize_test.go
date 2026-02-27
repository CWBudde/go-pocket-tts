package text

import (
	"errors"
	"testing"
)

func TestNormalize(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    string
		wantErr error
	}{
		{
			name:  "passthrough clean text",
			input: "Hello world",
			want:  "Hello world",
		},
		{
			name:  "trims leading whitespace",
			input: "  Hello",
			want:  "Hello",
		},
		{
			name:  "trims trailing whitespace",
			input: "Hello  ",
			want:  "Hello",
		},
		{
			name:  "trims leading and trailing whitespace",
			input: "  Hello world  ",
			want:  "Hello world",
		},
		{
			name:  "trims tabs and newlines from edges",
			input: "\t\n Hello \n\t",
			want:  "Hello",
		},
		{
			name:  "normalizes CRLF to LF",
			input: "line one\r\nline two",
			want:  "line one\nline two",
		},
		{
			name:  "normalizes bare CR to LF",
			input: "line one\rline two",
			want:  "line one\nline two",
		},
		{
			name:  "preserves existing LF",
			input: "line one\nline two",
			want:  "line one\nline two",
		},
		{
			name:  "normalizes mixed line endings",
			input: "a\r\nb\rc\nd",
			want:  "a\nb\nc\nd",
		},
		{
			name:    "rejects empty string",
			input:   "",
			wantErr: ErrEmptyText,
		},
		{
			name:    "rejects whitespace-only string",
			input:   "   \t\n  ",
			wantErr: ErrEmptyText,
		},
		{
			name:  "preserves unicode content",
			input: "  Héllo wörld  ",
			want:  "Héllo wörld",
		},
		{
			name:  "preserves internal whitespace",
			input: "  hello   world  ",
			want:  "hello   world",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Normalize(tt.input)
			if tt.wantErr != nil {
				if err == nil {
					t.Fatalf("expected error %v, got nil", tt.wantErr)
				}

				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("expected error %v, got %v", tt.wantErr, err)
				}

				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if got != tt.want {
				t.Errorf("Normalize(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}
