package text

import (
	"strings"
	"testing"
)

func TestChunkBySentence(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		maxChars int
		want     []string
	}{
		{
			name:     "single sentence no split needed",
			text:     "Hello world.",
			maxChars: 100,
			want:     []string{"Hello world."},
		},
		{
			name:     "two sentences within limit",
			text:     "Hello. World.",
			maxChars: 100,
			want:     []string{"Hello. World."},
		},
		{
			name:     "two sentences exceeding limit",
			text:     "Hello. World.",
			maxChars: 8,
			want:     []string{"Hello.", "World."},
		},
		{
			name:     "splits on exclamation mark",
			text:     "Hello! World!",
			maxChars: 8,
			want:     []string{"Hello!", "World!"},
		},
		{
			name:     "splits on question mark",
			text:     "Hello? World?",
			maxChars: 8,
			want:     []string{"Hello?", "World?"},
		},
		{
			name:     "mixed sentence terminators",
			text:     "First. Second! Third?",
			maxChars: 10,
			want:     []string{"First.", "Second!", "Third?"},
		},
		{
			name:     "trims whitespace from chunks",
			text:     "First.  Second.  Third.",
			maxChars: 10,
			want:     []string{"First.", "Second.", "Third."},
		},
		{
			name:     "no sentence terminator returns whole text",
			text:     "Hello world",
			maxChars: 5,
			want:     []string{"Hello world"},
		},
		{
			name:     "groups consecutive sentences within limit",
			text:     "A. B. C. D.",
			maxChars: 6,
			want:     []string{"A. B.", "C. D."},
		},
		{
			name:     "maxChars zero means no limit",
			text:     "First. Second. Third.",
			maxChars: 0,
			want:     []string{"First. Second. Third."},
		},
		{
			name:     "single sentence exceeding maxChars stays intact",
			text:     "This is a very long sentence.",
			maxChars: 5,
			want:     []string{"This is a very long sentence."},
		},
		{
			name:     "empty chunks from trailing punctuation are dropped",
			text:     "Hello.",
			maxChars: 100,
			want:     []string{"Hello."},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ChunkBySentence(tt.text, tt.maxChars)
			if len(got) != len(tt.want) {
				t.Fatalf("ChunkBySentence(%q, %d) returned %d chunks %v, want %d chunks %v",
					tt.text, tt.maxChars, len(got), got, len(tt.want), tt.want)
			}

			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("chunk[%d] = %q, want %q", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestChunkBySentence_allChunksNonEmpty(t *testing.T) {
	text := "One. Two. Three! Four? Five."

	chunks := ChunkBySentence(text, 10)
	for i, c := range chunks {
		if strings.TrimSpace(c) == "" {
			t.Errorf("chunk[%d] is empty or whitespace-only", i)
		}
	}
}
