package texttest

import (
	"slices"
	"testing"
)

func TestEncodePieces(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want []string
	}{
		{"", nil},
		{".!...?", []string{"▁", ".", "!", ".", ".", ".", "?"}},
		{"Pi is 3.14.", []string{"▁Pi", "▁is", "▁3", ".", "14", "."}},
		{"98.6°F, ok", []string{"▁98", ".", "6", "°", "F", ",", "▁ok"}},
		{"word .", []string{"▁word", "▁", "."}},
		{"  Hi", []string{"▁", "▁", "▁Hi"}},
	} {
		pieces, err := Tokenizer{}.EncodePieces(tc.in)
		if err != nil {
			t.Fatalf("EncodePieces(%q): %v", tc.in, err)
		}

		got := make([]string, 0, len(pieces))
		for _, p := range pieces {
			got = append(got, p.Text)

			if p.ID != PieceID(p.Text) {
				t.Errorf("EncodePieces(%q): piece %q has id %d, want %d", tc.in, p.Text, p.ID, PieceID(p.Text))
			}
		}

		if !slices.Equal(got, tc.want) {
			t.Errorf("EncodePieces(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestEncodeMatchesEncodePieces(t *testing.T) {
	const in = "Hello, world. Pi is 3.14!"

	ids, err := Tokenizer{}.Encode(in)
	if err != nil {
		t.Fatal(err)
	}

	pieces, err := Tokenizer{}.EncodePieces(in)
	if err != nil {
		t.Fatal(err)
	}

	if len(ids) != len(pieces) {
		t.Fatalf("len(Encode) = %d, len(EncodePieces) = %d", len(ids), len(pieces))
	}

	for i := range ids {
		if ids[i] != pieces[i].ID {
			t.Errorf("id[%d] = %d, piece id = %d", i, ids[i], pieces[i].ID)
		}
	}
}
