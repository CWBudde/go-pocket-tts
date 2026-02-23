package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/audio"
)

func TestReadSynthText(t *testing.T) {
	t.Run("uses flag text", func(t *testing.T) {
		got, err := readSynthText("hello", strings.NewReader("ignored"))
		if err != nil {
			t.Fatalf("readSynthText returned error: %v", err)
		}
		if got != "hello" {
			t.Fatalf("expected hello, got %q", got)
		}
	})

	t.Run("falls back to stdin", func(t *testing.T) {
		got, err := readSynthText("", strings.NewReader(" from stdin \n"))
		if err != nil {
			t.Fatalf("readSynthText returned error: %v", err)
		}
		if got != "from stdin" {
			t.Fatalf("expected trimmed stdin text, got %q", got)
		}
	})

	t.Run("fails when both empty", func(t *testing.T) {
		_, err := readSynthText("", strings.NewReader("   \n\t"))
		if err == nil {
			t.Fatal("expected error for empty input")
		}
	})
}

func TestBuildPassthroughArgs(t *testing.T) {
	got, err := buildPassthroughArgs([]string{
		"temperature=0.3",
		"--top-p=0.95",
		"-x=1",
	})
	if err != nil {
		t.Fatalf("buildPassthroughArgs returned error: %v", err)
	}

	want := []string{
		"--temperature=0.3",
		"--top-p=0.95",
		"-x=1",
	}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("unexpected passthrough args: got %v want %v", got, want)
	}
}

func TestResolveSynthBackend(t *testing.T) {
	tests := []struct {
		name    string
		flag    string
		cfg     string
		want    string
		wantErr bool
	}{
		{name: "default native-onnx", want: "native-onnx"},
		{name: "config cli", cfg: "cli", want: "cli"},
		{name: "flag native alias overrides config", flag: "native", cfg: "cli", want: "native-onnx"},
		{name: "flag native-onnx overrides config", flag: "native-onnx", cfg: "cli", want: "native-onnx"},
		{name: "safetensors backend", flag: "native-safetensors", want: "native-safetensors"},
		{name: "invalid", flag: "python", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := resolveSynthBackend(tt.flag, tt.cfg)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("resolveSynthBackend returned error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("expected %q, got %q", tt.want, got)
			}
		})
	}
}

func TestBuildSynthesisChunks(t *testing.T) {
	t.Run("no chunk returns original input", func(t *testing.T) {
		got, err := buildSynthesisChunks("Hello world.", false, 10)
		if err != nil {
			t.Fatalf("buildSynthesisChunks returned error: %v", err)
		}
		if len(got) != 1 || got[0] != "Hello world." {
			t.Fatalf("unexpected chunks: %v", got)
		}
	})

	t.Run("chunk mode splits text", func(t *testing.T) {
		got, err := buildSynthesisChunks("One. Two. Three.", true, 8)
		if err != nil {
			t.Fatalf("buildSynthesisChunks returned error: %v", err)
		}
		want := []string{"One.", "Two.", "Three."}
		if strings.Join(got, "|") != strings.Join(want, "|") {
			t.Fatalf("unexpected chunks: got %v want %v", got, want)
		}
	})
}

func TestSynthesizeChunksConcatenatesPCM(t *testing.T) {
	orig := runChunkSynthesis
	t.Cleanup(func() { runChunkSynthesis = orig })

	runChunkSynthesis = func(_ context.Context, opts synthCLIOptions) ([]byte, error) {
		switch opts.Text {
		case "One.":
			return audio.EncodeWAV([]float32{0.1, 0.2})
		case "Two.":
			return audio.EncodeWAV([]float32{0.3, 0.4, 0.5})
		default:
			return nil, fmt.Errorf("unexpected text %q", opts.Text)
		}
	}

	wavData, err := synthesizeChunks(context.Background(), synthChunksOptions{
		Chunks:    []string{"One.", "Two."},
		ChunkMode: true,
	})
	if err != nil {
		t.Fatalf("synthesizeChunks returned error: %v", err)
	}

	decoded, err := audio.DecodeWAV(wavData)
	if err != nil {
		t.Fatalf("DecodeWAV returned error: %v", err)
	}
	if len(decoded) != 5 {
		t.Fatalf("unexpected merged sample count: got %d want %d", len(decoded), 5)
	}
}

func TestApplyDSPToWAV(t *testing.T) {
	in, err := audio.EncodeWAV([]float32{0.2, 0.4, 0.6, 0.8})
	if err != nil {
		t.Fatalf("EncodeWAV returned error: %v", err)
	}

	out, err := applyDSPToWAV(in, synthDSPOptions{
		Normalize: true,
		FadeInMS:  1,
		FadeOutMS: 1,
	})
	if err != nil {
		t.Fatalf("applyDSPToWAV returned error: %v", err)
	}

	decoded, err := audio.DecodeWAV(out)
	if err != nil {
		t.Fatalf("DecodeWAV returned error: %v", err)
	}
	if len(decoded) != 4 {
		t.Fatalf("unexpected sample count after DSP: got %d want %d", len(decoded), 4)
	}
}

func TestDSPAndWritePipeline_NoSubprocess(t *testing.T) {
	// Mock audio buffer: create a synthetic WAV directly.
	in, err := audio.EncodeWAV([]float32{0.1, 0.2, 0.3, 0.4})
	if err != nil {
		t.Fatalf("EncodeWAV returned error: %v", err)
	}

	processed, err := applyDSPToWAV(in, synthDSPOptions{
		Normalize: true,
		FadeInMS:  1,
		FadeOutMS: 1,
	})
	if err != nil {
		t.Fatalf("applyDSPToWAV returned error: %v", err)
	}

	var stdout bytes.Buffer
	if err := writeSynthOutput("-", processed, &stdout); err != nil {
		t.Fatalf("writeSynthOutput stdout returned error: %v", err)
	}
	if stdout.Len() == 0 {
		t.Fatal("expected stdout bytes")
	}
	if _, err := audio.DecodeWAV(stdout.Bytes()); err != nil {
		t.Fatalf("stdout bytes are not a valid WAV: %v", err)
	}
}

func TestWriteSynthOutput_File(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "out.wav")
	in, err := audio.EncodeWAV([]float32{0.2, 0.4})
	if err != nil {
		t.Fatalf("EncodeWAV returned error: %v", err)
	}

	if err := writeSynthOutput(out, in, nil); err != nil {
		t.Fatalf("writeSynthOutput file returned error: %v", err)
	}

	got, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("expected written file bytes")
	}
	if _, err := audio.DecodeWAV(got); err != nil {
		t.Fatalf("written file is not a valid WAV: %v", err)
	}
}
