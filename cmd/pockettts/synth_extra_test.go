package main

import (
	"errors"
	"os/exec"
	"testing"

	"github.com/example/go-pocket-tts/internal/audio"
)

func TestMapSynthError_NotFound(t *testing.T) {
	err := mapSynthError(exec.ErrNotFound)
	if err == nil {
		t.Fatal("expected non-nil error")
	}

	if !errors.Is(err, exec.ErrNotFound) {
		t.Errorf("expected ErrNotFound to be wrapped, got %v", err)
	}
}

func TestMapSynthError_ExitError(t *testing.T) {
	// Create an *exec.ExitError by running a command that fails.
	cmd := exec.Command("false")

	runErr := cmd.Run()
	if runErr == nil {
		t.Skip("'false' command succeeded unexpectedly")
	}

	mapped := mapSynthError(runErr)
	if mapped == nil {
		t.Fatal("expected non-nil error")
	}

	var exitErr *exec.ExitError
	if !errors.As(mapped, &exitErr) {
		t.Errorf("expected *exec.ExitError to be wrapped, got %T: %v", mapped, mapped)
	}
}

func TestMapSynthError_OtherError(t *testing.T) {
	sentinel := errors.New("some network error")

	got := mapSynthError(sentinel)
	if !errors.Is(got, sentinel) {
		t.Errorf("expected sentinel error to pass through unchanged, got %v", got)
	}
}

func TestBuildSynthesisChunks_EmptyInput(t *testing.T) {
	_, err := buildSynthesisChunks("", false, 220)
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestBuildSynthesisChunks_WhitespaceOnlyInput(t *testing.T) {
	_, err := buildSynthesisChunks("   \n\t  ", true, 220)
	if err == nil {
		t.Fatal("expected error for whitespace-only input")
	}
}

func TestBuildPassthroughArgs_EmptyItems(t *testing.T) {
	got, err := buildPassthroughArgs([]string{"", "  "})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(got) != 0 {
		t.Errorf("expected empty result for blank items, got %v", got)
	}
}

func TestBuildPassthroughArgs_MissingEquals(t *testing.T) {
	_, err := buildPassthroughArgs([]string{"noequals"})
	if err == nil {
		t.Fatal("expected error for item without '='")
	}
}

func TestBuildPassthroughArgs_EmptyKey(t *testing.T) {
	_, err := buildPassthroughArgs([]string{"=value"})
	if err == nil {
		t.Fatal("expected error for empty key")
	}
}

func TestWriteSynthOutput_NilStdout(t *testing.T) {
	err := writeSynthOutput("-", []byte("data"), nil)
	if err == nil {
		t.Fatal("expected error when stdout is nil")
	}
}

func TestConcatenateWAVChunks_InvalidData(t *testing.T) {
	_, err := concatenateWAVChunks([][]byte{[]byte("not a wav")})
	if err == nil {
		t.Fatal("expected error for invalid WAV data")
	}
}

func TestConcatenateWAVChunks_MergesSamples(t *testing.T) {
	wav1, err := audio.EncodeWAV([]float32{0.1, 0.2})
	if err != nil {
		t.Fatalf("EncodeWAV: %v", err)
	}

	wav2, err := audio.EncodeWAV([]float32{0.3, 0.4, 0.5})
	if err != nil {
		t.Fatalf("EncodeWAV: %v", err)
	}

	merged, err := concatenateWAVChunks([][]byte{wav1, wav2})
	if err != nil {
		t.Fatalf("concatenateWAVChunks: %v", err)
	}

	samples, err := audio.DecodeWAV(merged)
	if err != nil {
		t.Fatalf("DecodeWAV: %v", err)
	}

	if len(samples) != 5 {
		t.Errorf("expected 5 samples, got %d", len(samples))
	}
}

func TestApplyDSPToWAV_InvalidInput(t *testing.T) {
	_, err := applyDSPToWAV([]byte("not a wav"), synthDSPOptions{Normalize: true})
	if err == nil {
		t.Fatal("expected error for invalid WAV input")
	}
}

func TestApplyDSPToWAV_NoOpsAreIdempotent(t *testing.T) {
	in, err := audio.EncodeWAV([]float32{0.1, 0.2, 0.3})
	if err != nil {
		t.Fatalf("EncodeWAV: %v", err)
	}
	// No DSP options set — should round-trip cleanly.
	out, err := applyDSPToWAV(in, synthDSPOptions{})
	if err != nil {
		t.Fatalf("applyDSPToWAV: %v", err)
	}

	samples, err := audio.DecodeWAV(out)
	if err != nil {
		t.Fatalf("DecodeWAV: %v", err)
	}

	if len(samples) != 3 {
		t.Errorf("expected 3 samples, got %d", len(samples))
	}
}

func TestApplyDSPToWAV_DCBlock(t *testing.T) {
	in, err := audio.EncodeWAV([]float32{0.1, 0.2, 0.3, 0.4})
	if err != nil {
		t.Fatalf("EncodeWAV: %v", err)
	}

	out, err := applyDSPToWAV(in, synthDSPOptions{DCBlock: true})
	if err != nil {
		t.Fatalf("applyDSPToWAV with DCBlock: %v", err)
	}

	_, err = audio.DecodeWAV(out)
	if err != nil {
		t.Fatalf("output is not valid WAV: %v", err)
	}
}
