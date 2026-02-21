//go:build integration

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/audio"
)

func TestSynthIntegration(t *testing.T) {
	if _, err := exec.LookPath("pocket-tts"); err != nil {
		t.Skip("pocket-tts binary not found in PATH")
	}

	voice := strings.TrimSpace(os.Getenv("POCKETTTS_TEST_VOICE"))
	if voice == "" {
		t.Skip("set POCKETTTS_TEST_VOICE to a valid fixture voice for integration test")
	}

	out := filepath.Join(t.TempDir(), "out.wav")
	root := NewRootCmd()
	root.SetArgs([]string{
		"synth",
		"--backend", "cli",
		"--text", "Hello.",
		"--voice", voice,
		"--out", out,
	})

	if err := root.Execute(); err != nil {
		if strings.Contains(err.Error(), "executable not found") {
			t.Skipf("pocket-tts unavailable: %v", err)
		}
		t.Fatalf("synth command failed: %v", err)
	}

	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read output wav: %v", err)
	}
	if len(data) < 12 || string(data[:4]) != "RIFF" {
		t.Fatalf("invalid WAV RIFF header")
	}

	samples, err := audio.DecodeWAV(data)
	if err != nil {
		t.Fatalf("decode output wav: %v", err)
	}
	if len(samples) == 0 {
		t.Fatalf("expected non-zero duration audio")
	}
}
