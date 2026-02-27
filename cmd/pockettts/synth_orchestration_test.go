package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/audio"
	"github.com/example/go-pocket-tts/internal/config"
)

func TestRunSynthCommand_CLIFromStdinToStdout(t *testing.T) {
	orig := runChunkSynthesis
	t.Cleanup(func() { runChunkSynthesis = orig })

	var gotTexts []string

	runChunkSynthesis = func(_ context.Context, opts synthCLIOptions) ([]byte, error) {
		gotTexts = append(gotTexts, opts.Text)
		if opts.ConfigPath != "cli-config.yml" {
			t.Fatalf("unexpected cli config path: %q", opts.ConfigPath)
		}

		if !opts.Quiet {
			t.Fatal("expected quiet=true")
		}

		return audio.EncodeWAV([]float32{0.2, 0.4})
	}

	cfg := config.DefaultConfig()
	cfg.TTS.Backend = config.BackendCLI
	cfg.TTS.CLIConfigPath = "cli-config.yml"
	cfg.TTS.Quiet = true
	cfg.TTS.Voice = "alice"

	opts := synthRunOptions{
		Out:       "-",
		Backend:   config.BackendCLI,
		Normalize: true,
	}

	var stdout bytes.Buffer

	err := runSynthCommand(context.Background(), cfg, opts, strings.NewReader(" hello from stdin "), &stdout, io.Discard)
	if err != nil {
		t.Fatalf("runSynthCommand returned error: %v", err)
	}

	if len(gotTexts) != 1 || gotTexts[0] != "hello from stdin" {
		t.Fatalf("unexpected synthesized chunks: %v", gotTexts)
	}

	decoded, err := audio.DecodeWAV(stdout.Bytes())
	if err != nil {
		t.Fatalf("DecodeWAV(stdout) error: %v", err)
	}

	if len(decoded) != 2 {
		t.Fatalf("unexpected sample count: got %d want %d", len(decoded), 2)
	}
}

func TestRunSynthCommand_InvalidBackend(t *testing.T) {
	cfg := config.DefaultConfig()

	err := runSynthCommand(
		context.Background(),
		cfg,
		synthRunOptions{Text: "hello", Out: "-", Backend: "invalid-backend"},
		strings.NewReader(""),
		&bytes.Buffer{},
		io.Discard,
	)
	if err == nil || !strings.Contains(err.Error(), "invalid backend") {
		t.Fatalf("expected invalid backend error, got: %v", err)
	}
}

func TestRunSynthCommand_MapsExecErrNotFound(t *testing.T) {
	orig := runChunkSynthesis
	t.Cleanup(func() { runChunkSynthesis = orig })

	runChunkSynthesis = func(_ context.Context, _ synthCLIOptions) ([]byte, error) {
		return nil, exec.ErrNotFound
	}

	cfg := config.DefaultConfig()
	cfg.TTS.Backend = config.BackendCLI

	err := runSynthCommand(
		context.Background(),
		cfg,
		synthRunOptions{Text: "hello", Out: "-", Backend: config.BackendCLI},
		strings.NewReader(""),
		&bytes.Buffer{},
		io.Discard,
	)
	if err == nil {
		t.Fatal("expected non-nil error")
	}

	if !errors.Is(err, exec.ErrNotFound) {
		t.Fatalf("expected ErrNotFound wrapping, got: %v", err)
	}

	if !strings.Contains(err.Error(), "executable not found") {
		t.Fatalf("expected mapped synth error message, got: %v", err)
	}
}

func TestSynthesizeForBackend_NativeRejectsTTSArg(t *testing.T) {
	cfg := config.DefaultConfig()

	_, err := synthesizeForBackend(
		context.Background(),
		cfg,
		config.BackendNative,
		"",
		[]string{"hello"},
		[]string{"temperature=0.7"},
		false,
		io.Discard,
	)
	if err == nil || !strings.Contains(err.Error(), "only supported with --backend cli") {
		t.Fatalf("expected native tts-arg error, got: %v", err)
	}
}

func TestSynthesizeForBackend_NativePathCallsNativeSynthesize(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Paths.TokenizerModel = filepath.Join(t.TempDir(), "missing-tokenizer.model")

	_, err := synthesizeForBackend(
		context.Background(),
		cfg,
		config.BackendNative,
		filepath.Join("voices", "alice.safetensors"),
		[]string{"hello"},
		nil,
		false,
		io.Discard,
	)
	if err == nil {
		t.Fatal("expected native synth init error")
	}

	if !strings.Contains(err.Error(), "initialize native synth service") {
		t.Fatalf("expected native synth error context, got: %v", err)
	}
}

func TestSynthesizeForBackend_UnsupportedBackend(t *testing.T) {
	cfg := config.DefaultConfig()

	_, err := synthesizeForBackend(
		context.Background(),
		cfg,
		"unsupported",
		"",
		[]string{"hello"},
		nil,
		false,
		io.Discard,
	)
	if err == nil || !strings.Contains(err.Error(), "unsupported backend") {
		t.Fatalf("expected unsupported backend error, got: %v", err)
	}
}

func TestSynthesizeNative_ErrorsOnInvalidConfig(t *testing.T) {
	_, err := synthesizeNative(context.Background(), config.Config{}, []string{"hello"}, "")
	if err == nil || !strings.Contains(err.Error(), "initialize native synth service") {
		t.Fatalf("expected native init error, got: %v", err)
	}
}

func TestResolveVoiceForNative(t *testing.T) {
	t.Run("empty voice returns empty", func(t *testing.T) {
		got, err := resolveVoiceForNative("")
		if err != nil {
			t.Fatalf("resolveVoiceForNative returned error: %v", err)
		}

		if got != "" {
			t.Fatalf("expected empty voice path, got %q", got)
		}
	})

	t.Run("path-like voice returns as-is", func(t *testing.T) {
		in := filepath.Join("voices", "alice.safetensors")
		got, err := resolveVoiceForNative(in)
		if err != nil {
			t.Fatalf("resolveVoiceForNative returned error: %v", err)
		}

		if got != in {
			t.Fatalf("expected %q, got %q", in, got)
		}
	})

	t.Run("missing manifest returns empty", func(t *testing.T) {
		origWD := mustGetwd(t)
		tmp := t.TempDir()
		mustChdir(t, tmp)
		t.Cleanup(func() { mustChdir(t, origWD) })

		got, err := resolveVoiceForNative("alice")
		if err != nil {
			t.Fatalf("resolveVoiceForNative returned error: %v", err)
		}

		if got != "" {
			t.Fatalf("expected empty voice path when manifest missing, got %q", got)
		}
	})

	t.Run("known and unknown IDs in manifest", func(t *testing.T) {
		origWD := mustGetwd(t)
		tmp := t.TempDir()
		mustChdir(t, tmp)
		t.Cleanup(func() { mustChdir(t, origWD) })

		voiceDir := filepath.Join(tmp, "voices")
		err := os.MkdirAll(voiceDir, 0o755)
		if err != nil {
			t.Fatalf("mkdir voices dir: %v", err)
		}

		err = os.WriteFile(filepath.Join(voiceDir, "alice.bin"), []byte("voice"), 0o644)
		if err != nil {
			t.Fatalf("write voice file: %v", err)
		}

		manifest := `{"voices":[{"id":"alice","path":"alice.bin","license":"MIT"}]}`
		err = os.WriteFile(filepath.Join(voiceDir, "manifest.json"), []byte(manifest), 0o644)
		if err != nil {
			t.Fatalf("write manifest: %v", err)
		}

		gotKnown, err := resolveVoiceForNative("alice")
		if err != nil {
			t.Fatalf("resolve known voice failed: %v", err)
		}

		wantKnown := filepath.Join("voices", "alice.bin")
		if gotKnown != wantKnown {
			t.Fatalf("expected %q, got %q", wantKnown, gotKnown)
		}

		gotUnknown, err := resolveVoiceForNative("bob")
		if err != nil {
			t.Fatalf("resolve unknown voice failed: %v", err)
		}

		if gotUnknown != "" {
			t.Fatalf("expected empty path for unknown ID, got %q", gotUnknown)
		}
	})
}

func mustGetwd(t *testing.T) string {
	t.Helper()

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}

	return wd
}

func mustChdir(t *testing.T, dir string) {
	t.Helper()

	err := os.Chdir(dir)
	if err != nil {
		t.Fatalf("Chdir(%s): %v", dir, err)
	}
}
