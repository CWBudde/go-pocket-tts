package pockettts_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	pockettts "github.com/cwbudde/go-pocket-tts"
)

const testModel = "english_2026-01"

// testRoot lays out the flat english checkpoint, tokenizer and alba voice
// downloaded into this repo as a model root, skipping without them.
func testRoot(t *testing.T) string {
	t.Helper()

	m, err := pockettts.LookupModel(testModel)
	if err != nil {
		t.Fatal(err)
	}

	alba, _ := m.Voice("alba")
	root := t.TempDir()

	for src, f := range map[string]pockettts.File{
		filepath.Join("models", "tts_b6369a24.safetensors"): m.Weights,
		filepath.Join("models", "tokenizer.model"):          m.Tokenizer,
		filepath.Join("voices", "alba.safetensors"):         alba.File,
	} {
		abs, err := filepath.Abs(src)
		if err != nil {
			t.Fatal(err)
		}

		_, err = os.Stat(abs)
		if err != nil {
			t.Skipf("model files unavailable (run pockettts model download): %v", err)
		}

		dst := filepath.Join(root, filepath.FromSlash(f.Path))

		err = os.MkdirAll(filepath.Dir(dst), 0o755)
		if err != nil {
			t.Fatal(err)
		}

		err = os.Symlink(abs, dst)
		if err != nil {
			t.Fatal(err)
		}
	}

	return root
}

var (
	engineOnce sync.Once
	engine     *pockettts.Engine
	errEngine  error
)

// testEngine loads the test model once for all tests of the package.
func testEngine(t *testing.T) (*pockettts.Engine, *pockettts.Voice) {
	t.Helper()

	root := testRoot(t)

	engineOnce.Do(func() { engine, errEngine = pockettts.LoadDir(root, testModel) })

	if errEngine != nil {
		t.Fatal(errEngine)
	}

	voice, err := pockettts.LoadVoiceDir(root, testModel, "alba")
	if err != nil {
		t.Fatal(err)
	}

	return engine, voice
}

func TestSynthesizeIsReproducibleForASeed(t *testing.T) {
	e, voice := testEngine(t)
	opts := e.DefaultOptions()
	opts.Seed = 42

	a, err := e.Synthesize(context.Background(), "Hello world. This is a test.", voice, opts)
	if err != nil {
		t.Fatal(err)
	}

	b, err := e.Synthesize(context.Background(), "Hello world. This is a test.", voice, opts)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(a, b) {
		t.Fatal("same seed produced different audio")
	}

	if len(a) < pockettts.SampleRate/2 {
		t.Fatalf("got %d samples; want at least half a second", len(a))
	}

	opts.Seed = 43

	c, err := e.Synthesize(context.Background(), "Hello world. This is a test.", voice, opts)
	if err != nil {
		t.Fatal(err)
	}

	if slices.Equal(a, c) {
		t.Fatal("different seeds produced identical audio")
	}
}

func TestSynthesizeIsReproducibleAtAWorkerCount(t *testing.T) {
	e, voice := testEngine(t)
	opts := e.DefaultOptions()
	opts.Seed = 7

	t.Cleanup(func() { pockettts.SetWorkers(1) })

	pockettts.SetWorkers(4)

	a, err := e.Synthesize(context.Background(), "Workers.", voice, opts)
	if err != nil {
		t.Fatal(err)
	}

	b, err := e.Synthesize(context.Background(), "Workers.", voice, opts)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(a, b) {
		t.Fatal("same seed and worker count produced different audio")
	}
}

func TestSynthesizeReportsChunkProgressAndCancels(t *testing.T) {
	e, voice := testEngine(t)
	ctx, cancel := context.WithCancel(context.Background())
	opts := e.DefaultOptions()

	var seen []pockettts.Progress

	opts.Progress = func(p pockettts.Progress) {
		seen = append(seen, p)
		if p.Chunk == 2 && p.Step == 3 {
			cancel()
		}
	}

	text := strings.Repeat("This sentence is long enough to need its own chunk of tokens. ", 3)

	_, err := e.Synthesize(ctx, text, voice, opts)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Synthesize after cancel = %v; want context.Canceled", err)
	}

	last := seen[len(seen)-1]
	if last.Chunk != 2 || last.Chunks < 2 || last.Step > 4 || last.MaxSteps <= 0 {
		t.Fatalf("last progress %+v; want chunk 2 of several, stopped within a step of 3", last)
	}
}

func TestOptionsAreValidated(t *testing.T) {
	e, voice := testEngine(t)

	for name, mutate := range map[string]func(*pockettts.Options){
		"negative temperature": func(o *pockettts.Options) { o.Temperature = -0.1 },
		"huge temperature":     func(o *pockettts.Options) { o.Temperature = 3 },
		"zero sampler steps":   func(o *pockettts.Options) { o.SamplerSteps = 0 },
		"many sampler steps":   func(o *pockettts.Options) { o.SamplerSteps = 65 },
	} {
		t.Run(name, func(t *testing.T) {
			opts := e.DefaultOptions()
			mutate(&opts)

			_, err := e.Synthesize(context.Background(), "Hello.", voice, opts)
			if err == nil {
				t.Fatal("invalid options accepted")
			}
		})
	}

	_, err := e.Synthesize(context.Background(), "  \n ", voice, e.DefaultOptions())
	if err == nil {
		t.Fatal("empty text accepted")
	}
}

func TestLoadDirReportsMissingAndTruncatedFiles(t *testing.T) {
	root := t.TempDir()

	_, err := pockettts.LoadDir(root, testModel)
	if err == nil || !strings.Contains(err.Error(), "download the model first") {
		t.Fatalf("LoadDir(empty root) = %v; want a download hint", err)
	}

	m, _ := pockettts.LookupModel(testModel)
	tok := filepath.Join(root, filepath.FromSlash(m.Tokenizer.Path))

	err = os.MkdirAll(filepath.Dir(tok), 0o755)
	if err != nil {
		t.Fatal(err)
	}

	err = os.WriteFile(tok, []byte("short"), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	_, err = pockettts.LoadDir(root, testModel)
	if err == nil || !strings.Contains(err.Error(), "download the model again") {
		t.Fatalf("LoadDir(truncated tokenizer) = %v; want a re-download hint", err)
	}

	_, err = pockettts.LoadVoiceDir(root, testModel, "nobody")
	if err == nil {
		t.Fatal("LoadVoiceDir(unknown voice) = nil")
	}
}

func TestParseVoiceRejectsGarbage(t *testing.T) {
	_, err := pockettts.ParseVoice([]byte("not safetensors"))
	if err == nil {
		t.Fatal("ParseVoice(garbage) = nil")
	}
}
