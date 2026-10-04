package server

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
)

// writeGermanLayout creates the PathsForLanguage files of german under the
// working directory: a voice manifest with juergen and placeholder model and
// tokenizer files, which languageSpecs only stats.
func writeGermanLayout(t *testing.T) {
	t.Helper()

	t.Chdir(t.TempDir())

	for _, dir := range []string{"voices/german", "models/german"} {
		err := os.MkdirAll(dir, 0o750)
		if err != nil {
			t.Fatal(err)
		}
	}

	writeTestVoice(t, "voices/german/juergen.safetensors")

	files := map[string]string{
		"voices/german/manifest.json":     `{"voices":[{"id":"juergen","path":"juergen.safetensors","license":"CC-BY-4.0"}]}`,
		"models/german/model.safetensors": "weights",
		"models/german/tokenizer.json":    "{}",
	}
	for name, data := range files {
		err := os.WriteFile(name, []byte(data), 0o600)
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestLanguageSpecs(t *testing.T) {
	writeGermanLayout(t)

	native := config.BackendNative

	for name, tc := range map[string]struct {
		backend   string
		languages []string
		wantKeys  []string
	}{
		"startup only":                {native, nil, nil},
		"startup, blank and repeats":  {native, []string{"english_2026-01", " ", "english_2026-01"}, nil},
		"german":                      {native, []string{"german", " german "}, []string{"german"}},
		"startup only on cli":         {config.BackendCLI, []string{"english_2026-01"}, nil},
		"startup only on native-onnx": {config.BackendNativeONNX, []string{"english_2026-01"}, nil},
	} {
		cfg := config.DefaultConfig()
		cfg.Server.Languages = tc.languages

		startup, specs, err := New(cfg, nil).languageSpecs(tc.backend)
		if err != nil {
			t.Errorf("%s: languageSpecs error = %v", name, err)
			continue
		}

		if startup != "english_2026-01" || len(specs) != len(tc.wantKeys) {
			t.Errorf("%s: languageSpecs = %q, %d specs; want english_2026-01 and %v", name, startup, len(specs), tc.wantKeys)
		}

		for _, key := range tc.wantKeys {
			if _, ok := specs[key]; !ok {
				t.Errorf("%s: no spec for %s", name, key)
			}
		}
	}

	// The other language's voices come from its own manifest, without a model.
	cfg := config.DefaultConfig()
	cfg.Server.Languages = []string{"german"}

	_, specs, err := New(cfg, nil).languageSpecs(native)
	if err != nil {
		t.Fatal(err)
	}

	voices := specs["german"].voices.ListVoices()
	if len(voices) != 1 || voices[0].ID != "juergen" {
		t.Errorf("german voices = %v; want juergen", voices)
	}
}

func TestLanguageSpecs_CustomModelHasNoName(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TTS.ModelConfigPath = "custom.yaml"

	startup, specs, err := New(cfg, nil).languageSpecs(config.BackendNative)
	if err != nil || startup != "" || len(specs) != 0 {
		t.Errorf("languageSpecs = %q, %v, %v; want the unnamed startup model only", startup, specs, err)
	}
}

func TestLanguageSpecs_Errors(t *testing.T) {
	writeGermanLayout(t)

	native := config.BackendNative

	for name, tc := range map[string]struct {
		backend   string
		languages []string
		edit      func(*config.Config)
		errSub    string
	}{
		"unknown language":      {native, []string{"klingon"}, nil, "unknown language"},
		"cli":                   {config.BackendCLI, []string{"german"}, nil, "needs the native-safetensors backend"},
		"native-onnx":           {config.BackendNativeONNX, []string{"german"}, nil, "needs the native-safetensors backend"},
		"max 0":                 {native, nil, func(c *config.Config) { c.Server.MaxLanguages = 0 }, "at least 1"},
		"custom model":          {native, []string{"german"}, func(c *config.Config) { c.TTS.ModelConfigPath = "x.yaml" }, "--model-config"},
		"missing model":         {native, []string{"german"}, removeFile("models/german/model.safetensors"), "model download --language german"},
		"missing tokenizer":     {native, []string{"german"}, removeFile("models/german/tokenizer.json"), "model download --language german"},
		"missing manifest":      {native, []string{"german"}, removeFile("voices/german/manifest.json"), "model download --language german"},
		"missing default voice": {native, []string{"german"}, removeFile("voices/german/juergen.safetensors"), "model download --language german"},
	} {
		t.Run(name, func(t *testing.T) {
			writeGermanLayout(t)

			cfg := config.DefaultConfig()
			cfg.Server.Languages = tc.languages

			if tc.edit != nil {
				tc.edit(&cfg)
			}

			_, _, err := New(cfg, nil).languageSpecs(tc.backend)
			if err == nil || !strings.Contains(err.Error(), tc.errSub) {
				t.Errorf("languageSpecs error = %v; want one containing %q", err, tc.errSub)
			}
		})
	}
}

// removeFile returns a config edit that deletes name instead, to break one
// file of the language layout.
func removeFile(name string) func(*config.Config) {
	return func(*config.Config) { _ = os.Remove(name) }
}

func TestStart_OtherLanguageMissingAssetsFailsFast(t *testing.T) {
	t.Chdir(t.TempDir())

	cfg := config.DefaultConfig()
	cfg.Server.ListenAddr = "127.0.0.1:0"
	cfg.Server.Languages = []string{"german"}
	cfg.Paths.VoiceManifest = filepath.Join(t.TempDir(), "absent.json")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// The startup language is broken too; the other language's check comes
	// first because it loads no weights.
	err := New(cfg, &tts.Service{}).Start(ctx)
	if err == nil || !strings.Contains(err.Error(), "model download --language german") {
		t.Fatalf("Start error = %v; want the german download hint", err)
	}

	if ctx.Err() != nil {
		t.Fatal("Start blocked until the context ended; want an immediate error")
	}
}
