package doctor_test

import (
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/doctor"
)

func nativeOnly(cfg doctor.Config) doctor.Config {
	cfg.SkipPocketTTS = true
	cfg.SkipPython = true

	return cfg
}

func TestRun_PrintsLanguage(t *testing.T) {
	var out strings.Builder

	doctor.Run(nativeOnly(doctor.Config{Language: "german"}), &out)

	if !strings.Contains(out.String(), "language: german") {
		t.Errorf("output should name the language; got:\n%s", out.String())
	}
}

func TestRun_VoiceManifestMissingFails(t *testing.T) {
	var out strings.Builder

	result := doctor.Run(nativeOnly(doctor.Config{
		VoiceManifestPath: "/nonexistent/voices/german/manifest.json",
		DefaultVoice:      "juergen",
		ResolveVoice:      func(string) (string, error) { t.Error("ResolveVoice called without a manifest"); return "", nil },
	}), &out)

	if !hasFailureContaining(result.Failures(), "voice manifest") || !hasFailureContaining(result.Failures(), "model download") {
		t.Errorf("want a voice manifest failure with a download hint; got %v", result.Failures())
	}
}

func TestRun_DefaultVoiceUnresolvableFails(t *testing.T) {
	var out strings.Builder

	result := doctor.Run(nativeOnly(doctor.Config{
		VoiceManifestPath: "doctor_test.go", // exists
		DefaultVoice:      "juergen",
		ResolveVoice:      func(string) (string, error) { return "", sentinelError(`unknown voice id "juergen"`) },
	}), &out)

	if !hasFailureContaining(result.Failures(), `default voice "juergen"`) {
		t.Errorf("want a default voice failure; got %v", result.Failures())
	}
}

func TestRun_DefaultVoicePasses(t *testing.T) {
	var out strings.Builder

	result := doctor.Run(nativeOnly(doctor.Config{
		VoiceManifestPath: "doctor_test.go",
		DefaultVoice:      "juergen",
		ResolveVoice: func(id string) (string, error) {
			if id != "juergen" {
				t.Errorf("ResolveVoice(%q); want juergen", id)
			}

			return "voices/german/juergen.safetensors", nil
		},
	}), &out)

	if result.Failed() {
		t.Errorf("expected pass; failures: %v", result.Failures())
	}

	if !strings.Contains(out.String(), "default voice juergen: voices/german/juergen.safetensors") {
		t.Errorf("output should show the resolved default voice; got:\n%s", out.String())
	}
}

func TestRun_TokenizerLoadFails(t *testing.T) {
	var out strings.Builder

	result := doctor.Run(nativeOnly(doctor.Config{
		TokenizerModelPath: "doctor_test.go",
		LoadTokenizer:      func(string) error { return sentinelError("vocab size=4000 but n_bins=3999") },
	}), &out)

	if !hasFailureContaining(result.Failures(), "n_bins=3999") {
		t.Errorf("want the tokenizer load error; got %v", result.Failures())
	}
}

func TestRun_TokenizerLoadSkippedWhenMissing(t *testing.T) {
	var out strings.Builder

	result := doctor.Run(nativeOnly(doctor.Config{
		TokenizerModelPath: "/nonexistent/tokenizer.json",
		LoadTokenizer:      func(string) error { t.Error("LoadTokenizer called for a missing file"); return nil },
	}), &out)

	if len(result.Failures()) != 1 {
		t.Errorf("want only the not-found failure; got %v", result.Failures())
	}
}

func TestRun_TokenizerLoadPasses(t *testing.T) {
	var out strings.Builder

	result := doctor.Run(nativeOnly(doctor.Config{
		TokenizerModelPath: "doctor_test.go",
		LoadTokenizer:      func(string) error { return nil },
	}), &out)

	if result.Failed() {
		t.Errorf("expected pass; failures: %v", result.Failures())
	}

	if !strings.Contains(out.String(), "tokenizer load: ok") {
		t.Errorf("output should contain 'tokenizer load: ok'; got:\n%s", out.String())
	}
}
