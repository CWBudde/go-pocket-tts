package doctor_test

import (
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/doctor"
)

// ---------------------------------------------------------------------------
// all-pass scenario
// ---------------------------------------------------------------------------

func TestRun_AllChecksPass(t *testing.T) {
	cfg := doctor.Config{
		PocketTTSVersion: func() (string, error) { return "1.2.3", nil },
		PythonVersion:    func() (string, error) { return "3.11.4", nil },
		VoiceFiles:       []string{},
	}

	var out strings.Builder
	result := doctor.Run(cfg, &out)

	if result.Failed() {
		t.Errorf("expected all checks to pass; failures: %v", result.Failures())
	}

	if !strings.Contains(out.String(), "pocket-tts") {
		t.Error("output should mention pocket-tts")
	}
}

// ---------------------------------------------------------------------------
// pocket-tts binary missing
// ---------------------------------------------------------------------------

func TestRun_PocketTTSMissingFails(t *testing.T) {
	cfg := doctor.Config{
		PocketTTSVersion: func() (string, error) { return "", errBinaryNotFound },
		PythonVersion:    func() (string, error) { return "3.11.4", nil },
		VoiceFiles:       []string{},
	}

	var out strings.Builder
	result := doctor.Run(cfg, &out)

	if !result.Failed() {
		t.Fatal("expected failure when pocket-tts is not found")
	}

	if !hasFailureContaining(result.Failures(), "pocket-tts") {
		t.Errorf("expected failure mentioning pocket-tts, got: %v", result.Failures())
	}
}

// ---------------------------------------------------------------------------
// Python version out of range
// ---------------------------------------------------------------------------

func TestRun_PythonTooOldFails(t *testing.T) {
	cfg := doctor.Config{
		PocketTTSVersion: func() (string, error) { return "1.0.0", nil },
		PythonVersion:    func() (string, error) { return "3.9.7", nil },
		VoiceFiles:       []string{},
	}

	var out strings.Builder
	result := doctor.Run(cfg, &out)

	if !result.Failed() {
		t.Fatal("expected failure for Python 3.9 (< 3.10)")
	}

	if !hasFailureContaining(result.Failures(), "python") {
		t.Errorf("expected failure mentioning python, got: %v", result.Failures())
	}
}

func TestRun_PythonTooNewFails(t *testing.T) {
	cfg := doctor.Config{
		PocketTTSVersion: func() (string, error) { return "1.0.0", nil },
		PythonVersion:    func() (string, error) { return "3.15.0", nil },
		VoiceFiles:       []string{},
	}

	var out strings.Builder
	result := doctor.Run(cfg, &out)

	if !result.Failed() {
		t.Fatal("expected failure for Python 3.15 (>= 3.15)")
	}
}

func TestRun_PythonInRangePasses(t *testing.T) {
	for _, ver := range []string{"3.10.0", "3.11.9", "3.12.0", "3.14.1"} {
		t.Run(ver, func(t *testing.T) {
			cfg := doctor.Config{
				PocketTTSVersion: func() (string, error) { return "1.0.0", nil },
				PythonVersion:    func() (string, error) { return ver, nil },
				VoiceFiles:       []string{},
			}
			var out strings.Builder

			result := doctor.Run(cfg, &out)
			if result.Failed() {
				t.Errorf("Python %s should pass but got failures: %v", ver, result.Failures())
			}
		})
	}
}

// ---------------------------------------------------------------------------
// voice file existence
// ---------------------------------------------------------------------------

func TestRun_MissingVoiceFileFails(t *testing.T) {
	cfg := doctor.Config{
		PocketTTSVersion: func() (string, error) { return "1.0.0", nil },
		PythonVersion:    func() (string, error) { return "3.11.4", nil },
		VoiceFiles:       []string{"/nonexistent/voice.safetensors"},
	}

	var out strings.Builder
	result := doctor.Run(cfg, &out)

	if !result.Failed() {
		t.Fatal("expected failure for missing voice file")
	}

	if !hasFailureContaining(result.Failures(), "voice") {
		t.Errorf("expected failure mentioning voice, got: %v", result.Failures())
	}
}

// ---------------------------------------------------------------------------
// colour-coded output
// ---------------------------------------------------------------------------

func TestRun_OutputContainsPassAndFailMarkers(t *testing.T) {
	cfg := doctor.Config{
		PocketTTSVersion: func() (string, error) { return "", errBinaryNotFound },
		PythonVersion:    func() (string, error) { return "3.11.0", nil },
		VoiceFiles:       []string{},
	}

	var out strings.Builder
	doctor.Run(cfg, &out)

	body := out.String()
	if !strings.Contains(body, doctor.PassMark) {
		t.Errorf("output missing pass marker %q:\n%s", doctor.PassMark, body)
	}

	if !strings.Contains(body, doctor.FailMark) {
		t.Errorf("output missing fail marker %q:\n%s", doctor.FailMark, body)
	}
}

func TestRun_SkipRuntimeChecks(t *testing.T) {
	cfg := doctor.Config{
		SkipPocketTTS: true,
		SkipPython:    true,
		VoiceFiles:    []string{},
	}

	var out strings.Builder

	result := doctor.Run(cfg, &out)
	if result.Failed() {
		t.Fatalf("expected no failures when runtime checks are skipped, got: %v", result.Failures())
	}

	body := out.String()
	if !strings.Contains(body, "pocket-tts binary: skipped") {
		t.Fatalf("expected pocket-tts skipped output, got:\n%s", body)
	}

	if !strings.Contains(body, "python version: skipped") {
		t.Fatalf("expected python skipped output, got:\n%s", body)
	}
}

// ---------------------------------------------------------------------------
// native model checks
// ---------------------------------------------------------------------------

func TestRun_NativeModelPresent(t *testing.T) {
	// Use a file we know exists (the test file itself).
	cfg := doctor.Config{
		SkipPocketTTS:   true,
		SkipPython:      true,
		NativeModelPath: "doctor_test.go",
	}

	var out strings.Builder

	result := doctor.Run(cfg, &out)
	if result.Failed() {
		t.Errorf("expected pass; failures: %v", result.Failures())
	}

	if !strings.Contains(out.String(), "safetensors model: doctor_test.go") {
		t.Errorf("output should mention safetensors model; got:\n%s", out.String())
	}
}

func TestRun_NativeModelMissing(t *testing.T) {
	cfg := doctor.Config{
		SkipPocketTTS:   true,
		SkipPython:      true,
		NativeModelPath: "/nonexistent/model.safetensors",
	}

	var out strings.Builder

	result := doctor.Run(cfg, &out)
	if !result.Failed() {
		t.Fatal("expected failure for missing safetensors model")
	}

	if !hasFailureContaining(result.Failures(), "safetensors") {
		t.Errorf("expected failure mentioning safetensors, got: %v", result.Failures())
	}
}

func TestRun_TokenizerModelMissing(t *testing.T) {
	cfg := doctor.Config{
		SkipPocketTTS:      true,
		SkipPython:         true,
		TokenizerModelPath: "/nonexistent/tokenizer.model",
	}

	var out strings.Builder

	result := doctor.Run(cfg, &out)
	if !result.Failed() {
		t.Fatal("expected failure for missing tokenizer model")
	}

	if !hasFailureContaining(result.Failures(), "tokenizer") {
		t.Errorf("expected failure mentioning tokenizer, got: %v", result.Failures())
	}
}

func TestRun_ValidateSafetensorsCallback(t *testing.T) {
	cfg := doctor.Config{
		SkipPocketTTS:   true,
		SkipPython:      true,
		NativeModelPath: "doctor_test.go", // exists
		ValidateSafetensors: func(_ string) error {
			return sentinelError("bad keys")
		},
	}

	var out strings.Builder

	result := doctor.Run(cfg, &out)
	if !result.Failed() {
		t.Fatal("expected failure from validation callback")
	}

	if !hasFailureContaining(result.Failures(), "validation") {
		t.Errorf("expected failure mentioning validation, got: %v", result.Failures())
	}
}

func TestRun_ValidateSafetensorsPassesOnSuccess(t *testing.T) {
	cfg := doctor.Config{
		SkipPocketTTS:   true,
		SkipPython:      true,
		NativeModelPath: "doctor_test.go",
		ValidateSafetensors: func(_ string) error {
			return nil
		},
	}

	var out strings.Builder

	result := doctor.Run(cfg, &out)
	if result.Failed() {
		t.Errorf("expected pass; failures: %v", result.Failures())
	}

	if !strings.Contains(out.String(), "validation: ok") {
		t.Errorf("output should contain 'validation: ok'; got:\n%s", out.String())
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

type sentinelError string

func (e sentinelError) Error() string { return string(e) }

var errBinaryNotFound = sentinelError("binary not found")

func hasFailureContaining(failures []string, substr string) bool {
	substr = strings.ToLower(substr)
	for _, f := range failures {
		if strings.Contains(strings.ToLower(f), substr) {
			return true
		}
	}

	return false
}
