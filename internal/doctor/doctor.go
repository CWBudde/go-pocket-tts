// Package doctor provides environment preflight checks for pockettts.
package doctor

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// PassMark and FailMark are the prefix symbols printed for each check result.
const (
	PassMark = "✓"
	FailMark = "✗"
)

// VersionFunc returns a version string or an error if the component is unavailable.
type VersionFunc func() (string, error)

// Config holds injectable dependencies for each doctor check.
type Config struct {
	// PocketTTSVersion returns the output of `pocket-tts --version`.
	PocketTTSVersion VersionFunc
	// SkipPocketTTS skips pocket-tts version check (native backend mode).
	SkipPocketTTS bool
	// PythonVersion returns the Python version string (e.g. "3.11.4").
	PythonVersion VersionFunc
	// SkipPython skips Python version check (native backend mode).
	SkipPython bool
	// VoiceFiles is the list of voice file paths to verify on disk.
	VoiceFiles []string

	// NativeModelPath, if non-empty, is checked for existence (native-safetensors backend).
	NativeModelPath string
	// TokenizerModelPath, if non-empty, is checked for existence.
	TokenizerModelPath string
	// ValidateSafetensors, if set, is called to validate the model file contents.
	ValidateSafetensors func(path string) error
	// LoadTokenizer, if set, loads the tokenizer once it exists, for example
	// to check its vocab size against the model config's n_bins.
	LoadTokenizer func(path string) error

	// Language, if non-empty, names the model config whose files are checked.
	Language string
	// VoiceManifestPath, if non-empty, must exist: native synthesis needs a voice.
	VoiceManifestPath string
	// DefaultVoice, if non-empty, must resolve through ResolveVoice once the
	// voice manifest exists; synthesis without --voice uses it.
	DefaultVoice string
	// ResolveVoice maps a voice ID of the voice manifest to its file.
	ResolveVoice func(id string) (string, error)
}

// Result collects the outcome of all checks.
type Result struct {
	failures []string
}

// Failed returns true if any check failed.
func (r *Result) Failed() bool { return len(r.failures) > 0 }

// Failures returns the list of failure messages.
func (r *Result) Failures() []string { return append([]string(nil), r.failures...) }

// AddFailure appends an external failure message to the result.
func (r *Result) AddFailure(msg string) { r.failures = append(r.failures, msg) }

func (r *Result) fail(msg string) { r.failures = append(r.failures, msg) }

// Run executes all configured checks and writes human-readable output to w.
// Each check line is prefixed with PassMark or FailMark.
func Run(cfg Config, w io.Writer) Result {
	var res Result

	if cfg.Language != "" {
		_, _ = fmt.Fprintf(w, "language: %s\n", cfg.Language)
	}

	// ---- pocket-tts binary ------------------------------------------------
	if cfg.SkipPocketTTS {
		_, _ = fmt.Fprintf(w, "%s pocket-tts binary: skipped\n", PassMark)
	} else {
		ver, err := cfg.PocketTTSVersion()
		if err != nil {
			res.fail(fmt.Sprintf("pocket-tts binary: %v", err))
			_, _ = fmt.Fprintf(w, "%s pocket-tts binary: not found (%v)\n", FailMark, err)
		} else {
			_, _ = fmt.Fprintf(w, "%s pocket-tts binary: %s\n", PassMark, ver)
		}
	}

	// ---- Python version ---------------------------------------------------
	//nolint:nestif // Preserve detailed staged reporting for python availability and version compatibility.
	if cfg.SkipPython {
		_, _ = fmt.Fprintf(w, "%s python version: skipped\n", PassMark)
	} else {
		pyVer, pyGetErr := cfg.PythonVersion()
		if pyGetErr != nil {
			res.fail(fmt.Sprintf("python version: %v", pyGetErr))
			_, _ = fmt.Fprintf(w, "%s python version: not found (%v)\n", FailMark, pyGetErr)
		} else {
			pyCheckErr := checkPythonVersion(pyVer)
			if pyCheckErr != nil {
				res.fail(fmt.Sprintf("python version: %v", pyCheckErr))
				_, _ = fmt.Fprintf(w, "%s python version %s: %v\n", FailMark, pyVer, pyCheckErr)
			} else {
				_, _ = fmt.Fprintf(w, "%s python version: %s\n", PassMark, pyVer)
			}
		}
	}

	checkVoiceManifest(cfg, w, &res)

	// ---- voice files ------------------------------------------------------
	for _, path := range cfg.VoiceFiles {
		_, statErr := os.Stat(path)
		if statErr != nil {
			res.fail(fmt.Sprintf("voice file %q: %v", path, statErr))
			_, _ = fmt.Fprintf(w, "%s voice file %s: not found\n", FailMark, path)
		} else {
			_, _ = fmt.Fprintf(w, "%s voice file: %s\n", PassMark, path)
		}
	}

	// ---- native safetensors model -----------------------------------------
	//nolint:nestif // Keep model existence and optional validation checks grouped in doctor output order.
	if cfg.NativeModelPath != "" {
		_, statErr := os.Stat(cfg.NativeModelPath)
		if statErr != nil {
			res.fail(fmt.Sprintf("safetensors model %q: not found", cfg.NativeModelPath))
			_, _ = fmt.Fprintf(w, "%s safetensors model: not found (%s)\n", FailMark, cfg.NativeModelPath)
		} else {
			_, _ = fmt.Fprintf(w, "%s safetensors model: %s\n", PassMark, cfg.NativeModelPath)

			if cfg.ValidateSafetensors != nil {
				valErr := cfg.ValidateSafetensors(cfg.NativeModelPath)
				if valErr != nil {
					res.fail(fmt.Sprintf("safetensors model validation: %v", valErr))
					_, _ = fmt.Fprintf(w, "%s safetensors model validation: %v\n", FailMark, valErr)
				} else {
					_, _ = fmt.Fprintf(w, "%s safetensors model validation: ok\n", PassMark)
				}
			}
		}
	}

	checkTokenizer(cfg, w, &res)

	return res
}

// checkTokenizer checks that the tokenizer exists and, with LoadTokenizer,
// that it loads.
func checkTokenizer(cfg Config, w io.Writer, res *Result) {
	if cfg.TokenizerModelPath == "" {
		return
	}

	_, statErr := os.Stat(cfg.TokenizerModelPath)
	if statErr != nil {
		res.fail(fmt.Sprintf("tokenizer model %q: not found", cfg.TokenizerModelPath))
		_, _ = fmt.Fprintf(w, "%s tokenizer model: not found (%s)\n", FailMark, cfg.TokenizerModelPath)

		return
	}

	_, _ = fmt.Fprintf(w, "%s tokenizer model: %s\n", PassMark, cfg.TokenizerModelPath)

	if cfg.LoadTokenizer == nil {
		return
	}

	loadErr := cfg.LoadTokenizer(cfg.TokenizerModelPath)
	if loadErr != nil {
		res.fail(fmt.Sprintf("tokenizer load: %v", loadErr))
		_, _ = fmt.Fprintf(w, "%s tokenizer load: %v\n", FailMark, loadErr)

		return
	}

	_, _ = fmt.Fprintf(w, "%s tokenizer load: ok\n", PassMark)
}

// checkVoiceManifest checks that the voice manifest exists and that the
// default voice resolves through it.
func checkVoiceManifest(cfg Config, w io.Writer, res *Result) {
	if cfg.VoiceManifestPath == "" {
		return
	}

	_, statErr := os.Stat(cfg.VoiceManifestPath)
	if statErr != nil {
		res.fail(fmt.Sprintf("voice manifest %q: not found; run 'pockettts model download' (same --language)",
			cfg.VoiceManifestPath))
		_, _ = fmt.Fprintf(w, "%s voice manifest: not found (%s)\n", FailMark, cfg.VoiceManifestPath)

		return
	}

	_, _ = fmt.Fprintf(w, "%s voice manifest: %s\n", PassMark, cfg.VoiceManifestPath)

	if cfg.DefaultVoice == "" || cfg.ResolveVoice == nil {
		return
	}

	path, err := cfg.ResolveVoice(cfg.DefaultVoice)
	if err != nil {
		res.fail(fmt.Sprintf("default voice %q: %v; run 'pockettts model download' (same --language)", cfg.DefaultVoice, err))
		_, _ = fmt.Fprintf(w, "%s default voice %s: %v\n", FailMark, cfg.DefaultVoice, err)

		return
	}

	_, _ = fmt.Fprintf(w, "%s default voice %s: %s\n", PassMark, cfg.DefaultVoice, path)
}

// checkPythonVersion returns an error if ver is outside [3.10, 3.15).
// ver is expected to be a string like "3.11.4".
func checkPythonVersion(ver string) error {
	major, minor, err := parseMajorMinor(ver)
	if err != nil {
		return fmt.Errorf("cannot parse %q: %w", ver, err)
	}

	if major != 3 {
		return fmt.Errorf("requires Python 3, got %d", major)
	}

	if minor < 10 {
		return fmt.Errorf("requires Python >=3.10, got 3.%d", minor)
	}

	if minor >= 15 {
		return fmt.Errorf("requires Python <3.15, got 3.%d", minor)
	}

	return nil
}

func parseMajorMinor(ver string) (major, minor int, err error) {
	parts := strings.SplitN(ver, ".", 3)
	if len(parts) < 2 {
		return 0, 0, fmt.Errorf("unexpected version format %q", ver)
	}

	major, err = strconv.Atoi(parts[0])
	if err != nil {
		return 0, 0, fmt.Errorf("bad major in %q: %w", ver, err)
	}

	minor, err = strconv.Atoi(parts[1])
	if err != nil {
		return 0, 0, fmt.Errorf("bad minor in %q: %w", ver, err)
	}

	return major, minor, nil
}
