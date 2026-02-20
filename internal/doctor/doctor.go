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

	// ---- pocket-tts binary ------------------------------------------------
	if cfg.SkipPocketTTS {
		fmt.Fprintf(w, "%s pocket-tts binary: skipped\n", PassMark)
	} else {
		ver, err := cfg.PocketTTSVersion()
		if err != nil {
			res.fail(fmt.Sprintf("pocket-tts binary: %v", err))
			fmt.Fprintf(w, "%s pocket-tts binary: not found (%v)\n", FailMark, err)
		} else {
			fmt.Fprintf(w, "%s pocket-tts binary: %s\n", PassMark, ver)
		}
	}

	// ---- Python version ---------------------------------------------------
	if cfg.SkipPython {
		fmt.Fprintf(w, "%s python version: skipped\n", PassMark)
	} else {
		pyVer, err := cfg.PythonVersion()
		if err != nil {
			res.fail(fmt.Sprintf("python version: %v", err))
			fmt.Fprintf(w, "%s python version: not found (%v)\n", FailMark, err)
		} else if pyErr := checkPythonVersion(pyVer); pyErr != nil {
			res.fail(fmt.Sprintf("python version: %v", pyErr))
			fmt.Fprintf(w, "%s python version %s: %v\n", FailMark, pyVer, pyErr)
		} else {
			fmt.Fprintf(w, "%s python version: %s\n", PassMark, pyVer)
		}
	}

	// ---- voice files ------------------------------------------------------
	for _, path := range cfg.VoiceFiles {
		if _, err := os.Stat(path); err != nil {
			res.fail(fmt.Sprintf("voice file %q: %v", path, err))
			fmt.Fprintf(w, "%s voice file %s: not found\n", FailMark, path)
		} else {
			fmt.Fprintf(w, "%s voice file: %s\n", PassMark, path)
		}
	}

	return res
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
