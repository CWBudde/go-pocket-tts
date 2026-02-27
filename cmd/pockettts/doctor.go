package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/doctor"
	"github.com/example/go-pocket-tts/internal/model"
	"github.com/example/go-pocket-tts/internal/safetensors"
	"github.com/example/go-pocket-tts/internal/tts"
	"github.com/spf13/cobra"
)

func newDoctorCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Run local runtime and model checks",
		RunE: func(_ *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			exe := cfg.TTS.CLIPath
			if exe == "" {
				exe = "pocket-tts"
			}

			backend, err := config.NormalizeBackend(cfg.TTS.Backend)
			if err != nil {
				return err
			}

			nativeMode := backend == config.BackendNative || backend == config.BackendNativeONNX
			_, _ = fmt.Fprintf(os.Stdout, "backend: %s\n", backend)

			dcfg := doctor.Config{
				PocketTTSVersion: func() (string, error) {
					return probePocketTTSVersion(exe)
				},
				SkipPocketTTS: nativeMode,
				PythonVersion: probePythonVersion,
				SkipPython:    nativeMode,
				VoiceFiles:    collectVoiceFiles(),
			}
			if backend == config.BackendNative {
				dcfg.NativeModelPath = cfg.Paths.ModelPath
				dcfg.TokenizerModelPath = cfg.Paths.TokenizerModel
				dcfg.ValidateSafetensors = safetensors.ValidateModelKeys
			}

			result := doctor.Run(dcfg, os.Stdout)

			// ONNX model verify as an additional check.
			// Skip gracefully when no manifest is present (models not yet downloaded).
			const onnxManifest = "models/onnx/manifest.json"

			//nolint:nestif // Keep doctor verify status reporting in one place with explicit skip/fail messaging.
			if backend == config.BackendNative {
				_, _ = fmt.Fprintf(
					os.Stdout,
					"%s model verify: skipped (backend %s does not require ONNX graphs)\n",
					doctor.PassMark,
					config.BackendNative,
				)
			} else {
				_, manifestStatErr := os.Stat(onnxManifest)
				if os.IsNotExist(manifestStatErr) {
					_, _ = fmt.Fprintf(os.Stdout, "%s model verify: skipped (no manifest at %s)\n", doctor.PassMark, onnxManifest)
				} else {
					verifyErr := model.VerifyONNX(model.VerifyOptions{
						ManifestPath: onnxManifest,
						ORTLibrary:   cfg.Runtime.ORTLibraryPath,
						Stdout:       os.Stdout,
						Stderr:       os.Stderr,
					})
					if verifyErr != nil {
						result.AddFailure(fmt.Sprintf("model verify: %v", verifyErr))
						_, _ = fmt.Fprintf(os.Stdout, "%s model verify: %v\n", doctor.FailMark, verifyErr)
					} else {
						_, _ = fmt.Fprintf(os.Stdout, "%s model verify: ok\n", doctor.PassMark)
					}
				}
			}

			if result.Failed() {
				for _, f := range result.Failures() {
					// #nosec G705 -- Writes plain diagnostic text to stderr for CLI output, not HTML rendering.
					fmt.Fprintf(os.Stderr, "FAIL: %s\n", f)
				}

				return errors.New("doctor checks failed")
			}

			_, _ = fmt.Fprintln(os.Stdout, "doctor checks passed")

			return nil
		},
	}

	return cmd
}

// probePocketTTSVersion runs `pocket-tts --version` and returns its output.
func probePocketTTSVersion(exe string) (string, error) {
	out, err := exec.CommandContext(context.Background(), exe, "--version").Output()
	if err != nil {
		return "", fmt.Errorf("%s --version failed: %w", exe, err)
	}

	return strings.TrimSpace(string(out)), nil
}

// probePythonVersion tries python3 then python and returns the version string.
func probePythonVersion() (string, error) {
	for _, bin := range []string{"python3", "python"} {
		out, err := exec.CommandContext(context.Background(), bin, "--version").Output()
		if err != nil {
			continue
		}
		// Output is e.g. "Python 3.11.4\n"
		raw := strings.TrimSpace(string(out))

		raw = strings.TrimPrefix(raw, "Python ")
		if raw != "" {
			return raw, nil
		}
	}

	return "", errors.New("python3/python not found on PATH")
}

// collectVoiceFiles returns resolved absolute voice file paths from the
// manifest. Paths are resolved relative to the manifest directory, not to the
// working directory, so doctor checks are correct regardless of CWD.
func collectVoiceFiles() []string {
	vm, err := tts.NewVoiceManager("voices/manifest.json")
	if err != nil {
		return nil
	}

	voices := vm.ListVoices()

	paths := make([]string, 0, len(voices))
	for _, v := range voices {
		resolved, err := vm.ResolvePath(v.ID)
		if err != nil {
			// Voice file missing or unresolvable — include the raw path so the
			// doctor check can report the failure with a useful message.
			paths = append(paths, v.Path)
			continue
		}
		// Make the path absolute so the doctor stat check is CWD-independent.
		abs, err := filepath.Abs(resolved)
		if err == nil {
			resolved = abs
		}

		paths = append(paths, resolved)
	}

	return paths
}
