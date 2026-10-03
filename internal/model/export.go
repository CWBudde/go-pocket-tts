package model

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
)

type ExportOptions struct {
	ModelsDir string
	OutDir    string
	Int8      bool
	Variant   string
	Language  string
	Config    string
	PythonBin string
	MaxSeq    int // KV-cache max sequence length (0 = script default 256; use 512+ for voice conditioning)
	Stdout    io.Writer
	Stderr    io.Writer
}

func ExportONNX(opts ExportOptions) error {
	if opts.ModelsDir == "" {
		return errors.New("models dir is required")
	}

	if opts.OutDir == "" {
		return errors.New("out dir is required")
	}

	if opts.Language != "" && opts.Config != "" {
		return errors.New("language and config are mutually exclusive")
	}

	if opts.Language == "" && opts.Config == "" && opts.Variant == "" {
		return errors.New("language, config or variant is required")
	}

	if opts.Stdout == nil {
		opts.Stdout = io.Discard
	}

	if opts.Stderr == nil {
		opts.Stderr = io.Discard
	}

	pythonBin := opts.PythonBin
	if pythonBin == "" {
		pythonBin = detectPocketTTSPython()
	}

	err := validateExportTooling(pythonBin)
	if err != nil {
		return err
	}

	scriptPath, err := resolveScriptPath(filepath.Join("scripts", "export_onnx.py"))
	if err != nil {
		return fmt.Errorf("resolve export helper: %w", err)
	}

	cmd := exec.Command(pythonBin, exportArgs(opts, scriptPath)...)
	cmd.Stdout = opts.Stdout

	cmd.Stderr = opts.Stderr

	err = cmd.Run()
	if err != nil {
		return fmt.Errorf("run ONNX export helper: %w", err)
	}

	return nil
}

// exportArgs builds the export script's arguments. --variant is the script's
// deprecated alias for a language and is passed only when set.
func exportArgs(opts ExportOptions, scriptPath string) []string {
	args := []string{scriptPath, "--models-dir", opts.ModelsDir, "--out-dir", opts.OutDir}
	if opts.Variant != "" {
		args = append(args, "--variant", opts.Variant)
	}

	if opts.Language != "" {
		args = append(args, "--language", opts.Language)
	}

	if opts.Config != "" {
		args = append(args, "--config", opts.Config)
	}

	if opts.Int8 {
		args = append(args, "--int8")
	}

	if opts.MaxSeq > 0 {
		args = append(args, "--max-seq", strconv.Itoa(opts.MaxSeq))
	}

	return args
}

func validateExportTooling(pythonBin string) error {
	_, err := exec.LookPath(pythonBin)
	if err != nil {
		return fmt.Errorf("python interpreter %q not found: %w", pythonBin, err)
	}

	check := exec.Command(pythonBin, "-c", "import pocket_tts, torch, onnx")
	check.Stdout = io.Discard

	check.Stderr = os.Stderr

	err = check.Run()
	if err != nil {
		return fmt.Errorf("python tooling dependencies missing for export (need pocket_tts, torch, onnx): %w", err)
	}

	return nil
}
