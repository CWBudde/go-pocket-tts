package model

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
)

type ExportOptions struct {
	ModelsDir string
	OutDir    string
	Int8      bool
	Variant   string
	PythonBin string
	MaxSeq    int // KV-cache max sequence length (0 = script default 256; use 512+ for voice conditioning)
	Stdout    io.Writer
	Stderr    io.Writer
}

func ExportONNX(opts ExportOptions) error {
	if opts.ModelsDir == "" {
		return fmt.Errorf("models dir is required")
	}
	if opts.OutDir == "" {
		return fmt.Errorf("out dir is required")
	}
	if opts.Variant == "" {
		opts.Variant = "b6369a24"
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
	if err := validateExportTooling(pythonBin); err != nil {
		return err
	}

	scriptPath, err := resolveScriptPath(filepath.Join("scripts", "export_onnx.py"))
	if err != nil {
		return fmt.Errorf("resolve export helper: %w", err)
	}

	args := []string{scriptPath, "--models-dir", opts.ModelsDir, "--out-dir", opts.OutDir, "--variant", opts.Variant}
	if opts.Int8 {
		args = append(args, "--int8")
	}
	if opts.MaxSeq > 0 {
		args = append(args, "--max-seq", fmt.Sprintf("%d", opts.MaxSeq))
	}

	cmd := exec.Command(pythonBin, args...)
	cmd.Stdout = opts.Stdout
	cmd.Stderr = opts.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("run ONNX export helper: %w", err)
	}

	return nil
}

func validateExportTooling(pythonBin string) error {
	if _, err := exec.LookPath(pythonBin); err != nil {
		return fmt.Errorf("python interpreter %q not found: %w", pythonBin, err)
	}

	check := exec.Command(pythonBin, "-c", "import pocket_tts, torch, onnx")
	check.Stdout = io.Discard
	check.Stderr = os.Stderr
	if err := check.Run(); err != nil {
		return fmt.Errorf("python tooling dependencies missing for export (need pocket_tts, torch, onnx): %w", err)
	}
	return nil
}
