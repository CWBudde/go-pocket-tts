package model

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const defaultPythonBin = "python3"

func detectPocketTTSPython() string {
	pocketBin, err := exec.LookPath("pocket-tts")
	if err != nil {
		return defaultPythonBin
	}

	fh, err := os.Open(pocketBin)
	if err != nil {
		return defaultPythonBin
	}

	defer func() { _ = fh.Close() }()

	s := bufio.NewScanner(fh)
	if !s.Scan() {
		return defaultPythonBin
	}

	line := strings.TrimSpace(s.Text())
	if !strings.HasPrefix(line, "#!") {
		return defaultPythonBin
	}

	interpreter := strings.TrimSpace(strings.TrimPrefix(line, "#!"))
	if interpreter == "" {
		return defaultPythonBin
	}

	if _, err := os.Stat(interpreter); err != nil {
		return defaultPythonBin
	}

	return interpreter
}

func resolveScriptPath(rel string) (string, error) {
	if rel == "" {
		return "", errors.New("script path is required")
	}

	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("get working directory: %w", err)
	}

	paths := []string{
		filepath.Join(cwd, rel),
		filepath.Join(cwd, "..", "..", rel),
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return filepath.Clean(p), nil
		}
	}

	return "", fmt.Errorf("script %q not found from %s", rel, cwd)
}
