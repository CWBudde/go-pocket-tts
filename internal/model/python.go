package model

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func detectPocketTTSPython() string {
	pocketBin, err := exec.LookPath("pocket-tts")
	if err != nil {
		return "python3"
	}
	fh, err := os.Open(pocketBin)
	if err != nil {
		return "python3"
	}
	defer fh.Close()

	s := bufio.NewScanner(fh)
	if !s.Scan() {
		return "python3"
	}
	line := strings.TrimSpace(s.Text())
	if !strings.HasPrefix(line, "#!") {
		return "python3"
	}
	interpreter := strings.TrimSpace(strings.TrimPrefix(line, "#!"))
	if interpreter == "" {
		return "python3"
	}
	if _, err := os.Stat(interpreter); err != nil {
		return "python3"
	}
	return interpreter
}

func resolveScriptPath(rel string) (string, error) {
	if rel == "" {
		return "", fmt.Errorf("script path is required")
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
