package config

import (
	"fmt"
	"strings"
)

func NormalizeBackend(raw string) (string, error) {
	backend := strings.ToLower(strings.TrimSpace(raw))
	if backend == "" {
		backend = "native"
	}
	switch backend {
	case "native", "cli":
		return backend, nil
	default:
		return "", fmt.Errorf("invalid backend %q (expected native|cli)", raw)
	}
}
