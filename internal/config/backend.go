package config

import (
	"fmt"
	"strings"
)

const (
	BackendNative     = "native-safetensors"
	BackendNativeONNX = "native-onnx"
	BackendCLI        = "cli"
)

func NormalizeBackend(raw string) (string, error) {
	backend := strings.ToLower(strings.TrimSpace(raw))
	if backend == "" {
		backend = BackendNative
	}
	switch backend {
	case BackendNative, BackendNativeONNX, BackendCLI:
		return backend, nil
	case "native":
		return BackendNative, nil
	default:
		return "", fmt.Errorf(
			"invalid backend %q (expected %s|%s|%s|native)",
			raw,
			BackendNative,
			BackendNativeONNX,
			BackendCLI,
		)
	}
}
