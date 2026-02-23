package config

import (
	"fmt"
	"strings"
)

const (
	BackendNative            = "native-onnx"
	BackendNativeLegacyAlias = "native"
	BackendNativeSafetensors = "native-safetensors"
	BackendCLI               = "cli"
)

func NormalizeBackend(raw string) (string, error) {
	backend := strings.ToLower(strings.TrimSpace(raw))
	if backend == "" {
		backend = BackendNative
	}
	switch backend {
	case BackendNative, BackendCLI, BackendNativeSafetensors:
		return backend, nil
	case BackendNativeLegacyAlias:
		return BackendNativeSafetensors, nil
	default:
		return "", fmt.Errorf(
			"invalid backend %q (expected %s|%s|%s|%s)",
			raw,
			BackendNative,
			BackendNativeLegacyAlias,
			BackendNativeSafetensors,
			BackendCLI,
		)
	}
}
