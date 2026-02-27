package onnx

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sync"
	"sync/atomic"

	"github.com/example/go-pocket-tts/internal/config"
)

type RuntimeInfo struct {
	LibraryPath string
	Version     string
	Initialized bool
}

var versionPattern = regexp.MustCompile(`([0-9]+\.[0-9]+\.[0-9]+)`)

var (
	bootstrapOnce sync.Once
	bootstrapInfo RuntimeInfo
	errBootstrap  error
	shutdownFlag  atomic.Bool
)

func Bootstrap(cfg config.RuntimeConfig) (RuntimeInfo, error) {
	bootstrapOnce.Do(func() {
		info, err := DetectRuntime(cfg)
		if err != nil {
			errBootstrap = err
			return
		}

		// Keep this process-local marker for future ORT bindings.
		err = os.Setenv("POCKETTTS_ORT_LIB", info.LibraryPath)
		if err != nil {
			errBootstrap = fmt.Errorf("set POCKETTTS_ORT_LIB: %w", err)
			return
		}

		bootstrapInfo = info
		bootstrapInfo.Initialized = true
	})

	if errBootstrap != nil {
		return RuntimeInfo{}, errBootstrap
	}

	return bootstrapInfo, nil
}

func Shutdown() error {
	if !bootstrapInfo.Initialized {
		return nil
	}

	if shutdownFlag.Swap(true) {
		return nil
	}

	// Placeholder cleanup point for native ORT environment/session teardown.
	bootstrapInfo.Initialized = false

	return nil
}

func DetectRuntime(cfg config.RuntimeConfig) (RuntimeInfo, error) {
	path := cfg.ORTLibraryPath
	if path == "" {
		path = os.Getenv("POCKETTTS_ORT_LIB")
	}

	if path == "" {
		path = os.Getenv("ORT_LIBRARY_PATH")
	}

	if path == "" {
		candidates := []string{
			"/usr/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"C:/onnxruntime/lib/onnxruntime.dll",
		}
		for _, c := range candidates {
			_, err := os.Stat(c)
			if err == nil {
				path = c
				break
			}
		}
	}

	if path == "" {
		return RuntimeInfo{LibraryPath: "not found", Version: "unknown"}, errors.New("unable to detect ONNX Runtime library path")
	}

	_, err := os.Stat(path)
	if err != nil {
		return RuntimeInfo{LibraryPath: path, Version: "unknown"}, fmt.Errorf("onnx runtime library path check failed: %w", err)
	}

	version := cfg.ORTVersion
	if version == "" {
		version = os.Getenv("ORT_VERSION")
	}

	if version == "" {
		version = inferVersionFromPath(path)
	}

	if version == "" {
		version = "unknown"
	}

	return RuntimeInfo{LibraryPath: path, Version: version}, nil
}

func inferVersionFromPath(path string) string {
	name := filepath.Base(path)
	if m := versionPattern.FindStringSubmatch(name); len(m) == 2 {
		return m[1]
	}

	return ""
}
