package onnx

import (
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/example/go-pocket-tts/internal/config"
)

func resetRuntimeStateForTest() {
	bootstrapOnce = sync.Once{}
	bootstrapInfo = RuntimeInfo{}
	bootstrapErr = nil
	shutdownFlag.Store(false)
}

func TestDetectRuntimePrefersPOCKETTTSORTLIB(t *testing.T) {
	tmp := t.TempDir()
	lib := filepath.Join(tmp, "libonnxruntime.so")
	if err := os.WriteFile(lib, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write fake lib: %v", err)
	}

	t.Setenv("POCKETTTS_ORT_LIB", lib)
	t.Setenv("ORT_LIBRARY_PATH", filepath.Join(tmp, "does-not-exist"))

	info, err := DetectRuntime(config.RuntimeConfig{})
	if err != nil {
		t.Fatalf("DetectRuntime failed: %v", err)
	}
	if info.LibraryPath != lib {
		t.Fatalf("expected %q, got %q", lib, info.LibraryPath)
	}
}

func TestBootstrapRunsOnce(t *testing.T) {
	resetRuntimeStateForTest()

	tmp := t.TempDir()
	lib1 := filepath.Join(tmp, "lib1.so")
	lib2 := filepath.Join(tmp, "lib2.so")
	if err := os.WriteFile(lib1, []byte("one"), 0o644); err != nil {
		t.Fatalf("write lib1: %v", err)
	}
	if err := os.WriteFile(lib2, []byte("two"), 0o644); err != nil {
		t.Fatalf("write lib2: %v", err)
	}

	cfg1 := config.RuntimeConfig{Threads: 1, ORTLibraryPath: lib1}
	cfg2 := config.RuntimeConfig{Threads: 1, ORTLibraryPath: lib2}

	info1, err := Bootstrap(cfg1)
	if err != nil {
		t.Fatalf("first bootstrap failed: %v", err)
	}
	info2, err := Bootstrap(cfg2)
	if err != nil {
		t.Fatalf("second bootstrap failed: %v", err)
	}

	if info1.LibraryPath != lib1 {
		t.Fatalf("expected first lib path %q, got %q", lib1, info1.LibraryPath)
	}
	if info2.LibraryPath != lib1 {
		t.Fatalf("expected once semantics to keep %q, got %q", lib1, info2.LibraryPath)
	}

	if err := Shutdown(); err != nil {
		t.Fatalf("shutdown failed: %v", err)
	}
}
