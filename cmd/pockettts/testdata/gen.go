//go:build ignore

// gen.go generates the fixture WAV files in this directory.
// Run with: go run ./cmd/pockettts/testdata/gen.go
package main

import (
	"os"
	"path/filepath"
	"runtime"

	"github.com/example/go-pocket-tts/internal/audio"
)

func main() {
	_, file, _, _ := runtime.Caller(0)
	dir := filepath.Dir(file)

	// 100 ms of silence at 24000 Hz = 2400 samples
	samples := make([]float32, 2400)
	data, err := audio.EncodeWAV(samples)
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "silence_100ms.wav"), data, 0o644); err != nil {
		panic(err)
	}
}
