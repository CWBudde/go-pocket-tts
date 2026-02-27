package main

import (
	"fmt"
	"os"

	"github.com/example/go-pocket-tts/internal/onnx"
)

func main() {
	defer func() {
		_ = onnx.Shutdown()
	}()

	err := NewRootCmd().Execute()
	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)

		os.Exit(1)
	}
}
