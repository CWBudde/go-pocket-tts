package main

import (
	"fmt"
	"os"

	"github.com/example/go-pocket-tts/internal/onnx"
)

func main() {
	err := NewRootCmd().Execute()

	shutdownErr := onnx.Shutdown()
	if shutdownErr != nil && err == nil {
		err = shutdownErr
	}

	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)

		os.Exit(1)
	}
}
