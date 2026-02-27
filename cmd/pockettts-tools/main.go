package main

import (
	"fmt"
	"os"
)

func main() {
	err := NewRootCmd().Execute()
	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)

		os.Exit(1)
	}
}
