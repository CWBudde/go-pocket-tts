//go:build windows

package model

import (
	"errors"
	"io"
)

type VerifyOptions struct {
	ManifestPath  string
	ORTLibrary    string
	ORTAPIVersion uint32
	Stdout        io.Writer
	Stderr        io.Writer
}

func VerifyONNX(_ VerifyOptions) error {
	return errors.New("onnx model verification is unavailable on windows in this build")
}
