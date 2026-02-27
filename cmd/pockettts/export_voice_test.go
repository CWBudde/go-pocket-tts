package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/onnx"
	"github.com/example/go-pocket-tts/internal/safetensors"
)

type fakeVoiceEncoder struct {
	input   string
	closed  bool
	output  []float32
	runErr  error
	closeFn func()
}

func (f *fakeVoiceEncoder) EncodeVoice(audioPath string) ([]float32, error) {
	f.input = audioPath
	if f.runErr != nil {
		return nil, f.runErr
	}

	return append([]float32(nil), f.output...), nil
}

func (f *fakeVoiceEncoder) Close() {
	f.closed = true
	if f.closeFn != nil {
		f.closeFn()
	}
}

func TestNewExportVoiceCmd_Flags(t *testing.T) {
	cmd := newExportVoiceCmd()
	if cmd.Use != "export-voice" {
		t.Fatalf("Use = %q, want export-voice", cmd.Use)
	}

	for _, tc := range []struct {
		name string
		def  string
	}{
		{name: "input", def: ""},
		{name: "audio", def: ""},
		{name: "out", def: ""},
		{name: "model-safetensors", def: ""},
		{name: "id", def: "custom-voice"},
		{name: "license", def: "unknown"},
	} {
		flag := cmd.Flags().Lookup(tc.name)
		if flag == nil {
			t.Fatalf("flag %q not registered", tc.name)
		}

		if flag.DefValue != tc.def {
			t.Fatalf("flag %q default = %q, want %q", tc.name, flag.DefValue, tc.def)
		}
	}
}

func TestExportVoiceCmd_RequiresInput(t *testing.T) {
	cmd := NewRootCmd()
	cmd.SilenceUsage = true
	cmd.SetArgs([]string{"export-voice", "--out=/tmp/out.safetensors"})

	err := cmd.Execute()
	if err == nil {
		t.Fatal("expected error when --input is missing")
	}

	if !strings.Contains(err.Error(), "--input") {
		t.Fatalf("error %q should mention --input", err.Error())
	}
}

func TestExportVoiceCmd_RequiresOut(t *testing.T) {
	in := filepath.Join(t.TempDir(), "in.wav")

	err := os.WriteFile(in, []byte{0, 1}, 0o644)
	if err != nil {
		t.Fatalf("write input fixture: %v", err)
	}

	cmd := NewRootCmd()
	cmd.SilenceUsage = true
	cmd.SetArgs([]string{"export-voice", "--input=" + in})

	err = cmd.Execute()
	if err == nil {
		t.Fatal("expected error when --out is missing")
	}

	if !strings.Contains(err.Error(), "--out") {
		t.Fatalf("error %q should mention --out", err.Error())
	}
}

func TestExportVoiceCmd_WritesSafetensorsViaNativeEncoder(t *testing.T) {
	origBuilder := buildVoiceEncoder

	t.Cleanup(func() { buildVoiceEncoder = origBuilder })

	fake := &fakeVoiceEncoder{
		output: make([]float32, 2*onnx.VoiceEmbeddingDim),
	}
	fake.output[0] = 1.25
	fake.output[onnx.VoiceEmbeddingDim+1] = -2.5

	var capturedWeightsPath string
	buildVoiceEncoder = func(_ config.Config, modelWeightsPath string) (voiceEncoder, error) {
		capturedWeightsPath = modelWeightsPath
		return fake, nil
	}

	in := filepath.Join(t.TempDir(), "prompt.wav")

	err := os.WriteFile(in, []byte{1, 2, 3, 4}, 0o644)
	if err != nil {
		t.Fatalf("write input fixture: %v", err)
	}

	out := filepath.Join(t.TempDir(), "voice.safetensors")

	modelPath := filepath.Join(t.TempDir(), "tts_b6369a24.safetensors")

	err = os.WriteFile(modelPath, []byte("stub"), 0o644)
	if err != nil {
		t.Fatalf("write model fixture: %v", err)
	}

	cmd := NewRootCmd()
	cmd.SilenceUsage = true
	cmd.SetArgs([]string{
		"export-voice",
		"--input=" + in,
		"--out=" + out,
		"--model-safetensors=" + modelPath,
		"--id=my-voice",
		"--license=CC-BY-4.0",
	})

	err = cmd.Execute()
	if err != nil {
		t.Fatalf("export-voice command failed: %v", err)
	}

	if fake.input != in {
		t.Fatalf("EncodeVoice called with input %q, want %q", fake.input, in)
	}

	if !fake.closed {
		t.Fatal("expected encoder.Close() to be called")
	}

	if capturedWeightsPath != modelPath {
		t.Fatalf("model weights path = %q, want %q", capturedWeightsPath, modelPath)
	}

	got, shape, err := safetensors.LoadVoiceEmbedding(out)
	if err != nil {
		t.Fatalf("LoadVoiceEmbedding(%s): %v", out, err)
	}

	if len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != onnx.VoiceEmbeddingDim {
		t.Fatalf("shape = %v, want [1 2 %d]", shape, onnx.VoiceEmbeddingDim)
	}

	if len(got) != len(fake.output) {
		t.Fatalf("data length = %d, want %d", len(got), len(fake.output))
	}

	if got[0] != fake.output[0] || got[onnx.VoiceEmbeddingDim+1] != fake.output[onnx.VoiceEmbeddingDim+1] {
		t.Fatalf("output values mismatch")
	}
}
