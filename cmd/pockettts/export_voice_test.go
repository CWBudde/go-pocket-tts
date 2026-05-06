package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/onnx"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
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
		{name: "format", def: exportVoiceFormatLegacyEmbedding},
		{name: "language", def: "english_2026-01"},
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

func TestExportVoiceCmd_WritesUpstreamModelStateViaPythonExporter(t *testing.T) {
	origExporter := exportVoiceModelState
	origBuilder := buildVoiceEncoder

	t.Cleanup(func() {
		exportVoiceModelState = origExporter
		buildVoiceEncoder = origBuilder
	})

	var called bool
	var gotAudioPath string
	var gotOutPath string
	var gotLanguage string
	exportVoiceModelState = func(_ context.Context, _ config.Config, audioPath, outPath, language string) error {
		called = true
		gotAudioPath = audioPath
		gotOutPath = outPath
		gotLanguage = language

		return safetensors.WriteFile(outPath, []safetensors.Tensor{
			{
				Name:  "transformer.layers.0.self_attn/cache",
				Shape: []int64{2, 1, 1, 1, 1},
				Data:  []float32{1, 2},
			},
			{
				Name:  "transformer.layers.0.self_attn/offset",
				Shape: []int64{1},
				Data:  []float32{1},
			},
		})
	}
	buildVoiceEncoder = func(_ config.Config, _ string) (voiceEncoder, error) {
		t.Fatal("legacy voice encoder should not be built for --format=model-state")
		return nil, nil
	}

	in := filepath.Join(t.TempDir(), "prompt.wav")
	if err := os.WriteFile(in, []byte{1, 2, 3, 4}, 0o644); err != nil {
		t.Fatalf("write input fixture: %v", err)
	}

	out := filepath.Join(t.TempDir(), "voice.safetensors")

	cmd := NewRootCmd()
	cmd.SilenceUsage = true
	cmd.SetArgs([]string{
		"export-voice",
		"--input=" + in,
		"--out=" + out,
		"--format=model-state",
		"--language=english_2026-01",
	})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("export-voice command failed: %v", err)
	}

	if !called {
		t.Fatal("model-state exporter was not called")
	}

	if gotAudioPath != in {
		t.Fatalf("audio path = %q, want %q", gotAudioPath, in)
	}

	if gotOutPath != out {
		t.Fatalf("out path = %q, want %q", gotOutPath, out)
	}

	if gotLanguage != "english_2026-01" {
		t.Fatalf("language = %q, want english_2026-01", gotLanguage)
	}

	kind, err := safetensors.InspectVoiceFile(out)
	if err != nil {
		t.Fatalf("InspectVoiceFile: %v", err)
	}

	if kind != safetensors.VoiceFileModelState {
		t.Fatalf("voice file kind = %q, want %q", kind, safetensors.VoiceFileModelState)
	}
}
