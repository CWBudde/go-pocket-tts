package main

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	nativemodel "github.com/cwbudde/go-pocket-tts/internal/native"
	"github.com/cwbudde/go-pocket-tts/internal/onnx"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
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

// TestExportVoiceCmd_WritesUpstreamModelStateViaPythonExporter: backends
// other than native keep exporting the model state with the Python CLI.
func TestExportVoiceCmd_WritesUpstreamModelStateViaPythonExporter(t *testing.T) {
	origExporter := exportVoiceModelStatePython
	origNative := exportVoiceModelStateNative
	origBuilder := buildVoiceEncoder

	t.Cleanup(func() {
		exportVoiceModelStatePython = origExporter
		exportVoiceModelStateNative = origNative
		buildVoiceEncoder = origBuilder
	})

	exportVoiceModelStateNative = func(config.Config, string, string, string) error {
		t.Fatal("native model-state exporter should not run on the cli backend")
		return nil
	}

	var called bool
	var gotAudioPath string
	var gotOutPath string
	var gotLanguage string
	exportVoiceModelStatePython = func(_ context.Context, _ config.Config, audioPath, outPath, language string) error {
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
		return nil, errors.New("unexpected legacy voice encoder build")
	}

	in := filepath.Join(t.TempDir(), "prompt.wav")

	err := os.WriteFile(in, []byte{1, 2, 3, 4}, 0o644)
	if err != nil {
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
		"--backend=cli",
	})

	err = cmd.Execute()
	if err != nil {
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

// TestExportVoiceCmd_ModelStateNativeBackend: on the native backend
// --format model-state builds the state in Go from the checkpoint and model
// config the legacy export uses (--model-safetensors, else
// --paths-model-path, which follows --language); Python is not involved.
func TestExportVoiceCmd_ModelStateNativeBackend(t *testing.T) {
	origPython := exportVoiceModelStatePython
	origNative := exportVoiceModelStateNative

	t.Cleanup(func() {
		exportVoiceModelStatePython = origPython
		exportVoiceModelStateNative = origNative
	})

	exportVoiceModelStatePython = func(context.Context, config.Config, string, string, string) error {
		t.Fatal("the Python exporter should not run on the native backend")
		return nil
	}

	in := filepath.Join(t.TempDir(), "prompt.wav")

	err := os.WriteFile(in, []byte{1, 2, 3, 4}, 0o644)
	if err != nil {
		t.Fatal(err)
	}

	explicit := filepath.Join(t.TempDir(), "gated.safetensors")

	for _, tc := range []struct {
		name        string
		args        []string
		wantWeights string
		wantLang    string
		wantBOS     bool
	}{
		{
			name:        "language default path",
			args:        []string{"--language=german"},
			wantWeights: "models/german/model.safetensors",
			wantLang:    "german",
			wantBOS:     true,
		},
		{
			name:        "paths-model-path",
			args:        []string{"--language=german", "--paths-model-path=" + explicit},
			wantWeights: explicit,
			wantLang:    "german",
			wantBOS:     true,
		},
		{
			name:        "model-safetensors",
			args:        []string{"--model-safetensors=" + explicit, "--backend=native"},
			wantWeights: explicit,
			wantLang:    "english_2026-01",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := filepath.Join(t.TempDir(), "voice.safetensors")

			var called bool

			exportVoiceModelStateNative = func(cfg config.Config, audioPath, outPath, weights string) error {
				called = true

				if audioPath != in || outPath != out {
					t.Errorf("paths = %q, %q; want %q, %q", audioPath, outPath, in, out)
				}

				if weights != tc.wantWeights {
					t.Errorf("weights = %q, want %q", weights, tc.wantWeights)
				}

				if cfg.Model == nil || cfg.TTS.Language != tc.wantLang || cfg.Model.FlowLM.InsertBOSBeforeVoice != tc.wantBOS {
					t.Errorf("model config for %q (BOS before voice %v), want %q (%v)",
						cfg.TTS.Language, cfg.Model != nil && cfg.Model.FlowLM.InsertBOSBeforeVoice, tc.wantLang, tc.wantBOS)
				}

				return nil
			}

			cmd := NewRootCmd()
			cmd.SilenceUsage = true
			cmd.SetArgs(append([]string{"export-voice", "--input=" + in, "--out=" + out, "--format=model-state"}, tc.args...))

			err := cmd.Execute()
			if err != nil {
				t.Fatalf("export-voice: %v", err)
			}

			if !called {
				t.Fatal("native model-state exporter was not called")
			}
		})
	}
}

// TestExportVoiceModelStateNative_Errors covers a missing checkpoint and one
// whose voice encoder is missing.
func TestExportVoiceModelStateNative_Errors(t *testing.T) {
	dir := t.TempDir()
	cfg := config.DefaultConfig()

	err := exportVoiceModelStateNative(cfg, "in.wav", filepath.Join(dir, "out.safetensors"), "")
	if err == nil || !strings.Contains(err.Error(), "--model-safetensors") {
		t.Errorf("no weights: err = %v, want a --model-safetensors hint", err)
	}

	weights := filepath.Join(dir, "model.safetensors")

	err = safetensors.WriteFile(weights, []safetensors.Tensor{{Name: "unrelated", Shape: []int64{1}, Data: []float32{1}}})
	if err != nil {
		t.Fatal(err)
	}

	err = exportVoiceModelStateNative(cfg, "in.wav", filepath.Join(dir, "out.safetensors"), weights)
	if err == nil || !strings.Contains(err.Error(), "mimi.encoder") {
		t.Errorf("checkpoint without encoder: err = %v, want the missing mimi.encoder tensor", err)
	}
}

// TestExportVoiceModelStateNative_UngatedCheckpoint: the ungated checkpoint
// has the Mimi encoder zeroed, so the export must stop with the gated-weights
// hint instead of writing a state every prompt would share.
func TestExportVoiceModelStateNative_UngatedCheckpoint(t *testing.T) {
	weights := filepath.Join("..", "..", "models", "german", "model.safetensors")

	_, err := os.Stat(weights)
	if err != nil {
		t.Skipf("ungated german checkpoint not found: %v", err)
	}

	cfg := gatedExportConfig(t, "german")
	out := filepath.Join(t.TempDir(), "voice.safetensors")

	err = exportVoiceModelStateNative(cfg, parityPromptPath(), out, weights)
	if !errors.Is(err, nativemodel.ErrMimiEncoderWeightsZeroed) {
		t.Fatalf("err = %v, want ErrMimiEncoderWeightsZeroed", err)
	}

	_, statErr := os.Stat(out)
	if statErr == nil {
		t.Fatal("a voice file was written for the ungated checkpoint")
	}
}

// TestExportVoiceModelStateNative_GatedGerman exports the parity prompt with
// the gated german checkpoint and checks that the file is an upstream model
// state the native runtime loads.
func TestExportVoiceModelStateNative_GatedGerman(t *testing.T) {
	gated := os.Getenv("POCKETTTS_GATED_MODELS")
	if gated == "" {
		gated = filepath.Join("..", "..", "models", "gated")
	}

	weights := filepath.Join(gated, "german", "model.safetensors")

	_, err := os.Stat(weights)
	if err != nil {
		t.Skipf("gated german checkpoint not found: %v", err)
	}

	cfg := gatedExportConfig(t, "german")
	out := filepath.Join(t.TempDir(), "voice.safetensors")

	err = exportVoiceModelStateNative(cfg, parityPromptPath(), out, weights)
	if err != nil {
		t.Fatalf("exportVoiceModelStateNative: %v", err)
	}

	kind, err := safetensors.InspectVoiceFile(out)
	if err != nil || kind != safetensors.VoiceFileModelState {
		t.Fatalf("InspectVoiceFile = %q, %v; want %q", kind, err, safetensors.VoiceFileModelState)
	}

	vs, err := safetensors.LoadVoiceModelState(out)
	if err != nil {
		t.Fatalf("LoadVoiceModelState: %v", err)
	}

	m, err := nativemodel.LoadModelFromSafetensors(weights, nativemodel.ConfigFor(cfg.Model))
	if err != nil {
		t.Fatal(err)
	}

	state, err := m.NewFlowStateFromVoiceModelState(vs)
	if err != nil {
		t.Fatalf("NewFlowStateFromVoiceModelState: %v", err)
	}

	// The parity prompt encodes to 27 frames, plus the BOS before the voice.
	if got := state.Offset(); got != 28 {
		t.Fatalf("state offset = %d, want 28", got)
	}
}

func gatedExportConfig(t *testing.T, language string) config.Config {
	t.Helper()

	mc, err := modelcfg.Lookup(language)
	if err != nil {
		t.Fatal(err)
	}

	cfg := config.DefaultConfig()
	cfg.TTS.Language = language
	cfg.Model = mc

	return cfg
}

func parityPromptPath() string {
	return filepath.Join("..", "..", "internal", "native", "testdata", "python_parity", "voice_prompt.wav")
}

// TestVoiceEncoderRunnerConfig_LatentDimFromModel checks that the speaker
// projection width follows the selected model's mimi inner_dim.
func TestVoiceEncoderRunnerConfig_LatentDimFromModel(t *testing.T) {
	for _, tc := range []struct {
		language string
		want     int
	}{
		{language: "english_2026-01", want: 512},
		{language: "german", want: 32},
		{language: "english_2026-09", want: 32},
	} {
		t.Run(tc.language, func(t *testing.T) {
			mc, err := modelcfg.Lookup(tc.language)
			if err != nil {
				t.Fatalf("Lookup: %v", err)
			}

			rcfg := voiceEncoderRunnerConfig(config.Config{Model: mc}, "model.safetensors")
			if rcfg.EncoderLatentDim != tc.want {
				t.Errorf("EncoderLatentDim = %d, want %d", rcfg.EncoderLatentDim, tc.want)
			}

			if rcfg.ModelWeightsPath != "model.safetensors" {
				t.Errorf("ModelWeightsPath = %q, want model.safetensors", rcfg.ModelWeightsPath)
			}
		})
	}

	if got := voiceEncoderRunnerConfig(config.Config{}, "").EncoderLatentDim; got != 0 {
		t.Errorf("without a model config EncoderLatentDim = %d, want 0 (engine default)", got)
	}
}

// TestBuildVoiceEncoder_PicksEncoderByBackend checks that the native backend
// loads the pure-Go encoder from the checkpoint (no ORT, no ONNX manifest)
// and native-onnx keeps the ONNX encoder graph.
func TestBuildVoiceEncoder_PicksEncoderByBackend(t *testing.T) {
	dir := t.TempDir()

	// A checkpoint without encoder tensors: the native loader names them.
	weights := filepath.Join(dir, "model.safetensors")

	err := safetensors.WriteFile(weights, []safetensors.Tensor{{Name: "unrelated", Shape: []int64{1}, Data: []float32{1}}})
	if err != nil {
		t.Fatal(err)
	}

	cfg := config.DefaultConfig()
	cfg.Paths.ONNXManifest = filepath.Join(dir, "missing-manifest.json")
	cfg.Runtime.ORTLibraryPath = filepath.Join(dir, "missing-libonnxruntime.so")

	cfg.TTS.Backend = config.BackendNative

	_, err = buildVoiceEncoder(cfg, weights)
	if err == nil || !strings.Contains(err.Error(), "mimi.encoder") {
		t.Errorf("native backend: err = %v, want the native loader's missing mimi.encoder tensor", err)
	}

	_, err = buildVoiceEncoder(cfg, "")
	if err == nil || !strings.Contains(err.Error(), "--model-safetensors") {
		t.Errorf("native backend without weights: err = %v, want a --model-safetensors hint", err)
	}

	cfg.TTS.Backend = config.BackendNativeONNX

	_, err = buildVoiceEncoder(cfg, weights)
	if err == nil || strings.Contains(err.Error(), "mimi.encoder") || !strings.Contains(err.Error(), "onnx") {
		t.Errorf("native-onnx backend: err = %v, want the ONNX engine's error", err)
	}
}

// TestBuildVoiceEncoder_NativeAppliesWorkers: the native encoder runs
// without a tts.Service, so it must apply --conv-workers / --runtime-workers
// itself; without them every kernel ran single-threaded.
func TestBuildVoiceEncoder_NativeAppliesWorkers(t *testing.T) {
	defer tensor.SetWorkers(tensor.Workers())

	tensor.SetWorkers(1)

	cfg := config.DefaultConfig()
	cfg.TTS.Backend = config.BackendNative
	cfg.Runtime.Workers = 5

	_, _ = buildVoiceEncoder(cfg, "")

	if got := tensor.Workers(); got != 5 {
		t.Errorf("tensor workers = %d after buildVoiceEncoder, want --runtime-workers 5", got)
	}
}

// TestNativeVoiceEncoder_GatedGerman runs the CLI's native encoder on the
// parity prompt with the gated german checkpoint and compares its first
// conditioning frame with upstream's (internal/native/testdata).
func TestNativeVoiceEncoder_GatedGerman(t *testing.T) {
	gated := os.Getenv("POCKETTTS_GATED_MODELS")
	if gated == "" {
		gated = filepath.Join("..", "..", "models", "gated")
	}

	weights := filepath.Join(gated, "german", "model.safetensors")

	_, err := os.Stat(weights)
	if err != nil {
		t.Skipf("gated german checkpoint not found: %v", err)
	}

	testdata := filepath.Join("..", "..", "internal", "native", "testdata", "python_parity")

	enc, err := newNativeVoiceEncoder(weights)
	if err != nil {
		t.Fatalf("newNativeVoiceEncoder: %v", err)
	}
	defer enc.Close()

	got, err := enc.EncodeVoice(filepath.Join(testdata, "voice_prompt.wav"))
	if err != nil {
		t.Fatalf("EncodeVoice: %v", err)
	}

	raw, err := os.ReadFile(filepath.Join(testdata, "encoder_german.json"))
	if err != nil {
		t.Fatal(err)
	}

	var fx struct {
		Latent struct {
			Shape []int64 `json:"shape"`
		} `json:"latent"`
		ConditioningRows struct {
			Frames []int64   `json:"frames"`
			Data   []float32 `json:"data"`
		} `json:"conditioning_rows"`
	}

	err = json.Unmarshal(raw, &fx)
	if err != nil {
		t.Fatal(err)
	}

	frames := fx.Latent.Shape[1]
	if int64(len(got)) != frames*onnx.VoiceEmbeddingDim {
		t.Fatalf("embedding has %d values, want %d frames × %d", len(got), frames, onnx.VoiceEmbeddingDim)
	}

	if fx.ConditioningRows.Frames[0] != 0 {
		t.Fatalf("fixture's first conditioning row is frame %d, want 0", fx.ConditioningRows.Frames[0])
	}

	for i := range onnx.VoiceEmbeddingDim {
		want := float64(fx.ConditioningRows.Data[i])
		if diff := math.Abs(float64(got[i]) - want); diff > 2e-4+1e-3*math.Abs(want) {
			t.Fatalf("conditioning[0,%d] = %g, upstream %g", i, got[i], want)
		}
	}
}
