//go:build integration

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/example/go-pocket-tts/internal/testutil"
)

// TestSynthCLI_ShortText synthesizes a short phrase via the CLI backend and
// asserts a valid WAV with non-zero PCM samples at 24 kHz.
func TestSynthCLI_ShortText(t *testing.T) {
	testutil.RequirePocketTTS(t)

	voice := synthTestVoice(t)
	out := filepath.Join(t.TempDir(), "out.wav")

	root := NewRootCmd()
	root.SetArgs([]string{
		"synth",
		"--backend", "cli",
		"--text", "Hello world.",
		"--voice", voice,
		"--out", out,
	})
	if err := root.Execute(); err != nil {
		t.Fatalf("synth --backend cli failed: %v", err)
	}

	data := readFile(t, out)
	testutil.AssertValidWAV(t, data)
}

// TestSynthCLI_Chunked synthesizes multi-sentence text with --chunk via the
// CLI backend and asserts that the concatenated output is longer than a
// single-sentence synthesis would be.
func TestSynthCLI_Chunked(t *testing.T) {
	testutil.RequirePocketTTS(t)

	voice := synthTestVoice(t)
	outSingle := filepath.Join(t.TempDir(), "single.wav")
	outChunked := filepath.Join(t.TempDir(), "chunked.wav")

	runSynth := func(text, out string, chunk bool) {
		t.Helper()
		args := []string{
			"synth",
			"--backend", "cli",
			"--text", text,
			"--voice", voice,
			"--out", out,
		}
		if chunk {
			args = append(args, "--chunk")
		}
		root := NewRootCmd()
		root.SetArgs(args)
		if err := root.Execute(); err != nil {
			t.Fatalf("synth failed: %v", err)
		}
	}

	runSynth("Hello world.", outSingle, false)
	runSynth("Hello world. This is a longer sentence. It has three parts.", outChunked, true)

	dataSingle := readFile(t, outSingle)
	dataChunked := readFile(t, outChunked)

	testutil.AssertValidWAV(t, dataSingle)
	testutil.AssertValidWAV(t, dataChunked)

	if len(dataChunked) <= len(dataSingle) {
		t.Errorf("chunked output (%d bytes) expected to be larger than single-sentence output (%d bytes)", len(dataChunked), len(dataSingle))
	}
}

// TestSynthCLI_DSPChain synthesizes with --normalize --dc-block --fade-in-ms
// --fade-out-ms and asserts the output is still a valid WAV with equal sample
// count (DSP does not add/remove samples).
func TestSynthCLI_DSPChain(t *testing.T) {
	testutil.RequirePocketTTS(t)

	voice := synthTestVoice(t)
	outRaw := filepath.Join(t.TempDir(), "raw.wav")
	outDSP := filepath.Join(t.TempDir(), "dsp.wav")

	runSynth := func(out string, dsp bool) {
		t.Helper()
		args := []string{
			"synth",
			"--backend", "cli",
			"--text", "Hello.",
			"--voice", voice,
			"--out", out,
		}
		if dsp {
			args = append(args,
				"--normalize",
				"--dc-block",
				"--fade-in-ms", "10",
				"--fade-out-ms", "10",
			)
		}
		root := NewRootCmd()
		root.SetArgs(args)
		if err := root.Execute(); err != nil {
			t.Fatalf("synth failed: %v", err)
		}
	}

	runSynth(outRaw, false)
	runSynth(outDSP, true)

	dataRaw := readFile(t, outRaw)
	dataDSP := readFile(t, outDSP)

	testutil.AssertValidWAV(t, dataRaw)
	testutil.AssertValidWAV(t, dataDSP)

	rawSamples := wavSampleCount(t, dataRaw)
	dspSamples := wavSampleCount(t, dataDSP)
	if rawSamples != dspSamples {
		t.Errorf("DSP changed sample count: raw=%d dsp=%d", rawSamples, dspSamples)
	}
}

// TestSynthCLI_Stdout synthesizes to stdout (--out -) and asserts RIFF bytes
// in the captured output.
func TestSynthCLI_Stdout(t *testing.T) {
	testutil.RequirePocketTTS(t)

	voice := synthTestVoice(t)

	var buf bytes.Buffer
	root := NewRootCmd()
	root.SetArgs([]string{
		"synth",
		"--backend", "cli",
		"--text", "Hello.",
		"--voice", voice,
		"--out", "-",
	})
	root.SetOut(&buf)

	// writeSynthOutput writes to os.Stdout directly, so we capture via a pipe.
	pr, pw, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	origStdout := os.Stdout
	os.Stdout = pw

	execErr := root.Execute()

	pw.Close()
	os.Stdout = origStdout

	var captured bytes.Buffer
	if _, err := captured.ReadFrom(pr); err != nil {
		t.Fatalf("read pipe: %v", err)
	}
	pr.Close()

	if execErr != nil {
		t.Fatalf("synth --out - failed: %v", execErr)
	}

	data := captured.Bytes()
	if len(data) < 4 || string(data[0:4]) != "RIFF" {
		t.Fatalf("stdout does not start with RIFF header (got %d bytes)", len(data))
	}
	testutil.AssertValidWAV(t, data)
}

// TestSynthNative_ShortText synthesizes a short phrase via the native backend
// and asserts a valid WAV with non-zero samples. Skips if ONNX Runtime or
// model files are absent.
func TestSynthNative_ShortText(t *testing.T) {
	testutil.RequireONNXRuntime(t)

	out := filepath.Join(t.TempDir(), "out.wav")

	root := NewRootCmd()
	root.SetArgs([]string{
		"synth",
		"--backend", "native-onnx",
		"--text", "Hello world.",
		"--out", out,
	})
	if err := root.Execute(); err != nil {
		t.Fatalf("synth --backend native failed: %v", err)
	}

	data := readFile(t, out)
	testutil.AssertValidWAV(t, data)
}

// TestSynthNative_Chunked synthesizes multi-sentence text with --chunk via the
// native backend and asserts that the PCM sample count grows with chunk count.
func TestSynthNative_Chunked(t *testing.T) {
	testutil.RequireONNXRuntime(t)

	outSingle := filepath.Join(t.TempDir(), "single.wav")
	outChunked := filepath.Join(t.TempDir(), "chunked.wav")

	runSynth := func(text, out string, chunk bool) {
		t.Helper()
		args := []string{
			"synth",
			"--backend", "native-onnx",
			"--text", text,
			"--out", out,
		}
		if chunk {
			args = append(args, "--chunk")
		}
		root := NewRootCmd()
		root.SetArgs(args)
		if err := root.Execute(); err != nil {
			t.Fatalf("synth failed: %v", err)
		}
	}

	runSynth("Hello.", outSingle, false)
	runSynth("Hello. Goodbye. See you.", outChunked, true)

	dataSingle := readFile(t, outSingle)
	dataChunked := readFile(t, outChunked)

	testutil.AssertValidWAV(t, dataSingle)
	testutil.AssertValidWAV(t, dataChunked)

	singleSamples := wavSampleCount(t, dataSingle)
	chunkedSamples := wavSampleCount(t, dataChunked)
	if chunkedSamples <= singleSamples {
		t.Errorf("native chunked sample count (%d) expected to exceed single-sentence count (%d)", chunkedSamples, singleSamples)
	}
}

// TestSynthNativeSafetensors_ShortText synthesizes via the safetensors-native
// backend and asserts a valid WAV at 24 kHz.
func TestSynthNativeSafetensors_ShortText(t *testing.T) {
	modelPath, tokPath := requireNativeSafetensorsAssets(t)
	out := filepath.Join(t.TempDir(), "native_safetensors.wav")

	root := NewRootCmd()
	root.SetArgs([]string{
		"--paths-model-path", modelPath,
		"--paths-tokenizer-model", tokPath,
		"synth",
		"--backend", "native-safetensors",
		"--text", "Hello from safetensors native backend.",
		"--out", out,
	})
	if err := root.Execute(); err != nil {
		t.Fatalf("synth --backend native-safetensors failed: %v", err)
	}

	data := readFile(t, out)
	testutil.AssertValidWAV(t, data)
}

// synthTestVoice returns the voice to use for CLI integration tests.
// It reads POCKETTTS_TEST_VOICE from the environment; if unset the test is skipped.
func synthTestVoice(t testing.TB) string {
	t.Helper()
	v := os.Getenv("POCKETTTS_TEST_VOICE")
	if v == "" {
		t.Skip("set POCKETTTS_TEST_VOICE to a valid voice for CLI integration tests")
	}
	return v
}

// readFile reads a file and fails the test on error.
func readFile(t testing.TB, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %q: %v", path, err)
	}
	return data
}

// wavSampleCount returns the number of 16-bit mono samples in a WAV file.
func wavSampleCount(t testing.TB, data []byte) int {
	t.Helper()
	// 16-bit mono: data chunk size / 2.
	// We use the same chunk-walker as AssertValidWAV internally.
	// Parse the data chunk size from the header offset directly (standard 44-byte WAV).
	// For robustness, scan for the data chunk.
	const hdrSize = 44
	if len(data) < hdrSize {
		t.Fatalf("WAV too short for sample count: %d bytes", len(data))
	}
	// Walk chunks from offset 12 (after RIFF/WAVE).
	offset := 12
	for offset+8 <= len(data) {
		id := string(data[offset : offset+4])
		size := uint32(data[offset+4]) | uint32(data[offset+5])<<8 | uint32(data[offset+6])<<16 | uint32(data[offset+7])<<24
		if id == "data" {
			return int(size) / 2
		}
		offset += 8 + int(size)
		if size%2 != 0 {
			offset++
		}
	}
	t.Fatal("WAV: data chunk not found")
	return 0
}

func requireNativeSafetensorsAssets(t testing.TB) (modelPath, tokPath string) {
	t.Helper()
	modelCandidates := []string{
		filepath.Join("models", "tts_b6369a24.safetensors"),
		filepath.Join("..", "..", "models", "tts_b6369a24.safetensors"),
	}
	tokCandidates := []string{
		filepath.Join("models", "tokenizer.model"),
		filepath.Join("..", "..", "models", "tokenizer.model"),
	}
	for _, p := range modelCandidates {
		if _, err := os.Stat(p); err == nil {
			modelPath = p
			break
		}
	}
	for _, p := range tokCandidates {
		if _, err := os.Stat(p); err == nil {
			tokPath = p
			break
		}
	}
	if modelPath == "" || tokPath == "" {
		t.Skipf("native safetensors assets unavailable (model=%q tokenizer=%q)", modelPath, tokPath)
	}
	return modelPath, tokPath
}
