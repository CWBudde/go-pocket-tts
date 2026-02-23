package tts

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

type parityTokenizer struct{}

func (p parityTokenizer) Encode(text string) ([]int64, error) {
	words := strings.Fields(text)
	out := make([]int64, len(words))
	for i := range words {
		out[i] = int64(i + 1)
	}
	return out, nil
}

type parityRuntime struct {
	samples []float32
}

func (p *parityRuntime) GenerateAudio(_ context.Context, _ []int64, _ RuntimeGenerateConfig) ([]float32, error) {
	return append([]float32(nil), p.samples...), nil
}

func (p *parityRuntime) Close() {}

func TestRunParityCase_SkipsUnimplementedBackend(t *testing.T) {
	factory := func(backend string) (*Service, error) {
		switch backend {
		case "native-onnx":
			return &Service{
				runtime:   &parityRuntime{samples: []float32{0.25, -0.5, 0.75}},
				tokenizer: parityTokenizer{},
			}, nil
		case "native-safetensors":
			return nil, ErrBackendNotImplemented
		default:
			return nil, errors.New("unexpected backend")
		}
	}

	got, err := RunParityCase(
		factory,
		[]string{"native-onnx", "native-safetensors"},
		"One two three. Four five six.",
		"",
		42,
	)
	if err != nil {
		t.Fatalf("RunParityCase returned error: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("RunParityCase len = %d; want 2", len(got))
	}

	first := got[0]
	if first.Status != ParityStatusOK {
		t.Fatalf("first status = %q; want %q", first.Status, ParityStatusOK)
	}
	if first.Seed != 42 {
		t.Fatalf("first seed = %d; want 42", first.Seed)
	}
	if first.TokenCount == 0 || first.ChunkCount == 0 {
		t.Fatalf("first token/chunk counts should be > 0, got tokens=%d chunks=%d", first.TokenCount, first.ChunkCount)
	}
	if first.SampleCount != 3 {
		t.Fatalf("first sample count = %d; want 3", first.SampleCount)
	}
	if first.PCMHashSHA256 == "" {
		t.Fatal("first PCM hash should be non-empty")
	}

	second := got[1]
	if second.Status != ParityStatusSkipped {
		t.Fatalf("second status = %q; want %q", second.Status, ParityStatusSkipped)
	}
	if second.Reason == "" {
		t.Fatal("second skip reason should be non-empty")
	}
}

func TestParitySnapshots_SaveLoadRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "parity.json")
	in := []ParitySnapshot{
		{
			Backend:       "native-onnx",
			Seed:          7,
			TokenCount:    12,
			ChunkCount:    2,
			SampleCount:   960,
			PeakAbs:       0.9,
			RMS:           0.2,
			PCMHashSHA256: "abc123",
			Status:        ParityStatusOK,
		},
		{
			Backend: "native-safetensors",
			Seed:    7,
			Status:  ParityStatusSkipped,
			Reason:  "not implemented",
		},
	}

	if err := SaveParitySnapshots(path, in); err != nil {
		t.Fatalf("SaveParitySnapshots returned error: %v", err)
	}

	out, err := LoadParitySnapshots(path)
	if err != nil {
		t.Fatalf("LoadParitySnapshots returned error: %v", err)
	}
	if len(out) != len(in) {
		t.Fatalf("loaded snapshots len = %d; want %d", len(out), len(in))
	}
	if out[0].Backend != in[0].Backend || out[0].PCMHashSHA256 != in[0].PCMHashSHA256 {
		t.Fatalf("loaded first snapshot mismatch: got %+v want %+v", out[0], in[0])
	}
	if out[1].Status != ParityStatusSkipped {
		t.Fatalf("loaded second status = %q; want %q", out[1].Status, ParityStatusSkipped)
	}
}
