package native

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/audio"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/ops"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
)

// voiceEncoderFixture is written by scripts/dump_voice_encoder_parity.py.
type voiceEncoderFixture struct {
	Source struct {
		Upstream string `json:"upstream"`
		Config   string `json:"config"`
		Prompt   string `json:"prompt"`
	} `json:"source"`
	Prepared struct {
		Length int       `json:"length"`
		Sum    float64   `json:"sum"`
		Head   []float32 `json:"head"`
		Tail   []float32 `json:"tail"`
	} `json:"prepared"`
	Latent           tensorJSON `json:"latent"`
	ConditioningRows struct {
		Frames []int64   `json:"frames"`
		Shape  []int64   `json:"shape"`
		Data   []float32 `json:"data"`
	} `json:"conditioning_rows"`
}

// gatedModelsDir holds checkpoints of the gated kyutai/pocket-tts repo, one
// directory per language (the ungated ones have the Mimi encoder zeroed).
func gatedModelsDir() string {
	if dir := os.Getenv("POCKETTTS_GATED_MODELS"); dir != "" {
		return dir
	}

	return filepath.Join("..", "..", "models", "gated")
}

// TestPythonParity_MimiEncoder runs the native voice encoder on the fixture
// prompt and compares the prepared audio, the Mimi latent and the speaker
// conditioning with upstream. It needs the gated checkpoints; see
// scripts/dump_voice_encoder_parity.py.
func TestPythonParity_MimiEncoder(t *testing.T) {
	t.Parallel()

	paths, err := filepath.Glob(filepath.Join("testdata", "python_parity", "encoder_*.json"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no encoder fixtures: %v", err)
	}

	for _, path := range paths {
		var fx voiceEncoderFixture

		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}

		err = json.Unmarshal(data, &fx)
		if err != nil {
			t.Fatalf("decode %s: %v", path, err)
		}

		t.Run(fx.Source.Config, func(t *testing.T) {
			t.Parallel()
			checkVoiceEncoderParity(t, &fx)
		})
	}
}

func checkVoiceEncoderParity(t *testing.T, fx *voiceEncoderFixture) {
	t.Helper()

	weights, _ := filepath.Glob(filepath.Join(gatedModelsDir(), fx.Source.Config, "*.safetensors"))
	if len(weights) != 1 {
		t.Skipf("gated %s checkpoint not found in %s (set POCKETTTS_GATED_MODELS)", fx.Source.Config, gatedModelsDir())
	}

	enc, err := LoadVoiceEncoderFromSafetensors(weights[0], DefaultMimiConfig())
	if errors.Is(err, ErrMimiEncoderWeightsZeroed) {
		t.Skipf("%s is not a gated checkpoint: %v", weights[0], err)
	}

	if err != nil {
		t.Fatalf("load voice encoder: %v", err)
	}

	wav, err := os.ReadFile(filepath.Join("testdata", "python_parity", fx.Source.Prompt))
	if err != nil {
		t.Fatalf("read prompt: %v", err)
	}

	samples, rate, err := audio.DecodePromptWAV(wav)
	if err != nil {
		t.Fatalf("decode prompt: %v", err)
	}

	prepared, err := audio.PrepareVoicePrompt(samples, rate)
	if err != nil {
		t.Fatalf("prepare prompt: %v", err)
	}

	checkPreparedPrompt(t, prepared, fx)

	latent, err := enc.EncodeToLatent(prepared)
	if err != nil {
		t.Fatalf("EncodeToLatent: %v", err)
	}

	wantLatent, err := fx.Latent.tensor()
	if err != nil {
		t.Fatalf("latent fixture: %v", err)
	}

	// The SEANet convs, 2 transformer layers and the downsample conv add up to
	// about the error of the Mimi decoder path.
	tol := ops.Tolerance{Abs: 2e-4, Rel: 1e-3}
	assertTensorParity(t, "encode_to_latent", latent, wantLatent, tol)

	cond, err := enc.Condition(latent)
	if err != nil {
		t.Fatalf("Condition: %v", err)
	}

	rows := conditioningRows(t, cond, fx.ConditioningRows.Frames)

	wantRows, err := tensor.New(fx.ConditioningRows.Data, fx.ConditioningRows.Shape)
	if err != nil {
		t.Fatalf("conditioning fixture: %v", err)
	}

	assertTensorParity(t, "_encode_audio", rows, wantRows, tol)
}

func checkPreparedPrompt(t *testing.T, prepared []float32, fx *voiceEncoderFixture) {
	t.Helper()

	if len(prepared) != fx.Prepared.Length {
		t.Fatalf("prepared prompt has %d samples, upstream %d", len(prepared), fx.Prepared.Length)
	}

	var sum float64
	for _, v := range prepared {
		sum += float64(v)
	}

	if math.Abs(sum-fx.Prepared.Sum) > 1e-4 {
		t.Errorf("prepared prompt sum %g, upstream %g", sum, fx.Prepared.Sum)
	}

	head, tail := prepared[:len(fx.Prepared.Head)], prepared[len(prepared)-len(fx.Prepared.Tail):]
	for i := range head {
		if math.Abs(float64(head[i]-fx.Prepared.Head[i])) > 1e-7 || math.Abs(float64(tail[i]-fx.Prepared.Tail[i])) > 1e-7 {
			t.Fatalf("prepared prompt edge %d: head %g/%g tail %g/%g (go/upstream)",
				i, head[i], fx.Prepared.Head[i], tail[i], fx.Prepared.Tail[i])
		}
	}
}

// conditioningRows picks frames of a [1,T,D] conditioning as [len(frames),D].
func conditioningRows(t *testing.T, cond *tensor.Tensor, frames []int64) *tensor.Tensor {
	t.Helper()

	shape := cond.Shape()
	d := shape[2]
	raw := cond.RawData()
	out := make([]float32, 0, int64(len(frames))*d)

	for _, f := range frames {
		if f >= shape[1] {
			t.Fatalf("conditioning has %d frames, fixture row %d", shape[1], f)
		}

		out = append(out, raw[f*d:(f+1)*d]...)
	}

	rows, err := tensor.New(out, []int64{int64(len(frames)), d})
	if err != nil {
		t.Fatalf("conditioning rows: %v", err)
	}

	return rows
}
