package native

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/audio"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/ops"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
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
	ModelState *voiceModelStateFixture `json:"model_state"`
}

// voiceModelStateFixture is the state upstream get_state_for_audio_prompt
// builds from the prompt, as export_model_state writes it: per module the
// layout of each tensor, the offset and pad, and the K and V cache rows
// [len(positions), H, Dh] at positions.
type voiceModelStateFixture struct {
	Positions []int64 `json:"positions"`
	Modules   []struct {
		Module  string `json:"module"`
		Tensors map[string]struct {
			DType string  `json:"dtype"`
			Shape []int64 `json:"shape"`
		} `json:"tensors"`
		Offset int64      `json:"offset"`
		Pad    int64      `json:"pad"`
		KRows  tensorJSON `json:"k_rows"`
		VRows  tensorJSON `json:"v_rows"`
	} `json:"modules"`
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

	for _, fx := range loadVoiceEncoderFixtures(t) {
		t.Run(fx.Source.Config, func(t *testing.T) {
			t.Parallel()
			checkVoiceEncoderParity(t, fx)
		})
	}
}

// loadVoiceEncoderFixtures reads every testdata/python_parity/encoder_*.json.
func loadVoiceEncoderFixtures(t *testing.T) []*voiceEncoderFixture {
	t.Helper()

	paths, err := filepath.Glob(filepath.Join("testdata", "python_parity", "encoder_*.json"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no encoder fixtures: %v", err)
	}

	fixtures := make([]*voiceEncoderFixture, 0, len(paths))

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

		fixtures = append(fixtures, &fx)
	}

	return fixtures
}

// gatedCheckpoint returns the gated checkpoint of config, or skips.
func gatedCheckpoint(t *testing.T, config string) string {
	t.Helper()

	weights, _ := filepath.Glob(filepath.Join(gatedModelsDir(), config, "*.safetensors"))
	if len(weights) != 1 {
		t.Skipf("gated %s checkpoint not found in %s (set POCKETTTS_GATED_MODELS)", config, gatedModelsDir())
	}

	return weights[0]
}

// loadGatedVoiceEncoder loads the voice encoder of a gated checkpoint, or
// skips when the checkpoint's encoder is zeroed.
func loadGatedVoiceEncoder(t *testing.T, vb *VarBuilder, path string) *VoiceEncoder {
	t.Helper()

	enc, err := LoadVoiceEncoder(vb, DefaultMimiConfig())
	if errors.Is(err, ErrMimiEncoderWeightsZeroed) {
		t.Skipf("%s is not a gated checkpoint: %v", path, err)
	}

	if err != nil {
		t.Fatalf("load voice encoder: %v", err)
	}

	return enc
}

// preparedFixturePrompt reads the fixture's prompt WAV and prepares it like
// get_state_for_audio_prompt.
func preparedFixturePrompt(t *testing.T, fx *voiceEncoderFixture) []float32 {
	t.Helper()

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

	return prepared
}

func checkVoiceEncoderParity(t *testing.T, fx *voiceEncoderFixture) {
	t.Helper()

	weights := gatedCheckpoint(t, fx.Source.Config)

	store, err := safetensors.OpenStore(weights, safetensors.StoreOptions{})
	if err != nil {
		t.Fatal(err)
	}

	enc := loadGatedVoiceEncoder(t, NewVarBuilder(store), weights)
	store.Close()

	prepared := preparedFixturePrompt(t, fx)
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

// TestPythonParity_VoiceModelState builds the voice model state from the
// fixture prompt like export-voice --format model-state and compares the file
// it writes with upstream get_state_for_audio_prompt + export_model_state:
// the modules, every tensor's dtype and shape, the offsets and pads, and the
// K and V cache rows. It needs the gated checkpoints, like
// TestPythonParity_MimiEncoder.
func TestPythonParity_VoiceModelState(t *testing.T) {
	t.Parallel()

	for _, fx := range loadVoiceEncoderFixtures(t) {
		t.Run(fx.Source.Config, func(t *testing.T) {
			t.Parallel()

			if fx.ModelState == nil {
				t.Fatal("fixture has no model_state; regenerate it with scripts/dump_voice_encoder_parity.py")
			}

			checkVoiceModelStateParity(t, fx)
		})
	}
}

func checkVoiceModelStateParity(t *testing.T, fx *voiceEncoderFixture) {
	t.Helper()

	weights := gatedCheckpoint(t, fx.Source.Config)

	mc, err := modelcfg.Lookup(fx.Source.Config)
	if err != nil {
		t.Fatal(err)
	}

	store, err := safetensors.OpenStore(weights, safetensors.StoreOptions{})
	if err != nil {
		t.Fatal(err)
	}

	enc := loadGatedVoiceEncoder(t, NewVarBuilder(store), weights)

	m, err := LoadModelFromStore(store, ConfigFor(mc))
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	store.Close()

	cond, err := enc.Encode(preparedFixturePrompt(t, fx))
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	state, err := m.PromptVoice(cond)
	if err != nil {
		t.Fatalf("PromptVoice: %v", err)
	}

	vs, err := state.VoiceModelState()
	if err != nil {
		t.Fatalf("VoiceModelState: %v", err)
	}

	blob, err := safetensors.EncodeVoiceModelState(vs)
	if err != nil {
		t.Fatalf("EncodeVoiceModelState: %v", err)
	}

	exported, err := safetensors.OpenStoreFromBytes(blob, safetensors.StoreOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer exported.Close()

	var wantNames []string

	for _, mod := range fx.ModelState.Modules {
		for key := range mod.Tensors {
			wantNames = append(wantNames, mod.Module+"/"+key)
		}
	}

	slices.Sort(wantNames)

	if got := exported.Names(); !slices.Equal(got, wantNames) {
		t.Fatalf("exported tensors %v, upstream %v", got, wantNames)
	}

	// The cache holds the K/V of 6 transformer layers on top of the
	// encoder's output, so it carries about the encoder's error.
	tol := ops.Tolerance{Abs: 2e-4, Rel: 1e-3}

	for _, mod := range fx.ModelState.Modules {
		tensors := make(map[string]*safetensors.Tensor, len(mod.Tensors))

		for key, want := range mod.Tensors {
			got, err := exported.Tensor(mod.Module + "/" + key)
			if err != nil {
				t.Fatal(err)
			}

			if got.DType != want.DType || !slices.Equal(got.Shape, want.Shape) {
				t.Fatalf("%s/%s = %s %v, upstream %s %v", mod.Module, key, got.DType, got.Shape, want.DType, want.Shape)
			}

			tensors[key] = got
		}

		if got := tensors["offset"].Data[0]; int64(got) != mod.Offset {
			t.Fatalf("%s offset = %v, upstream %d", mod.Module, got, mod.Offset)
		}

		if got := tensors["pad"].Data[0]; int64(got) != mod.Pad {
			t.Fatalf("%s pad = %v, upstream %d", mod.Module, got, mod.Pad)
		}

		for kv, rows := range []tensorJSON{mod.KRows, mod.VRows} {
			want, err := rows.tensor()
			if err != nil {
				t.Fatal(err)
			}

			name := mod.Module + []string{" K", " V"}[kv]
			assertTensorParity(t, name, cacheRows(t, tensors["cache"], kv, fx.ModelState.Positions), want, tol)
		}
	}
}

// cacheRows picks positions of half kv (0 = K, 1 = V) of an upstream
// [2, 1, T, H, Dh] cache as [len(positions), H, Dh].
func cacheRows(t *testing.T, cache *safetensors.Tensor, kv int, positions []int64) *tensor.Tensor {
	t.Helper()

	steps, heads, headDim := cache.Shape[2], cache.Shape[3], cache.Shape[4]
	row := heads * headDim
	half := cache.Data[int64(kv)*steps*row : int64(kv+1)*steps*row]
	out := make([]float32, 0, int64(len(positions))*row)

	for _, p := range positions {
		out = append(out, half[p*row:(p+1)*row]...)
	}

	return mustTensorN(t, out, []int64{int64(len(positions)), heads, headDim})
}
