package native

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

// requireGermanCheckpoint returns models/german/model.safetensors, as written
// by `pockettts model download --language german`, or skips.
func requireGermanCheckpoint(t *testing.T) string {
	t.Helper()

	candidates := []string{
		filepath.Join("models", "german", "model.safetensors"),
		filepath.Join("..", "..", "models", "german", "model.safetensors"),
	}
	for _, path := range candidates {
		_, err := os.Stat(path)
		if err == nil {
			return path
		}
	}

	t.Skipf("german checkpoint not available in any expected location: %v", candidates)

	return ""
}

// TestLoadModelFromStore_ReleasesCheckpointBytes checks that a loaded model
// does not keep the checkpoint bytes alive next to its decoded weights: the
// web app cannot hold both for the 24-layer models in 4 GB of WASM memory.
func TestLoadModelFromStore_ReleasesCheckpointBytes(t *testing.T) {
	data, err := os.ReadFile(requireGermanCheckpoint(t))
	if err != nil {
		t.Fatal(err)
	}

	released := make(chan struct{})
	runtime.AddCleanup(&data[0], func(ch chan struct{}) { close(ch) }, released)

	store, err := safetensors.OpenStoreFromBytes(data, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("open store: %v", err)
	}

	mc, err := modelcfg.Lookup("german")
	if err != nil {
		t.Fatal(err)
	}

	model, err := LoadModelFromStore(store, ConfigFor(mc))
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	data, store = nil, nil //nolint:wastedassign,ineffassign // drop the test's own references

	for range 20 {
		runtime.GC()

		select {
		case <-released:
			runtime.KeepAlive(model)
			return
		case <-time.After(50 * time.Millisecond):
		}
	}

	runtime.KeepAlive(model)
	t.Fatal("the loaded model keeps the checkpoint bytes alive")
}

// Every config except english_2026-01 sets mimi.inner_dim: 32. Only the
// encoder side depends on it (mimi.downsample and flow_lm.speaker_proj_weight);
// the decoder that LatentToMimi/MimiDecode run is the same as before.
func TestGermanCheckpoint_InnerDimOnlyChangesEncoderSide(t *testing.T) {
	store, err := safetensors.OpenStore(requireGermanCheckpoint(t), safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	for name, want := range map[string][]int64{
		"mimi.downsample.conv.conv.weight":   {32, 512, 32},
		"flow_lm.speaker_proj_weight":        {1024, 32},
		"mimi.quantizer.output_proj.weight":  {512, 32, 1},
		"mimi.upsample.convtr.convtr.weight": {512, 1, 32},
	} {
		tns, err := store.Tensor(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		if !slices.Equal(tns.Shape, want) {
			t.Errorf("%s shape = %v, want %v", name, tns.Shape, want)
		}
	}
}

func TestLatentToMimiAndDecode_RealGermanCheckpoint(t *testing.T) {
	m, err := LoadModelFromSafetensors(requireGermanCheckpoint(t), DefaultConfig())
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	defer m.Close()

	const frames = 3

	data := make([]float32, frames*32)
	for i := range data {
		data[i] = float32(math.Sin(float64(i))) // non-zero, deterministic
	}

	latent, err := tensor.New(data, []int64{1, frames, 32})
	if err != nil {
		t.Fatalf("latent: %v", err)
	}

	mimiLatent, err := m.LatentToMimi(latent)
	if err != nil {
		t.Fatalf("latent_to_mimi: %v", err)
	}

	if got := mimiLatent.Shape(); !slices.Equal(got, []int64{1, 512, frames}) {
		t.Fatalf("mimi latent shape = %v", got)
	}

	audio, err := m.MimiDecode(mimiLatent)
	if err != nil {
		t.Fatalf("mimi decode: %v", err)
	}

	// 24 kHz at 12.5 latent frames per second = 1920 samples per frame.
	if got := audio.Shape(); !slices.Equal(got, []int64{1, 1, frames * 1920}) {
		t.Fatalf("audio shape = %v", got)
	}

	nonZero := 0

	for i, v := range audio.RawData() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("audio[%d] = %v", i, v)
		}

		if v != 0 {
			nonZero++
		}
	}

	if nonZero == 0 {
		t.Fatal("decoded audio is all zeros")
	}
}

func TestFlowStateFromVoiceModelState_RealGermanVoice(t *testing.T) {
	voicePath := filepath.Join("..", "..", "voices", "german", "juergen.safetensors")

	_, err := os.Stat(voicePath)
	if err != nil {
		t.Skipf("german voice not available: %v", err)
	}

	m, err := LoadModelFromSafetensors(requireGermanCheckpoint(t), DefaultConfig())
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	defer m.Close()

	voice, err := safetensors.LoadVoiceModelState(voicePath)
	if err != nil {
		t.Fatalf("load voice: %v", err)
	}

	state, err := m.NewFlowStateFromVoiceModelState(voice)
	if err != nil {
		t.Fatalf("NewFlowStateFromVoiceModelState: %v", err)
	}

	if got := state.Offset(); got != 124 {
		t.Fatalf("state offset = %d, want 124", got)
	}
}
