package native

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

// requireRepoFile returns the file at the repo-relative path parts, looked up
// from the package directory and the repo root, or skips.
func requireRepoFile(t *testing.T, parts ...string) string {
	t.Helper()

	rel := filepath.Join(parts...)
	for _, path := range []string{rel, filepath.Join("..", "..", rel)} {
		_, err := os.Stat(path)
		if err == nil {
			return path
		}
	}

	t.Skipf("%s not available (pockettts model download / pockettts-tools voice download)", rel)

	return ""
}

// loadLanguageModel loads models/<language>/model.safetensors with the
// native config of the embedded model config, or skips.
func loadLanguageModel(t *testing.T, language string) (*Model, string) {
	t.Helper()

	path := requireRepoFile(t, "models", language, "model.safetensors")

	mc, err := modelcfg.Lookup(language)
	if err != nil {
		t.Fatal(err)
	}

	m, err := LoadModelFromSafetensors(path, ConfigFor(mc))
	if err != nil {
		t.Fatalf("load %s: %v", language, err)
	}

	t.Cleanup(m.Close)

	return m, path
}

// voiceFlowState loads voices/<language>/<voice>.safetensors into a flow
// state, or skips.
func voiceFlowState(t *testing.T, m *Model, language, voice string) *FlowLMState {
	t.Helper()

	vs, err := safetensors.LoadVoiceModelState(requireRepoFile(t, "voices", language, voice+".safetensors"))
	if err != nil {
		t.Fatalf("load voice: %v", err)
	}

	state, err := m.NewFlowStateFromVoiceModelState(vs)
	if err != nil {
		t.Fatalf("NewFlowStateFromVoiceModelState: %v", err)
	}

	return state
}

// smokeGenerate prompts three text tokens, samples frames latents from the
// BOS frame and decodes them, checking shapes and that the audio is finite
// and not silent.
func smokeGenerate(t *testing.T, m *Model, state *FlowLMState, frames int) {
	t.Helper()

	text, err := m.TextEmbeddings([]int64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}

	err = m.PromptFlow(state, text)
	if err != nil {
		t.Fatalf("PromptFlow: %v", err)
	}

	bos := make([]float32, 32)
	for i := range bos {
		bos[i] = float32(math.NaN())
	}

	frame, err := tensor.New(bos, []int64{1, 1, 32})
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(1))
	latents := make([]*tensor.Tensor, 0, frames)

	for i := range frames {
		frame, _, err = m.SampleNextLatentStateful(state, frame, 1, -4, 0.3, rng)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}

		if !slices.Equal(frame.Shape(), []int64{1, 1, 32}) {
			t.Fatalf("step %d latent shape = %v", i, frame.Shape())
		}

		latents = append(latents, frame)
	}

	latent, err := tensor.Concat(latents, 1)
	if err != nil {
		t.Fatal(err)
	}

	mimiLatent, err := m.LatentToMimi(latent)
	if err != nil {
		t.Fatalf("LatentToMimi: %v", err)
	}

	audio, err := m.MimiDecode(mimiLatent)
	if err != nil {
		t.Fatalf("MimiDecode: %v", err)
	}

	if want := []int64{1, 1, int64(frames) * 1920}; !slices.Equal(audio.Shape(), want) {
		t.Fatalf("audio shape = %v, want %v", audio.Shape(), want)
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

func TestDriftingSamplerHead_RealCheckpoint(t *testing.T) {
	m, path := loadLanguageModel(t, "english_drifting_26-09")

	if got := len(m.flow.flowNet.timeEmbeds); got != 0 {
		t.Fatalf("drifting flow_net has %d time embeddings, want 0", got)
	}

	if m.flow.cfg.FlowType != modelcfg.FlowTypeDrifting {
		t.Fatalf("FlowType = %q, want %q", m.flow.cfg.FlowType, modelcfg.FlowTypeDrifting)
	}

	_, err := LoadModelFromSafetensors(path, DefaultConfig())
	if err == nil || !strings.Contains(err.Error(), "time_embed") {
		t.Fatalf("loading the drifting checkpoint as lsd: err = %v; want a time_embed count error", err)
	}

	smokeGenerate(t, m, voiceFlowState(t, m, "english_drifting_26-09", "alba"), 3)
}

// german_24l needs no new modules: the 24 transformer layers are detected
// from the weights, and its voices carry 24-layer KV caches.
func TestGerman24L_RealCheckpoint(t *testing.T) {
	m, _ := loadLanguageModel(t, "german_24l")

	if got := len(m.flow.transformer.layers); got != 24 {
		t.Fatalf("flow transformer layers = %d, want 24", got)
	}

	state := voiceFlowState(t, m, "german_24l", "juergen")
	if got := len(state.transformer.layers); got != 24 {
		t.Fatalf("voice state layers = %d, want 24", got)
	}

	smokeGenerate(t, m, state, 2)
}
