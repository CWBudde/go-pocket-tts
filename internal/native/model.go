package native

import (
	"fmt"
	"math/rand"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
	"github.com/example/go-pocket-tts/internal/safetensors"
)

type Config struct {
	FlowLM FlowLMConfig
	Mimi   MimiConfig
}

func DefaultConfig() Config {
	return Config{
		FlowLM: DefaultFlowLMConfig(),
		Mimi:   DefaultMimiConfig(),
	}
}

type Model struct {
	store *safetensors.Store
	flow  *FlowLM
	mimi  *MimiModel
}

func LoadModelFromSafetensors(path string, cfg Config) (*Model, error) {
	store, err := safetensors.OpenStore(path, safetensors.StoreOptions{})
	if err != nil {
		return nil, err
	}
	return LoadModelFromStore(store, cfg)
}

func LoadModelFromStore(store *safetensors.Store, cfg Config) (*Model, error) {
	if cfg.FlowLM.DModel == 0 {
		cfg = DefaultConfig()
	}
	vb := NewVarBuilder(store)
	flow, err := LoadFlowLM(vb, cfg.FlowLM)
	if err != nil {
		return nil, err
	}
	mimi, err := LoadMimiModel(vb, cfg.Mimi)
	if err != nil {
		return nil, err
	}
	return &Model{store: store, flow: flow, mimi: mimi}, nil
}

func (m *Model) Close() {
	if m != nil && m.store != nil {
		m.store.Close()
	}
}

func (m *Model) FlowLM() *FlowLM  { return m.flow }
func (m *Model) Mimi() *MimiModel { return m.mimi }

func (m *Model) TextEmbeddings(tokenIDs []int64) (*tensor.Tensor, error) {
	if m == nil || m.flow == nil {
		return nil, fmt.Errorf("native: model flow_lm unavailable")
	}
	return m.flow.TextEmbeddings(tokenIDs)
}

func (m *Model) FlowMain(sequence, textEmbeddings *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	if m == nil || m.flow == nil {
		return nil, nil, fmt.Errorf("native: model flow_lm unavailable")
	}
	return m.flow.FlowMain(sequence, textEmbeddings)
}

func (m *Model) FlowDirection(condition, s, t, x *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil || m.flow == nil {
		return nil, fmt.Errorf("native: model flow_lm unavailable")
	}
	return m.flow.FlowDirection(condition, s, t, x)
}

func (m *Model) SampleNextLatent(sequence, textEmbeddings *tensor.Tensor, decodeSteps int, eosThreshold, temperature float32, rng *rand.Rand) (*tensor.Tensor, bool, error) {
	if m == nil || m.flow == nil {
		return nil, false, fmt.Errorf("native: model flow_lm unavailable")
	}
	return m.flow.SampleNextLatent(sequence, textEmbeddings, decodeSteps, eosThreshold, temperature, rng)
}

func (m *Model) NewFlowState() (*FlowLMState, error) {
	if m == nil || m.flow == nil {
		return nil, fmt.Errorf("native: model flow_lm unavailable")
	}
	return m.flow.InitState()
}

func (m *Model) PromptFlow(state *FlowLMState, textEmbeddings *tensor.Tensor) error {
	if m == nil || m.flow == nil {
		return fmt.Errorf("native: model flow_lm unavailable")
	}
	return m.flow.PromptText(state, textEmbeddings)
}

func (m *Model) SampleNextLatentStateful(state *FlowLMState, sequenceFrame *tensor.Tensor, decodeSteps int, eosThreshold, temperature float32, rng *rand.Rand) (*tensor.Tensor, bool, error) {
	if m == nil || m.flow == nil {
		return nil, false, fmt.Errorf("native: model flow_lm unavailable")
	}
	return m.flow.SampleNextLatentStateful(state, sequenceFrame, decodeSteps, eosThreshold, temperature, rng)
}

// LatentToMimi maps FlowLM latents [B, T, 32] to Mimi latents [B, 512, T].
func (m *Model) LatentToMimi(latent *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil || m.flow == nil || m.mimi == nil {
		return nil, fmt.Errorf("native: model is not fully initialized")
	}
	if latent == nil {
		return nil, fmt.Errorf("native: latent tensor is nil")
	}
	shape := latent.Shape()
	if len(shape) != 3 || shape[2] != m.flow.cfg.LDim {
		return nil, fmt.Errorf("native: latent shape must be [B,T,%d], got %v", m.flow.cfg.LDim, shape)
	}
	denorm, err := tensor.BroadcastMul(latent, m.flow.embStd)
	if err != nil {
		return nil, err
	}
	denorm, err = tensor.BroadcastAdd(denorm, m.flow.embMean)
	if err != nil {
		return nil, err
	}
	// [B, T, 32] -> [B, 32, T]
	denorm, err = denorm.Transpose(1, 2)
	if err != nil {
		return nil, err
	}
	return m.mimi.QuantizerProject(denorm)
}

// MimiDecode maps [B, 512, T] to [B, 1, N] PCM-like output.
func (m *Model) MimiDecode(mimiLatent *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil || m.mimi == nil {
		return nil, fmt.Errorf("native: model mimi unavailable")
	}
	return m.mimi.DecodeFromLatent(mimiLatent)
}

// EncodeVoiceHook exposes the future voice encoder path (Phase 20 target).
func (m *Model) EncodeVoiceHook(audio *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil || m.mimi == nil {
		return nil, fmt.Errorf("native: model mimi unavailable")
	}
	return m.mimi.EncodeToLatent(audio)
}
