package native

import (
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

func TestLoadBOSBeforeVoice(t *testing.T) {
	blob := buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"flow_lm.bos_before_voice": {dtype: "F32", shape: []int64{1, 1, 2}, data: f32Bytes([]float32{7, 8})},
	})

	flow := NewVarBuilder(mustStore(t, blob)).Path("flow_lm")

	got, err := loadBOSBeforeVoice(flow, FlowLMConfig{DModel: 2, InsertBOSBeforeVoice: true})
	if err != nil {
		t.Fatalf("flag set: %v", err)
	}

	if got == nil || !slices.Equal(got.Shape(), []int64{1, 1, 2}) || !slices.Equal(got.RawData(), []float32{7, 8}) {
		t.Fatalf("bos_before_voice = %v", got)
	}

	got, err = loadBOSBeforeVoice(flow, FlowLMConfig{DModel: 2})
	if err != nil || got != nil {
		t.Fatalf("flag unset: got %v, %v; want nil, nil", got, err)
	}

	_, err = loadBOSBeforeVoice(flow, FlowLMConfig{DModel: 3, InsertBOSBeforeVoice: true})
	if err == nil {
		t.Fatal("wrong DModel: want shape error")
	}

	empty := NewVarBuilder(mustStore(t, buildSafetensors(t, map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"flow_lm.bos_emb": {dtype: "F32", shape: []int64{2}, data: f32Bytes([]float32{1, 2})},
	}))).Path("flow_lm")

	_, err = loadBOSBeforeVoice(empty, FlowLMConfig{DModel: 2, InsertBOSBeforeVoice: true})
	if err == nil || !strings.Contains(err.Error(), "bos_before_voice") {
		t.Fatalf("missing tensor: err = %v; want one naming bos_before_voice", err)
	}
}

func TestFlowLMVoicePrompt_BOSBeforeVoice(t *testing.T) {
	voice, err := tensor.New([]float32{1, 2, 3, 4}, []int64{1, 2, 2})
	if err != nil {
		t.Fatal(err)
	}

	bos, err := tensor.New([]float32{7, 8}, []int64{1, 1, 2})
	if err != nil {
		t.Fatal(err)
	}

	with := &FlowLM{bosBeforeVoice: bos, cfg: FlowLMConfig{DModel: 2, InsertBOSBeforeVoice: true}}

	got, err := with.VoicePrompt(voice)
	if err != nil {
		t.Fatalf("VoicePrompt: %v", err)
	}

	if !slices.Equal(got.Shape(), []int64{1, 3, 2}) || !slices.Equal(got.RawData(), []float32{7, 8, 1, 2, 3, 4}) {
		t.Fatalf("VoicePrompt = %v %v; want [1 3 2] [7 8 1 2 3 4]", got.Shape(), got.RawData())
	}

	without := &FlowLM{cfg: FlowLMConfig{DModel: 2}}

	got, err = without.VoicePrompt(voice)
	if err != nil || got != voice {
		t.Fatalf("VoicePrompt without flag = %v, %v; want the voice embedding unchanged", got, err)
	}
}

func TestConfigFor_BOSBeforeVoice(t *testing.T) {
	for _, tc := range []struct {
		language string
		want     bool
	}{
		{"english_2026-01", false},
		{"english_2026-09", true},
		{"german", true},
	} {
		mc, err := modelcfg.Lookup(tc.language)
		if err != nil {
			t.Fatal(err)
		}

		cfg := ConfigFor(mc)
		if cfg.FlowLM.InsertBOSBeforeVoice != tc.want {
			t.Errorf("ConfigFor(%s).FlowLM.InsertBOSBeforeVoice = %v, want %v", tc.language, cfg.FlowLM.InsertBOSBeforeVoice, tc.want)
		}

		def := DefaultConfig()
		cfg.FlowLM.InsertBOSBeforeVoice = def.FlowLM.InsertBOSBeforeVoice

		if cfg != def {
			t.Errorf("ConfigFor(%s) changes more than InsertBOSBeforeVoice: %+v", tc.language, cfg)
		}
	}

	if ConfigFor(nil) != DefaultConfig() {
		t.Error("ConfigFor(nil) != DefaultConfig()")
	}
}

func TestVoicePrompt_RealGermanCheckpoint(t *testing.T) {
	path := requireGermanCheckpoint(t)

	mc, err := modelcfg.Lookup("german")
	if err != nil {
		t.Fatal(err)
	}

	m, err := LoadModelFromSafetensors(path, ConfigFor(mc))
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	defer m.Close()

	voice, err := tensor.New(make([]float32, 2*1024), []int64{1, 2, 1024})
	if err != nil {
		t.Fatal(err)
	}

	prompt, err := m.VoicePrompt(voice)
	if err != nil {
		t.Fatalf("VoicePrompt: %v", err)
	}

	if !slices.Equal(prompt.Shape(), []int64{1, 3, 1024}) {
		t.Fatalf("prompt shape = %v, want [1 3 1024]", prompt.Shape())
	}

	store, err := safetensors.OpenStore(path, safetensors.StoreOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	bos, err := store.Tensor("flow_lm.bos_before_voice")
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(prompt.RawData()[:1024], bos.Data) {
		t.Fatal("prompt frame 0 is not flow_lm.bos_before_voice")
	}

	state, err := m.NewFlowState()
	if err != nil {
		t.Fatal(err)
	}

	text, err := m.TextEmbeddings([]int64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}

	seq, err := tensor.Concat([]*tensor.Tensor{prompt, text}, 1)
	if err != nil {
		t.Fatal(err)
	}

	err = m.PromptFlow(state, seq)
	if err != nil {
		t.Fatalf("PromptFlow: %v", err)
	}

	if got := state.Offset(); got != 1+2+3 {
		t.Fatalf("state offset = %d, want 6 (bos + 2 voice + 3 text)", got)
	}
}

func mustStore(t *testing.T, blob []byte) *safetensors.Store {
	t.Helper()

	st, err := safetensors.OpenStoreFromBytes(blob, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("open store: %v", err)
	}

	return st
}
