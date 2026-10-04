package native

import (
	"errors"
	"fmt"
	"slices"

	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

// PromptVoice returns a fresh state prompted with the audio conditioning
// [1, T, DModel] (VoiceEncoder.Encode), like upstream
// get_state_for_audio_prompt: VoicePrompt prepends bos_before_voice when the
// config asks for it, then the conditioning is prefilled.
func (f *FlowLM) PromptVoice(conditioning *tensor.Tensor) (*FlowLMState, error) {
	if conditioning == nil {
		return nil, errors.New("native: voice conditioning is nil")
	}

	if shape := conditioning.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] == 0 {
		return nil, fmt.Errorf("native: voice conditioning must be [1,T,D] with T > 0, got %v", shape)
	}

	prompt, err := f.VoicePrompt(conditioning)
	if err != nil {
		return nil, err
	}

	state, err := f.InitState()
	if err != nil {
		return nil, err
	}

	err = f.PromptText(state, prompt)
	if err != nil {
		return nil, fmt.Errorf("native: prompt voice: %w", err)
	}

	return state, nil
}

// PromptVoice returns a state prompted with the audio conditioning; see
// FlowLM.PromptVoice.
func (m *Model) PromptVoice(conditioning *tensor.Tensor) (*FlowLMState, error) {
	if m == nil || m.flow == nil {
		return nil, errors.New("native: model flow_lm unavailable")
	}

	return m.flow.PromptVoice(conditioning)
}

// VoiceModelState returns the state as upstream export_model_state stores the
// flow_lm state: per attention module an int64 offset, an int64 pad of 0 and
// the cache [2, B, offset, H, Dh] (K, then V) trimmed to the prompted
// positions. It is the inverse of FlowLM.InitStateFromVoiceModelState.
func (s *FlowLMState) VoiceModelState() (*safetensors.VoiceModelState, error) {
	if s == nil || s.transformer == nil || len(s.transformer.layers) == 0 {
		return nil, errors.New("native: flow_lm state unavailable")
	}

	out := &safetensors.VoiceModelState{Modules: make(map[string]map[string]*safetensors.Tensor, len(s.transformer.layers))}

	for i := range s.transformer.layers {
		module, err := voiceModuleFromLayerState(&s.transformer.layers[i])
		if err != nil {
			return nil, fmt.Errorf("native: export flow transformer layer %d: %w", i, err)
		}

		out.Modules[flowAttentionModuleName(i)] = module
	}

	return out, nil
}

// voiceModuleFromLayerState packs the first offset positions of the
// [B, H, Tk, Dh] key and value caches into upstream's [2, B, T, H, Dh] cache;
// splitVoiceKVCache unpacks it.
func voiceModuleFromLayerState(s *flowTransformerLayerState) (map[string]*safetensors.Tensor, error) {
	if s.kCache == nil || s.vCache == nil || s.offset <= 0 {
		return nil, errors.New("state holds no prompt")
	}

	shape := s.kCache.Shape()
	if !equalKVCacheLayout(shape, s.vCache.Shape()) {
		return nil, fmt.Errorf("KV cache shape mismatch k=%v v=%v", shape, s.vCache.Shape())
	}

	b, heads, capacity, headDim := shape[0], shape[1], shape[2], shape[3]
	steps := s.offset

	if steps > capacity {
		return nil, fmt.Errorf("offset %d exceeds cache length %d", steps, capacity)
	}

	kData, vData := s.kCache.RawData(), s.vCache.RawData()
	cache := make([]float32, 2*b*steps*heads*headDim)

	for batch := range b {
		for head := range heads {
			for step := range steps {
				src := int(((batch*heads+head)*capacity + step) * headDim)
				kDst := voiceKVIndex(0, batch, step, head, 0, b, steps, heads, headDim)
				vDst := voiceKVIndex(1, batch, step, head, 0, b, steps, heads, headDim)
				copy(cache[kDst:kDst+int(headDim)], kData[src:src+int(headDim)])
				copy(cache[vDst:vDst+int(headDim)], vData[src:src+int(headDim)])
			}
		}
	}

	return map[string]*safetensors.Tensor{
		"cache":  {DType: "F32", Shape: []int64{2, b, steps, heads, headDim}, Data: cache},
		"offset": {DType: "I64", Shape: []int64{b}, Data: slices.Repeat([]float32{float32(steps)}, int(b))},
		"pad":    {DType: "I64", Shape: []int64{b}, Data: make([]float32, b)},
	}, nil
}
