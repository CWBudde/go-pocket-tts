package native

import (
	"math"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/runtime/ops"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

func TestFlowLMInitState(t *testing.T) {
	var nilFlow *FlowLM

	_, err := nilFlow.InitState()
	if err == nil || !strings.Contains(err.Error(), "transformer unavailable") {
		t.Fatalf("expected nil flow transformer error, got: %v", err)
	}

	f := &FlowLM{}

	_, err = f.InitState()
	if err == nil || !strings.Contains(err.Error(), "transformer unavailable") {
		t.Fatalf("expected missing transformer error, got: %v", err)
	}

	f.transformer = &flowTransformer{layers: []*flowTransformerLayer{{}, {}}}

	state, err := f.InitState()
	if err != nil {
		t.Fatalf("InitState returned error: %v", err)
	}

	if state == nil || state.transformer == nil {
		t.Fatal("expected non-nil state and transformer cache")
	}

	if got := len(state.transformer.layers); got != 2 {
		t.Fatalf("state layer count = %d, want 2", got)
	}
}

func TestFlowLMPromptTextGuards(t *testing.T) {
	f := &FlowLM{cfg: FlowLMConfig{DModel: 4}}
	state := &FlowLMState{transformer: &flowTransformerState{}}

	_, err := f.TextEmbeddings([]int64{1})
	if err == nil || !strings.Contains(err.Error(), "not initialized") {
		t.Fatalf("expected TextEmbeddings not initialized error, got: %v", err)
	}

	err = f.PromptText(state, mustTensorN(t, []float32{1, 2, 3, 4}, []int64{1, 1, 4}))
	if err == nil || !strings.Contains(err.Error(), "transformer unavailable") {
		t.Fatalf("expected missing transformer error, got: %v", err)
	}

	f.transformer = &flowTransformer{}

	err = f.PromptText(nil, mustTensorN(t, []float32{1, 2, 3, 4}, []int64{1, 1, 4}))
	if err == nil || !strings.Contains(err.Error(), "state unavailable") {
		t.Fatalf("expected nil state error, got: %v", err)
	}

	err = f.PromptText(&FlowLMState{}, mustTensorN(t, []float32{1, 2, 3, 4}, []int64{1, 1, 4}))
	if err == nil || !strings.Contains(err.Error(), "state unavailable") {
		t.Fatalf("expected missing transformer state error, got: %v", err)
	}

	err = f.PromptText(state, nil)
	if err == nil || !strings.Contains(err.Error(), "embeddings are nil") {
		t.Fatalf("expected nil embeddings error, got: %v", err)
	}

	err = f.PromptText(state, mustTensorN(t, []float32{1, 2, 3, 4}, []int64{1, 4}))
	if err == nil || !strings.Contains(err.Error(), "must be [B,T,D]") {
		t.Fatalf("expected rank error, got: %v", err)
	}

	err = f.PromptText(state, mustTensorN(t, []float32{1, 2, 3, 4, 5}, []int64{1, 1, 5}))
	if err == nil || !strings.Contains(err.Error(), "width must be") {
		t.Fatalf("expected width mismatch error, got: %v", err)
	}

	empty := mustTensorN(t, []float32{}, []int64{1, 0, 4})

	err = f.PromptText(state, empty)
	if err != nil {
		t.Fatalf("PromptText empty sequence returned error: %v", err)
	}

	f.transformer.layers = []*flowTransformerLayer{{}}

	err = f.PromptText(state, mustTensorN(t, []float32{1, 2, 3, 4}, []int64{1, 1, 4}))
	if err == nil || !strings.Contains(err.Error(), "state layer count") {
		t.Fatalf("expected prefill layer-count error, got: %v", err)
	}
}

func TestFlowLSDDecodeGuards(t *testing.T) {
	f := &FlowLM{}
	cond := mustTensorN(t, []float32{0, 0}, []int64{1, 2})
	x0 := mustTensorN(t, []float32{1, 2}, []int64{1, 2})

	_, err := f.LSDDecode(cond, x0, 0)
	if err == nil || !strings.Contains(err.Error(), "steps must be >0") {
		t.Fatalf("expected step count error, got: %v", err)
	}

	badX0 := mustTensorN(t, []float32{1, 2}, []int64{1, 1, 2})

	_, err = f.LSDDecode(cond, badX0, 1)
	if err == nil || !strings.Contains(err.Error(), "must be [B, D]") {
		t.Fatalf("expected x0 rank error, got: %v", err)
	}

	_, err = f.LSDDecode(cond, x0, 1)
	if err == nil || !strings.Contains(err.Error(), "flow net unavailable") {
		t.Fatalf("expected flow net unavailable error, got: %v", err)
	}
}

func TestMakeGaussianNoise(t *testing.T) {
	_, err := makeGaussianNoise(0, 4, 1, nil)
	if err == nil || !strings.Contains(err.Error(), "invalid gaussian noise shape") {
		t.Fatalf("expected invalid shape error, got: %v", err)
	}

	a, err := makeGaussianNoise(2, 3, 1.0, nil)
	if err != nil {
		t.Fatalf("makeGaussianNoise returned error: %v", err)
	}

	b, err := makeGaussianNoise(2, 3, 1.0, nil)
	if err != nil {
		t.Fatalf("makeGaussianNoise returned error: %v", err)
	}

	if !equalApproxN(a.RawData(), b.RawData(), 0) {
		t.Fatal("expected deterministic noise when rng is nil")
	}

	zero, err := makeGaussianNoise(1, 4, -0.5, nil)
	if err != nil {
		t.Fatalf("makeGaussianNoise(negative temperature) error: %v", err)
	}

	for i, v := range zero.RawData() {
		if math.Abs(float64(v)) > 1e-7 {
			t.Fatalf("expected zero noise at negative temperature, idx=%d val=%v", i, v)
		}
	}
}

func TestFlowTransformerStateGuards(t *testing.T) {
	var nilTransformer *flowTransformer

	_, err := nilTransformer.initState()
	if err == nil || !strings.Contains(err.Error(), "flow transformer is nil") {
		t.Fatalf("expected nil transformer error, got: %v", err)
	}

	tfm := &flowTransformer{layers: []*flowTransformerLayer{{}, {}}}

	state, err := tfm.initState()
	if err != nil {
		t.Fatalf("initState returned error: %v", err)
	}

	if len(state.layers) != 2 {
		t.Fatalf("state layer count = %d, want 2", len(state.layers))
	}

	x := mustTensorN(t, []float32{1, 2, 3, 4}, []int64{1, 1, 4})

	err = nilTransformer.prefill(x, state)
	if err == nil || !strings.Contains(err.Error(), "flow transformer is nil") {
		t.Fatalf("expected nil transformer error, got: %v", err)
	}

	err = tfm.prefill(x, nil)
	if err == nil || !strings.Contains(err.Error(), "state is nil") {
		t.Fatalf("expected nil state error, got: %v", err)
	}

	err = tfm.prefill(x, &flowTransformerState{})
	if err == nil || !strings.Contains(err.Error(), "state layer count") {
		t.Fatalf("expected layer count mismatch error, got: %v", err)
	}

	_, err = nilTransformer.step(x, state)
	if err == nil || !strings.Contains(err.Error(), "flow transformer is nil") {
		t.Fatalf("expected nil transformer step error, got: %v", err)
	}

	_, err = tfm.step(x, nil)
	if err == nil || !strings.Contains(err.Error(), "state is nil") {
		t.Fatalf("expected nil state step error, got: %v", err)
	}

	_, err = tfm.step(x, &flowTransformerState{})
	if err == nil || !strings.Contains(err.Error(), "state layer count") {
		t.Fatalf("expected layer count step mismatch error, got: %v", err)
	}
}

func TestFlowTransformerAppendKVAndDetectNumHeads(t *testing.T) {
	var nilState *flowTransformerLayerState

	err := nilState.appendKV(nil, nil)
	if err == nil || !strings.Contains(err.Error(), "state is nil") {
		t.Fatalf("expected nil state appendKV error, got: %v", err)
	}

	s := &flowTransformerLayerState{}

	err = s.appendKV(nil, nil)
	if err == nil || !strings.Contains(err.Error(), "requires non-nil") {
		t.Fatalf("expected nil k/v appendKV error, got: %v", err)
	}

	k1 := mustTensorN(t, []float32{1}, []int64{1, 1, 1, 1})
	v1 := mustTensorN(t, []float32{2}, []int64{1, 1, 1, 1})

	err = s.appendKV(k1, v1)
	if err != nil {
		t.Fatalf("appendKV first call error: %v", err)
	}

	if s.offset != 1 {
		t.Fatalf("offset after first append = %d, want 1", s.offset)
	}

	k2 := mustTensorN(t, []float32{3, 4}, []int64{1, 1, 2, 1})
	v2 := mustTensorN(t, []float32{5, 6}, []int64{1, 1, 2, 1})

	err = s.appendKV(k2, v2)
	if err != nil {
		t.Fatalf("appendKV second call error: %v", err)
	}

	if s.offset != 3 {
		t.Fatalf("offset after second append = %d, want 3", s.offset)
	}

	if got := s.kCache.Shape()[2]; got < 3 {
		t.Fatalf("kCache sequence capacity = %d, want at least 3", got)
	}

	if got := detectNumHeads(nil, 8); got != 8 {
		t.Fatalf("detectNumHeads(nil) = %d, want 8", got)
	}

	if got := detectNumHeads(NewVarBuilder(nil), 6); got != 6 {
		t.Fatalf("detectNumHeads(uninitialized vb) = %d, want 6", got)
	}
}

func TestFlowTransformerInitStateFromVoiceModelState(t *testing.T) {
	tfm := &flowTransformer{
		layers: []*flowTransformerLayer{
			{nHeads: 2, headDim: 1},
			{nHeads: 2, headDim: 1},
		},
	}

	voiceState := &safetensors.VoiceModelState{Modules: map[string]map[string]*safetensors.Tensor{
		"transformer.layers.0.self_attn": {
			"cache": {
				Name:  "transformer.layers.0.self_attn/cache",
				Shape: []int64{2, 1, 2, 2, 1},
				Data: []float32{
					1, 2, 3, 4,
					5, 6, 7, 8,
				},
			},
			"offset": {
				Name:  "transformer.layers.0.self_attn/offset",
				Shape: []int64{1},
				Data:  []float32{1},
			},
		},
		"transformer.layers.1.self_attn": {
			"cache": {
				Name:  "transformer.layers.1.self_attn/cache",
				Shape: []int64{2, 1, 2, 2, 1},
				Data: []float32{
					9, 10, 11, 12,
					13, 14, 15, 16,
				},
			},
			"offset": {
				Name:  "transformer.layers.1.self_attn/offset",
				Shape: []int64{1},
				Data:  []float32{2},
			},
		},
	}}

	state, err := tfm.initStateFromVoiceModelState(voiceState)
	if err != nil {
		t.Fatalf("initStateFromVoiceModelState: %v", err)
	}

	if len(state.layers) != 2 {
		t.Fatalf("state layers = %d, want 2", len(state.layers))
	}

	if got := state.layers[0].offset; got != 1 {
		t.Fatalf("layer0 offset = %d, want 1", got)
	}

	if got := state.layers[0].kCache.Shape(); !shapeEqual(got, []int64{1, 2, 2, 1}) {
		t.Fatalf("layer0 key cache shape = %v, want [1 2 2 1]", got)
	}

	if got := state.layers[0].kCache.RawData(); !equalApproxN(got, []float32{1, 3, 2, 4}, 0) {
		t.Fatalf("layer0 key cache data = %v, want [1 3 2 4]", got)
	}

	if got := state.layers[0].vCache.RawData(); !equalApproxN(got, []float32{5, 7, 6, 8}, 0) {
		t.Fatalf("layer0 value cache data = %v, want [5 7 6 8]", got)
	}

	if got := state.layers[1].offset; got != 2 {
		t.Fatalf("layer1 offset = %d, want 2", got)
	}

	if got := state.layers[1].kCache.Shape(); !shapeEqual(got, []int64{1, 2, 2, 1}) {
		t.Fatalf("layer1 key cache shape = %v, want [1 2 2 1]", got)
	}
}

func TestFlowTransformerInitStateFromVoiceModelStateGuards(t *testing.T) {
	tfm := &flowTransformer{layers: []*flowTransformerLayer{{nHeads: 2, headDim: 1}}}

	_, err := tfm.initStateFromVoiceModelState(nil)
	if err == nil || !strings.Contains(err.Error(), "voice model state is nil") {
		t.Fatalf("expected nil voice model state error, got: %v", err)
	}

	_, err = tfm.initStateFromVoiceModelState(&safetensors.VoiceModelState{Modules: map[string]map[string]*safetensors.Tensor{}})
	if err == nil || !strings.Contains(err.Error(), "missing module") {
		t.Fatalf("expected missing module error, got: %v", err)
	}

	_, err = tfm.initStateFromVoiceModelState(&safetensors.VoiceModelState{Modules: map[string]map[string]*safetensors.Tensor{
		"transformer.layers.0.self_attn": {
			"cache": {
				Name:  "transformer.layers.0.self_attn/cache",
				Shape: []int64{2, 1, 1, 2, 1},
				Data:  []float32{1, 2, 3, 4},
			},
			"offset": {
				Name:  "transformer.layers.0.self_attn/offset",
				Shape: []int64{1},
				Data:  []float32{2},
			},
		},
	}})
	if err == nil || !strings.Contains(err.Error(), "exceeds cache length") {
		t.Fatalf("expected offset bounds error, got: %v", err)
	}
}

func TestFlowTransformerStatefulAttentionMatchesFullLastToken(t *testing.T) {
	inProjWeight, err := tensor.New([]float32{
		1, 0,
		0, 1,
		1, 0,
		0, 1,
		1, 0,
		0, 1,
	}, []int64{6, 2})
	if err != nil {
		t.Fatalf("inProj weight: %v", err)
	}

	outProjWeight, err := tensor.New([]float32{1, 0, 0, 1}, []int64{2, 2})
	if err != nil {
		t.Fatalf("outProj weight: %v", err)
	}

	layer := &flowTransformerLayer{
		inProj:  &Linear{Weight: inProjWeight, inDim: 2, outDim: 6},
		outProj: &Linear{Weight: outProjWeight, inDim: 2, outDim: 2},
		nHeads:  1,
		headDim: 2,
	}

	x, err := tensor.New([]float32{
		1, 0,
		0, 1,
		1, 1,
	}, []int64{1, 3, 2})
	if err != nil {
		t.Fatalf("input: %v", err)
	}

	ropeCos, err := tensor.New([]float32{1, 0, -1}, []int64{3, 1})
	if err != nil {
		t.Fatalf("rope cos: %v", err)
	}

	ropeSin, err := tensor.New([]float32{0, 1, 0}, []int64{3, 1})
	if err != nil {
		t.Fatalf("rope sin: %v", err)
	}

	full, err := layer.selfAttention(x, ropeCos, ropeSin)
	if err != nil {
		t.Fatalf("full selfAttention: %v", err)
	}

	prefix, err := tensor.New(x.RawData()[:4], []int64{1, 2, 2})
	if err != nil {
		t.Fatalf("prefix: %v", err)
	}

	state := &flowTransformerLayerState{}
	_, k, v, err := layer.projectQKV(prefix, ropeCos, ropeSin, 0)
	if err != nil {
		t.Fatalf("project prefix: %v", err)
	}

	if err := state.appendKV(k, v); err != nil {
		t.Fatalf("append prefix: %v", err)
	}

	step, err := tensor.New(x.RawData()[4:], []int64{1, 1, 2})
	if err != nil {
		t.Fatalf("step: %v", err)
	}

	q, k, v, err := layer.projectQKV(step, ropeCos, ropeSin, state.offset)
	if err != nil {
		t.Fatalf("project step: %v", err)
	}

	stepStart := state.offset
	if err := state.appendKV(k, v); err != nil {
		t.Fatalf("append step: %v", err)
	}

	got, err := layer.attentionFromPositions(
		q,
		state.kCache,
		state.vCache,
		positionsRange(stepStart, 1),
		cachePositions(state.kCache, state.offset),
		ops.AttentionNoContext,
	)
	if err != nil {
		t.Fatalf("stateful attention: %v", err)
	}

	want := full.RawData()[4:6]
	if !equalApproxN(got.RawData(), want, 1e-5) {
		t.Fatalf("stateful last attention = %v, want full last token %v", got.RawData(), want)
	}
}

func mustTensorN(t *testing.T, data []float32, shape []int64) *tensor.Tensor {
	t.Helper()

	tt, err := tensor.New(data, shape)
	if err != nil {
		t.Fatalf("tensor.New(%v, %v): %v", data, shape, err)
	}

	return tt
}

func equalApproxN(got, want []float32, tol float64) bool {
	if len(got) != len(want) {
		return false
	}

	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			return false
		}
	}

	return true
}
