package native

import (
	"math"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
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

	_, err = nilTransformer.prefill(x, state)
	if err == nil || !strings.Contains(err.Error(), "flow transformer is nil") {
		t.Fatalf("expected nil transformer error, got: %v", err)
	}

	_, err = tfm.prefill(x, nil)
	if err == nil || !strings.Contains(err.Error(), "state is nil") {
		t.Fatalf("expected nil state error, got: %v", err)
	}

	_, err = tfm.prefill(x, &flowTransformerState{})
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

	if s.seqLen != 1 {
		t.Fatalf("seqLen after first append = %d, want 1", s.seqLen)
	}

	k2 := mustTensorN(t, []float32{3, 4}, []int64{1, 1, 2, 1})
	v2 := mustTensorN(t, []float32{5, 6}, []int64{1, 1, 2, 1})
	err = s.appendKV(k2, v2)
	if err != nil {
		t.Fatalf("appendKV second call error: %v", err)
	}

	if s.seqLen != 3 {
		t.Fatalf("seqLen after second append = %d, want 3", s.seqLen)
	}

	if got := s.kCache.Shape()[2]; got != 3 {
		t.Fatalf("kCache sequence dim = %d, want 3", got)
	}

	if got := detectNumHeads(nil, 8); got != 8 {
		t.Fatalf("detectNumHeads(nil) = %d, want 8", got)
	}

	if got := detectNumHeads(NewVarBuilder(nil), 6); got != 6 {
		t.Fatalf("detectNumHeads(uninitialized vb) = %d, want 6", got)
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
