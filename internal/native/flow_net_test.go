package native

import (
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
)

// Synthetic flow_net sizes: latent, model channels, condition, frequency
// embedding (freqs holds half of it).
const (
	synthLatent = 2
	synthChans  = 4
	synthCond   = 3
	synthFreq   = 4
)

// syntheticFlowNet builds a one-res-block flow_net with numTimeConds
// time_embed entries and deterministic, non-trivial weights.
func syntheticFlowNet(t *testing.T, numTimeConds int) *flowNet {
	t.Helper()

	specs := map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{}
	seed := 0

	add := func(name string, shape ...int64) {
		n := int64(1)
		for _, d := range shape {
			n *= d
		}

		vals := make([]float32, n)
		for i := range vals {
			vals[i] = float32(0.5 * math.Sin(float64(seed)*1.3+float64(i)*0.7))
		}

		seed++

		specs["flow_net."+name] = struct {
			dtype string
			shape []int64
			data  []byte
		}{dtype: "F32", shape: shape, data: f32Bytes(vals)}
	}

	for i := range numTimeConds {
		p := "time_embed." + strconv.Itoa(i) + "."
		add(p+"freqs", synthFreq/2)
		add(p+"mlp.0.weight", synthChans, synthFreq)
		add(p+"mlp.0.bias", synthChans)
		add(p+"mlp.2.weight", synthChans, synthChans)
		add(p+"mlp.2.bias", synthChans)
		add(p+"mlp.3.alpha", synthChans)
	}

	add("cond_embed.weight", synthChans, synthCond)
	add("cond_embed.bias", synthChans)
	add("input_proj.weight", synthChans, synthLatent)
	add("input_proj.bias", synthChans)
	add("res_blocks.0.in_ln.weight", synthChans)
	add("res_blocks.0.in_ln.bias", synthChans)
	add("res_blocks.0.mlp.0.weight", synthChans, synthChans)
	add("res_blocks.0.mlp.0.bias", synthChans)
	add("res_blocks.0.mlp.2.weight", synthChans, synthChans)
	add("res_blocks.0.mlp.2.bias", synthChans)
	add("res_blocks.0.adaLN_modulation.1.weight", 3*synthChans, synthChans)
	add("res_blocks.0.adaLN_modulation.1.bias", 3*synthChans)
	add("final_layer.linear.weight", synthLatent, synthChans)
	add("final_layer.linear.bias", synthLatent)
	add("final_layer.adaLN_modulation.1.weight", 2*synthChans, synthChans)
	add("final_layer.adaLN_modulation.1.bias", 2*synthChans)

	fn, err := loadFlowNet(NewVarBuilder(mustStore(t, buildSafetensors(t, specs))).Path("flow_net"))
	if err != nil {
		t.Fatalf("loadFlowNet(%d time conditions): %v", numTimeConds, err)
	}

	return fn
}

func TestLoadFlowNet_DetectsTimeConds(t *testing.T) {
	for _, n := range []int{0, 1, 2} {
		if got := len(syntheticFlowNet(t, n).timeEmbeds); got != n {
			t.Errorf("loadFlowNet with %d time_embed entries detected %d", n, got)
		}
	}
}

func TestFlowNetConditioning_AveragesTimeEmbeddings(t *testing.T) {
	c := mustTensorN(t, []float32{0.3, -0.2, 0.9}, []int64{1, synthCond})
	s := mustTensorN(t, []float32{0.25}, []int64{1, 1})
	tt := mustTensorN(t, []float32{0.75}, []int64{1, 1})

	for _, tc := range []struct {
		n  int
		ts []*tensor.Tensor
	}{
		{0, nil},
		{1, []*tensor.Tensor{s}},
		{2, []*tensor.Tensor{s, tt}},
	} {
		fn := syntheticFlowNet(t, tc.n)

		got, err := fn.conditioning(c, tc.ts)
		if err != nil {
			t.Fatalf("%d time conditions: %v", tc.n, err)
		}

		want, err := fn.condEmbed.Forward(c)
		if err != nil {
			t.Fatal(err)
		}

		wd := want.RawData()

		for i, te := range fn.timeEmbeds {
			emb, err := te.Forward(tc.ts[i])
			if err != nil {
				t.Fatal(err)
			}

			for j, v := range emb.RawData() {
				wd[j] += v / float32(tc.n)
			}
		}

		if !equalApproxN(got.RawData(), wd, 1e-6) {
			t.Errorf("%d time conditions: conditioning = %v, want cond_embed(c) + mean(time embeddings) = %v",
				tc.n, got.RawData(), wd)
		}
	}
}

func TestFlowNetForward_RejectsWrongTimeConds(t *testing.T) {
	c := mustTensorN(t, []float32{0.3, -0.2, 0.9}, []int64{1, synthCond})
	x := mustTensorN(t, []float32{1, -1}, []int64{1, synthLatent})
	s := mustTensorN(t, []float32{0}, []int64{1, 1})

	_, err := syntheticFlowNet(t, 0).Forward(c, []*tensor.Tensor{s}, x)
	if err == nil || !strings.Contains(err.Error(), "time conditions") {
		t.Fatalf("drifting head with a time input: err = %v; want a time-condition count error", err)
	}

	_, err = syntheticFlowNet(t, 2).Forward(c, []*tensor.Tensor{s}, x)
	if err == nil || !strings.Contains(err.Error(), "time conditions") {
		t.Fatalf("lsd head with one time input: err = %v; want a time-condition count error", err)
	}
}

func TestCheckFlowTimeConds(t *testing.T) {
	for _, tc := range []struct {
		flowType string
		got      int
		ok       bool
	}{
		{"", 2, true},
		{modelcfg.FlowTypeLSD, 2, true},
		{modelcfg.FlowTypeFlowMatching, 1, true},
		{modelcfg.FlowTypeDrifting, 0, true},
		{modelcfg.FlowTypeLSD, 0, false},
		{modelcfg.FlowTypeDrifting, 2, false},
		{"diffusion", 2, false},
	} {
		err := checkFlowTimeConds(tc.flowType, tc.got)
		if (err == nil) != tc.ok {
			t.Errorf("checkFlowTimeConds(%q, %d) = %v; want ok=%v", tc.flowType, tc.got, err, tc.ok)
		}
	}
}

func TestFlowLMDecode_DispatchesOnFlowType(t *testing.T) {
	cond := mustTensorN(t, []float32{0.3, -0.2, 0.9}, []int64{1, synthCond})
	x0 := mustTensorN(t, []float32{0.4, -1.1}, []int64{1, synthLatent})

	// drifting: one forward pass without time input; steps are ignored.
	drift := &FlowLM{flowNet: syntheticFlowNet(t, 0), cfg: FlowLMConfig{FlowType: modelcfg.FlowTypeDrifting}}

	got, err := drift.decode(cond, x0, 3)
	if err != nil {
		t.Fatalf("drifting decode: %v", err)
	}

	want, err := drift.flowNet.Forward(cond, nil, x0)
	if err != nil {
		t.Fatal(err)
	}

	if !equalApproxN(got.RawData(), want.RawData(), 0) {
		t.Errorf("drifting decode = %v, want v(x0) = %v", got.RawData(), want.RawData())
	}

	// flow_matching: Euler steps cur += v(i/n, cur)/n.
	ot := &FlowLM{flowNet: syntheticFlowNet(t, 1), cfg: FlowLMConfig{FlowType: modelcfg.FlowTypeFlowMatching}}

	got, err = ot.decode(cond, x0, 2)
	if err != nil {
		t.Fatalf("flow_matching decode: %v", err)
	}

	cur := x0.Clone()
	for i := range 2 {
		v, err := ot.flowNet.Forward(cond, []*tensor.Tensor{mustTensorN(t, []float32{float32(i) / 2}, []int64{1, 1})}, cur)
		if err != nil {
			t.Fatal(err)
		}

		for j, d := range v.RawData() {
			cur.RawData()[j] += d / 2
		}
	}

	if !equalApproxN(got.RawData(), cur.RawData(), 1e-6) {
		t.Errorf("flow_matching decode = %v, want %v", got.RawData(), cur.RawData())
	}

	if !equalApproxN(x0.RawData(), []float32{0.4, -1.1}, 0) {
		t.Errorf("decode modified x0: %v", x0.RawData())
	}

	// lsd (also the zero value): same as LSDDecode.
	lsd := &FlowLM{flowNet: syntheticFlowNet(t, 2)}

	got, err = lsd.decode(cond, x0, 2)
	if err != nil {
		t.Fatalf("lsd decode: %v", err)
	}

	want, err = lsd.LSDDecode(cond, x0, 2)
	if err != nil {
		t.Fatal(err)
	}

	if !equalApproxN(got.RawData(), want.RawData(), 0) {
		t.Errorf("lsd decode = %v, want LSDDecode = %v", got.RawData(), want.RawData())
	}

	unknown := &FlowLM{flowNet: syntheticFlowNet(t, 2), cfg: FlowLMConfig{FlowType: "diffusion"}}

	_, err = unknown.decode(cond, x0, 1)
	if err == nil || !strings.Contains(err.Error(), "diffusion") {
		t.Errorf("unknown flow type: err = %v; want one naming it", err)
	}
}

func TestFlowLMDecode_Guards(t *testing.T) {
	cond := mustTensorN(t, []float32{0.3, -0.2, 0.9}, []int64{1, synthCond})
	bad := mustTensorN(t, []float32{0.4, -1.1}, []int64{1, 1, synthLatent})

	for _, flowType := range []string{modelcfg.FlowTypeFlowMatching, modelcfg.FlowTypeDrifting} {
		f := &FlowLM{cfg: FlowLMConfig{FlowType: flowType}}

		_, err := f.decode(cond, bad, 1)
		if err == nil || !strings.Contains(err.Error(), "must be [B, D]") {
			t.Errorf("%s decode with rank-3 x0: err = %v", flowType, err)
		}
	}

	ot := &FlowLM{cfg: FlowLMConfig{FlowType: modelcfg.FlowTypeFlowMatching}}

	_, err := ot.decode(cond, mustTensorN(t, []float32{0.4, -1.1}, []int64{1, synthLatent}), 0)
	if err == nil || !strings.Contains(err.Error(), "steps must be >0") {
		t.Errorf("flow_matching decode with 0 steps: err = %v", err)
	}
}

func TestConfigFor_FlowType(t *testing.T) {
	for language, want := range map[string]string{
		"english_2026-01":        modelcfg.FlowTypeLSD,
		"german":                 modelcfg.FlowTypeLSD,
		"english_drifting_26-09": modelcfg.FlowTypeDrifting,
	} {
		mc, err := modelcfg.Lookup(language)
		if err != nil {
			t.Fatal(err)
		}

		if got := ConfigFor(mc).FlowLM.FlowType; got != want {
			t.Errorf("ConfigFor(%s).FlowLM.FlowType = %q, want %q", language, got, want)
		}
	}

	if got := DefaultConfig().FlowLM.FlowType; got != modelcfg.FlowTypeLSD {
		t.Errorf("DefaultConfig().FlowLM.FlowType = %q, want %q", got, modelcfg.FlowTypeLSD)
	}
}
