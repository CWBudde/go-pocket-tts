package native

import (
	"math"
	"math/rand"
	"slices"
	"strconv"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

// Synthetic FlowLM sizes: model width, heads, layers, feed-forward width and
// text vocabulary. The latent width is synthLatent, the flow_net the one of
// syntheticFlowNet.
const (
	synthFlowDModel = 8
	synthFlowHeads  = 2
	synthFlowLayers = 2
	synthFlowFFN    = 16
	synthFlowVocab  = 6
)

// syntheticFlowLM loads a small LSD FlowLM with deterministic, non-trivial
// weights through LoadFlowLM.
func syntheticFlowLM(t *testing.T, insertBOS bool) *FlowLM {
	t.Helper()

	var tensors []safetensors.Tensor

	seed := 100
	add := func(name string, shape ...int64) {
		tensors = append(tensors, safetensors.Tensor{Name: name, Shape: shape, Data: syntheticWeights(seed, shape)})
		seed++
	}

	const d = synthFlowDModel

	add("flow_lm.conditioner.embed.weight", synthFlowVocab, d)

	for i := range synthFlowLayers {
		p := "flow_lm.transformer.layers." + strconv.Itoa(i) + "."
		add(p+"norm1.weight", d)
		add(p+"norm1.bias", d)
		add(p+"norm2.weight", d)
		add(p+"norm2.bias", d)
		add(p+"self_attn.in_proj.weight", 3*d, d)
		add(p+"self_attn.out_proj.weight", d, d)
		add(p+"linear1.weight", synthFlowFFN, d)
		add(p+"linear2.weight", d, synthFlowFFN)
	}

	addSyntheticFlowNet(add, "flow_lm.flow_net.", 2, synthLatent, d)

	add("flow_lm.emb_std", synthLatent)
	add("flow_lm.emb_mean", synthLatent)
	add("flow_lm.bos_emb", synthLatent)
	add("flow_lm.bos_before_voice", 1, 1, d)
	add("flow_lm.input_linear.weight", d, synthLatent)
	add("flow_lm.input_linear.bias", d)
	add("flow_lm.out_norm.weight", d)
	add("flow_lm.out_norm.bias", d)
	add("flow_lm.out_eos.weight", 1, d)
	add("flow_lm.out_eos.bias", 1)

	blob, err := safetensors.EncodeTensors(tensors)
	if err != nil {
		t.Fatal(err)
	}

	flow, err := LoadFlowLM(NewVarBuilder(mustStore(t, blob)), FlowLMConfig{
		DModel:               d,
		NumHeads:             synthFlowHeads,
		MaxPeriod:            10000,
		LDim:                 synthLatent,
		InsertBOSBeforeVoice: insertBOS,
		FlowType:             modelcfg.FlowTypeLSD,
	})
	if err != nil {
		t.Fatalf("LoadFlowLM: %v", err)
	}

	return flow
}

// syntheticConditioning returns a [1, frames, synthFlowDModel] audio
// conditioning.
func syntheticConditioning(t *testing.T, frames int64) *tensor.Tensor {
	t.Helper()

	return mustTensorN(t, syntheticWeights(7, []int64{1, frames, synthFlowDModel}), []int64{1, frames, synthFlowDModel})
}

// exportAndReload writes state like export-voice --format model-state and
// loads it back like a model-state voice file.
func exportAndReload(t *testing.T, f *FlowLM, state *FlowLMState) (*FlowLMState, []byte) {
	t.Helper()

	vs, err := state.VoiceModelState()
	if err != nil {
		t.Fatalf("VoiceModelState: %v", err)
	}

	blob, err := safetensors.EncodeVoiceModelState(vs)
	if err != nil {
		t.Fatalf("EncodeVoiceModelState: %v", err)
	}

	loaded, err := safetensors.LoadVoiceModelStateFromBytes(blob)
	if err != nil {
		t.Fatalf("LoadVoiceModelStateFromBytes: %v", err)
	}

	reloaded, err := f.InitStateFromVoiceModelState(loaded)
	if err != nil {
		t.Fatalf("InitStateFromVoiceModelState: %v", err)
	}

	return reloaded, blob
}

// kvRows returns the first steps positions of a [B, H, capacity, Dh] cache.
func kvRows(t *testing.T, cache *tensor.Tensor, steps int64) *tensor.Tensor {
	t.Helper()

	shape := cache.Shape()
	b, h, capacity, dh := shape[0], shape[1], shape[2], shape[3]
	raw := cache.RawData()
	out := make([]float32, 0, b*h*steps*dh)

	for bh := range b * h {
		out = append(out, raw[bh*capacity*dh:(bh*capacity+steps)*dh]...)
	}

	return mustTensorN(t, out, []int64{b, h, steps, dh})
}

// TestFlowLMState_VoiceModelStateRoundTrip exports a prompted state whose
// caches have spare capacity and checks that the reload restores the same
// offsets and K/V caches, and that the file has upstream's layout.
func TestFlowLMState_VoiceModelStateRoundTrip(t *testing.T) {
	f := syntheticFlowLM(t, true)

	const frames = 3

	state, err := f.PromptVoice(syntheticConditioning(t, frames))
	if err != nil {
		t.Fatalf("PromptVoice: %v", err)
	}

	// Grown caches hold unused positions past the offset, as after AR steps.
	for i := range state.transformer.layers {
		err = state.transformer.layers[i].ensureKVCapacity(5)
		if err != nil {
			t.Fatal(err)
		}
	}

	reloaded, blob := exportAndReload(t, f, state)

	const steps = frames + 1 // BOS before voice

	for i, want := range state.transformer.layers {
		got := reloaded.transformer.layers[i]
		if want.offset != steps || got.offset != steps {
			t.Fatalf("layer %d offset: exported %d, reloaded %d; want %d", i, want.offset, got.offset, steps)
		}

		for _, kv := range []struct {
			name      string
			got, want *tensor.Tensor
		}{
			{"key", got.kCache, kvRows(t, want.kCache, steps)},
			{"value", got.vCache, kvRows(t, want.vCache, steps)},
		} {
			if !slices.Equal(kv.got.Shape(), kv.want.Shape()) || !slices.Equal(kv.got.RawData(), kv.want.RawData()) {
				t.Fatalf("layer %d %s cache %v differs from the exported %v", i, kv.name, kv.got.Shape(), kv.want.Shape())
			}
		}
	}

	store, err := safetensors.OpenStoreFromBytes(blob, safetensors.StoreOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	wantNames := make([]string, 0, 3*synthFlowLayers)

	for i := range synthFlowLayers {
		m := flowAttentionModuleName(i)
		wantNames = append(wantNames, m+"/cache", m+"/offset", m+"/pad")
	}

	if got := store.Names(); !slices.Equal(got, wantNames) {
		t.Fatalf("tensor names = %v, want %v", got, wantNames)
	}

	m := flowAttentionModuleName(0)
	for _, tc := range []struct {
		key   string
		dtype string
		shape []int64
	}{
		{"cache", "F32", []int64{2, 1, steps, synthFlowHeads, synthFlowDModel / synthFlowHeads}},
		{"offset", "I64", []int64{1}},
		{"pad", "I64", []int64{1}},
	} {
		got, err := store.Tensor(m + "/" + tc.key)
		if err != nil {
			t.Fatal(err)
		}

		if got.DType != tc.dtype || !slices.Equal(got.Shape, tc.shape) {
			t.Errorf("%s = %s %v, want %s %v (upstream export_model_state)", tc.key, got.DType, got.Shape, tc.dtype, tc.shape)
		}

		if tc.key == "pad" && got.Data[0] != 0 {
			t.Errorf("pad = %v, want 0", got.Data[0])
		}
	}
}

// TestFlowLMPromptVoice_BOSBeforeVoice checks that PromptVoice prompts
// flow_lm.bos_before_voice before the conditioning when the config asks for
// it, like upstream get_state_for_audio_prompt.
func TestFlowLMPromptVoice_BOSBeforeVoice(t *testing.T) {
	for _, insertBOS := range []bool{true, false} {
		f := syntheticFlowLM(t, insertBOS)
		cond := syntheticConditioning(t, 3)

		got, err := f.PromptVoice(cond)
		if err != nil {
			t.Fatalf("insertBOS=%v: PromptVoice: %v", insertBOS, err)
		}

		seq := cond
		if insertBOS {
			seq, err = tensor.Concat([]*tensor.Tensor{f.bosBeforeVoice, cond}, 1)
			if err != nil {
				t.Fatal(err)
			}
		}

		want, err := f.InitState()
		if err != nil {
			t.Fatal(err)
		}

		err = f.PromptText(want, seq)
		if err != nil {
			t.Fatal(err)
		}

		if got.Offset() != seq.Shape()[1] {
			t.Fatalf("insertBOS=%v: offset = %d, want %d", insertBOS, got.Offset(), seq.Shape()[1])
		}

		for i := range want.transformer.layers {
			if !slices.Equal(got.transformer.layers[i].kCache.RawData(), want.transformer.layers[i].kCache.RawData()) {
				t.Fatalf("insertBOS=%v: layer %d key cache differs from prompting %v", insertBOS, i, seq.Shape())
			}
		}
	}
}

// TestVoiceModelStateExport_GenerationMatchesAudioPrompt generates a few
// frames at temperature 0 from an exported and reloaded voice state and from
// the audio_prompt path of the runtime (voice and text prompted in one pass),
// for the same conditioning.
//
// The export itself is lossless: the reloaded state generates bit-identical
// frames to the state it was exported from. Against the audio_prompt path the
// frames agree within float32 rounding only (generationTol): prompting voice
// and text in two prefills instead of one changes which rows of the
// projections fall into the 4×4 blocks of the arm64 MatMulTransB kernel and
// which into its dot-product tail, so some sums accumulate in another order
// (2.4e-7 here with 5 voice positions; bit-identical with 4).
func TestVoiceModelStateExport_GenerationMatchesAudioPrompt(t *testing.T) {
	const generationTol = 1e-5

	for _, insertBOS := range []bool{true, false} {
		t.Run("bos="+strconv.FormatBool(insertBOS), func(t *testing.T) {
			f := syntheticFlowLM(t, insertBOS)
			cond := syntheticConditioning(t, 4)

			text, err := f.TextEmbeddings([]int64{1, 4, 2})
			if err != nil {
				t.Fatal(err)
			}

			fromAudioPrompt := audioPromptFlowState(t, f, cond, text)

			voiceState, err := f.PromptVoice(cond)
			if err != nil {
				t.Fatalf("PromptVoice: %v", err)
			}

			fromModelState, _ := exportAndReload(t, f, voiceState)

			for _, state := range []*FlowLMState{voiceState, fromModelState} {
				err = f.PromptText(state, text)
				if err != nil {
					t.Fatal(err)
				}
			}

			want := generateFrames(t, f, fromAudioPrompt, 4)
			exported := generateFrames(t, f, voiceState, 4)
			got := generateFrames(t, f, fromModelState, 4)

			assertFramesMatch(t, got, exported, 0)
			assertFramesMatch(t, got, want, generationTol)
		})
	}
}

// audioPromptFlowState prompts voice and text in one pass, like
// nativeSafetensorsRuntime.prepareFlowState for an audio_prompt voice.
func audioPromptFlowState(t *testing.T, f *FlowLM, cond, text *tensor.Tensor) *FlowLMState {
	t.Helper()

	voice, err := f.VoicePrompt(cond)
	if err != nil {
		t.Fatal(err)
	}

	seq, err := tensor.Concat([]*tensor.Tensor{voice, text}, 1)
	if err != nil {
		t.Fatal(err)
	}

	state, err := f.InitState()
	if err != nil {
		t.Fatal(err)
	}

	err = f.PromptText(state, seq)
	if err != nil {
		t.Fatal(err)
	}

	return state
}

// generatedFrame is one AR step: the latent and the EOS decision.
type generatedFrame struct {
	latent []float32
	eos    bool
}

// generateFrames runs n AR steps at temperature 0 from the BOS frame, feeding
// each latent back like the runtime's AR loop.
func generateFrames(t *testing.T, f *FlowLM, state *FlowLMState, n int) []generatedFrame {
	t.Helper()

	nan := float32(math.NaN())
	frame := mustTensorN(t, slices.Repeat([]float32{nan}, int(f.cfg.LDim)), []int64{1, 1, f.cfg.LDim})
	rng := rand.New(rand.NewSource(1))
	out := make([]generatedFrame, 0, n)

	for range n {
		next, eos, err := f.SampleNextLatentStateful(state, frame, 2, -4, 0, rng)
		if err != nil {
			t.Fatalf("SampleNextLatentStateful: %v", err)
		}

		out = append(out, generatedFrame{latent: append([]float32(nil), next.RawData()...), eos: eos})
		frame = next
	}

	return out
}

// assertFramesMatch compares generated frames element-wise within tol
// (absolute).
func assertFramesMatch(t *testing.T, got, want []generatedFrame, tol float64) {
	t.Helper()

	maxDiff := 0.0

	for i := range want {
		if got[i].eos != want[i].eos {
			t.Fatalf("frame %d EOS = %v, want %v", i, got[i].eos, want[i].eos)
		}

		for j, w := range want[i].latent {
			diff := math.Abs(float64(got[i].latent[j] - w))
			if math.IsNaN(diff) || diff > tol {
				t.Fatalf("frame %d latent[%d] = %g, want %g (tolerance %g)", i, j, got[i].latent[j], w, tol)
			}

			maxDiff = max(maxDiff, diff)
		}
	}

	t.Logf("%d frames match, max abs diff %g", len(want), maxDiff)
}

// TestVoiceModelStateExport_GenerationMatchesAudioPrompt_GatedGerman repeats
// the generation check on the gated german checkpoint with the parity prompt
// (skipped without it). Its 28 voice positions (BOS + 27 frames) fill whole
// 4-row kernel blocks, so here both prompt splits round alike and the frames
// come out bit-identical; the tolerance covers prompts of other lengths.
func TestVoiceModelStateExport_GenerationMatchesAudioPrompt_GatedGerman(t *testing.T) {
	var fx *voiceEncoderFixture

	for _, f := range loadVoiceEncoderFixtures(t) {
		if f.Source.Config == "german" {
			fx = f
		}
	}

	if fx == nil {
		t.Fatal("no german encoder fixture")
	}

	weights := gatedCheckpoint(t, "german")

	mc, err := modelcfg.Lookup("german")
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
		t.Fatal(err)
	}

	f := m.FlowLM()

	text, err := f.TextEmbeddings([]int64{12, 345, 67, 890, 23})
	if err != nil {
		t.Fatal(err)
	}

	fromAudioPrompt := audioPromptFlowState(t, f, cond, text)

	voiceState, err := f.PromptVoice(cond)
	if err != nil {
		t.Fatal(err)
	}

	fromModelState, _ := exportAndReload(t, f, voiceState)

	err = f.PromptText(fromModelState, text)
	if err != nil {
		t.Fatal(err)
	}

	assertFramesMatch(t, generateFrames(t, f, fromModelState, 3), generateFrames(t, f, fromAudioPrompt, 3), 1e-5)
}
