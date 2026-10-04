package tts

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/genloop"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	nativemodel "github.com/cwbudde/go-pocket-tts/internal/native"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

func TestNewNativeSafetensorsRuntime(t *testing.T) {
	rt := NewNativeSafetensorsRuntime(&nativemodel.Model{})
	if rt == nil {
		t.Fatal("NewNativeSafetensorsRuntime returned nil")
	}

	impl, ok := rt.(*nativeSafetensorsRuntime)
	if !ok {
		t.Fatalf("runtime type = %T, want *nativeSafetensorsRuntime", rt)
	}

	if impl.rng == nil {
		t.Fatal("expected rng to be initialized")
	}
}

func TestNativeSafetensorsRuntimeGenerateAudio_Guards(t *testing.T) {
	t.Run("nil receiver", func(t *testing.T) {
		var rt *nativeSafetensorsRuntime

		_, err := rt.GenerateAudio(context.Background(), []int64{1}, RuntimeGenerateConfig{})
		if err == nil || !strings.Contains(err.Error(), "runtime unavailable") {
			t.Fatalf("expected runtime unavailable error, got: %v", err)
		}
	})

	t.Run("nil model", func(t *testing.T) {
		rt := &nativeSafetensorsRuntime{}

		_, err := rt.GenerateAudio(context.Background(), []int64{1}, RuntimeGenerateConfig{})
		if err == nil || !strings.Contains(err.Error(), "runtime unavailable") {
			t.Fatalf("expected runtime unavailable error, got: %v", err)
		}
	})

	t.Run("empty tokens", func(t *testing.T) {
		rt := &nativeSafetensorsRuntime{model: &nativemodel.Model{}}

		_, err := rt.GenerateAudio(context.Background(), nil, RuntimeGenerateConfig{})
		if err == nil || !strings.Contains(err.Error(), "must not be empty") {
			t.Fatalf("expected empty token error, got: %v", err)
		}
	})

	t.Run("model text embedding error is wrapped", func(t *testing.T) {
		rt := &nativeSafetensorsRuntime{model: &nativemodel.Model{}}

		_, err := rt.GenerateAudio(context.Background(), []int64{1, 2, 3}, RuntimeGenerateConfig{})
		if err == nil {
			t.Fatal("expected non-nil error")
		}

		if !strings.Contains(err.Error(), "generate: text embeddings") {
			t.Fatalf("expected wrapped text embeddings error, got: %v", err)
		}
	})
}

func TestNativeSafetensorsRuntimeClose_NoPanic(_ *testing.T) {
	var nilRuntime *nativeSafetensorsRuntime
	nilRuntime.Close()

	rt := &nativeSafetensorsRuntime{}
	rt.Close()

	rt = &nativeSafetensorsRuntime{model: &nativemodel.Model{}}
	rt.Close()
}

func TestNewBOSSequenceTensor(t *testing.T) {
	bos, err := newBOSSequenceTensor()
	if err != nil {
		t.Fatalf("newBOSSequenceTensor returned error: %v", err)
	}

	shape := bos.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != nativeLatentDim {
		t.Fatalf("unexpected bos shape: %v", shape)
	}

	for i, v := range bos.RawData() {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("bos[%d] = %v, want NaN", i, v)
		}
	}
}

func TestStackLatentFramesTensor(t *testing.T) {
	_, err := stackLatentFramesTensor(nil)
	if err == nil || !strings.Contains(err.Error(), "no latent frames") {
		t.Fatalf("expected empty-frames error, got: %v", err)
	}

	f1 := mustTensor(t, seq(1, nativeLatentDim), []int64{1, 1, nativeLatentDim})
	f2 := mustTensor(t, seq(100, nativeLatentDim), []int64{1, 1, nativeLatentDim})

	stacked, err := stackLatentFramesTensor([]*tensor.Tensor{f1, f2})
	if err != nil {
		t.Fatalf("stackLatentFramesTensor returned error: %v", err)
	}

	shape := stacked.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != nativeLatentDim {
		t.Fatalf("unexpected stacked shape: %v", shape)
	}

	data := stacked.RawData()
	if len(data) != int(2*nativeLatentDim) {
		t.Fatalf("unexpected stacked data len: %d", len(data))
	}

	if data[0] != 1 || data[nativeLatentDim] != 100 {
		t.Fatalf("unexpected concatenation order: first=%v secondStart=%v", data[0], data[nativeLatentDim])
	}
}

func mustTensor(t *testing.T, data []float32, shape []int64) *tensor.Tensor {
	t.Helper()

	tt, err := tensor.New(data, shape)
	if err != nil {
		t.Fatalf("tensor.New(%v, %v): %v", data, shape, err)
	}

	return tt
}

func seq(start float32, n int64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = start + float32(i)
	}

	return out
}

func TestPrepareFlowState_BOSBeforeVoice_RealGermanCheckpoint(t *testing.T) {
	path := filepath.Join("..", "..", "models", "german", "model.safetensors")

	_, err := os.Stat(path)
	if err != nil {
		t.Skipf("german checkpoint not available: %v", err)
	}

	mc, err := modelcfg.Lookup("german")
	if err != nil {
		t.Fatal(err)
	}

	voice := &VoiceEmbedding{Data: make([]float32, 2*1024), Shape: []int64{1, 2, 1024}}

	for _, tc := range []struct {
		name string
		cfg  nativemodel.Config
		want int64
	}{
		{"insert_bos_before_voice", nativemodel.ConfigFor(mc), 1 + 2 + 3},
		{"without", nativemodel.DefaultConfig(), 2 + 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, err := nativemodel.LoadModelFromSafetensors(path, tc.cfg)
			if err != nil {
				t.Fatalf("load model: %v", err)
			}
			defer m.Close()

			rt := &nativeSafetensorsRuntime{model: m}

			text, err := m.TextEmbeddings([]int64{1, 2, 3})
			if err != nil {
				t.Fatal(err)
			}

			state, err := rt.prepareFlowState(text, RuntimeGenerateConfig{VoiceEmbedding: voice})
			if err != nil {
				t.Fatalf("prepareFlowState: %v", err)
			}

			if got := state.Offset(); got != tc.want {
				t.Fatalf("prompted positions = %d, want %d", got, tc.want)
			}
		})
	}
}

// fakeSampler returns a latentSampler that flags EOS from step eosFrom on and
// calls onStep (if set) with the 1-based call count, plus that count.
func fakeSampler(t *testing.T, eosFrom int, onStep func(calls int)) (latentSampler, *int) {
	t.Helper()

	calls := 0

	return func(_ *nativemodel.FlowLMState, _ *tensor.Tensor, _ int, _, _ float32, _ *rand.Rand) (*tensor.Tensor, bool, error) {
		step := calls
		calls++

		if onStep != nil {
			onStep(calls)
		}

		return mustTensor(t, make([]float32, nativeLatentDim), []int64{1, 1, nativeLatentDim}), step >= eosFrom, nil
	}, &calls
}

func TestRunARLoop_EOSStopRule(t *testing.T) {
	// EOS is flagged from step 2 on but only accepted from step
	// genloop.MinFramesBeforeEOS = 6; upstream keeps eos_step + F frames.
	for _, tc := range []struct {
		framesAfter int
		wantFrames  int
	}{
		{3, 9},
		{0, 6},
	} {
		sample, calls := fakeSampler(t, 2, nil)
		rt := &nativeSafetensorsRuntime{sample: sample}

		frames, err := rt.runARLoop(context.Background(), nil, nil, 256, 1, RuntimeGenerateConfig{FramesAfterEOS: tc.framesAfter})
		if err != nil {
			t.Fatalf("runARLoop: %v", err)
		}

		if len(frames) != tc.wantFrames || *calls != tc.wantFrames+1 {
			t.Errorf("frames_after_eos=%d: %d frames from %d steps, want %d from %d",
				tc.framesAfter, len(frames), *calls, tc.wantFrames, tc.wantFrames+1)
		}
	}
}

func TestRunARLoop_CancelStopsWithinOneStep(t *testing.T) {
	const cancelAt = 4

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sample, calls := fakeSampler(t, 1000, func(calls int) {
		if calls == cancelAt {
			cancel()
		}
	})
	rt := &nativeSafetensorsRuntime{sample: sample}

	_, err := rt.runARLoop(ctx, nil, nil, 256, 1, RuntimeGenerateConfig{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("runARLoop err = %v, want context.Canceled", err)
	}

	// At most the step that was already running when ctx was cancelled.
	if *calls > cancelAt+1 {
		t.Errorf("sampler calls = %d after cancelling at %d", *calls, cancelAt)
	}
}

func TestGenerateAudio_FadesInChunkStart_RealCheckpoint(t *testing.T) {
	modelPath, _ := requireNativeSafetensorsAssetsForUnit(t)

	m, err := nativemodel.LoadModelFromSafetensors(modelPath, nativemodel.DefaultConfig())
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	defer m.Close()

	// Seven zero latents with EOS on the last: 6 frames are kept.
	sample, _ := fakeSampler(t, genloop.MinFramesBeforeEOS, nil)
	rt := &nativeSafetensorsRuntime{model: m, sample: sample}

	pcm, err := rt.GenerateAudio(context.Background(), []int64{1, 2, 3}, RuntimeGenerateConfig{MaxSteps: 20})
	if err != nil {
		t.Fatalf("GenerateAudio: %v", err)
	}

	frames := make([]*tensor.Tensor, genloop.MinFramesBeforeEOS)
	for i := range frames {
		frames[i] = mustTensor(t, make([]float32, nativeLatentDim), []int64{1, 1, nativeLatentDim})
	}

	ref, err := rt.decodeLatents(frames)
	if err != nil {
		t.Fatalf("decodeLatents: %v", err)
	}

	want := ref.RawData()
	if len(pcm) != len(want) {
		t.Fatalf("pcm has %d samples, want %d", len(pcm), len(want))
	}

	n := int(m.Mimi().SampleRate() / 200)

	for i := range want {
		w := want[i]
		if i < n {
			w *= float32(i) / float32(n-1)
		}

		if pcm[i] != w {
			t.Fatalf("pcm[%d] = %v, want %v (decoded %v, fade over %d samples)", i, pcm[i], w, want[i], n)
		}
	}
}

// checkVoice primes the real model with the voice alone: a model-state voice
// missing one of the model's layers and an embedding of the wrong width fail,
// the shipped voice and a 1024-wide embedding pass.
func TestCheckVoice_RealGermanCheckpoint(t *testing.T) {
	modelPath := filepath.Join("..", "..", "models", "german", "model.safetensors")
	voicePath := filepath.Join("..", "..", "voices", "german", "juergen.safetensors")

	for _, p := range []string{modelPath, voicePath} {
		_, err := os.Stat(p)
		if err != nil {
			t.Skipf("german assets not available: %v", err)
		}
	}

	mc, err := modelcfg.Lookup("german")
	if err != nil {
		t.Fatal(err)
	}

	m, err := nativemodel.LoadModelFromSafetensors(modelPath, nativemodel.ConfigFor(mc))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	rt := &nativeSafetensorsRuntime{model: m}

	state, err := safetensors.LoadVoiceModelState(voicePath)
	if err != nil {
		t.Fatal(err)
	}

	err = rt.checkVoice(voiceConditioning{modelState: state})
	if err != nil {
		t.Errorf("checkVoice(juergen) = %v; want nil", err)
	}

	err = rt.checkVoice(voiceConditioning{embedding: &VoiceEmbedding{Data: make([]float32, 2*1024), Shape: []int64{1, 2, 1024}}})
	if err != nil {
		t.Errorf("checkVoice(1024-wide embedding) = %v; want nil", err)
	}

	err = rt.checkVoice(voiceConditioning{embedding: &VoiceEmbedding{Data: make([]float32, 2*512), Shape: []int64{1, 2, 512}}})
	if err == nil {
		t.Error("checkVoice(512-wide embedding) = nil; want a width error")
	}

	// A voice state for a model with fewer layers lacks the last layer's module.
	const last = "transformer.layers.5.self_attn"
	if state.Modules[last] == nil {
		t.Fatalf("juergen has no module %q; the german model has 6 layers", last)
	}

	delete(state.Modules, last)

	err = rt.checkVoice(voiceConditioning{modelState: state})
	if err == nil || !strings.Contains(err.Error(), last) {
		t.Errorf("checkVoice(state without %s) = %v; want a missing-module error", last, err)
	}
}
