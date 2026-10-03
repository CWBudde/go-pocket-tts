package tts

import (
	"context"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	nativemodel "github.com/cwbudde/go-pocket-tts/internal/native"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
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
