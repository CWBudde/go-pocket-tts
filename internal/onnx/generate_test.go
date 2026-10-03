package onnx

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"
)

// fakeStatefulEngine builds an Engine with fake runners for the stateful path.
// EOS fires after eosAfterSteps step calls.
func fakeStatefulEngine(t *testing.T, eosAfterSteps int) *Engine {
	t.Helper()

	stepCount := 0
	const numLayers = 2
	T := int64(3)

	textCond := &fakeRunner{
		name: "text_conditioner",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			tLen := inputs["tokens"].Shape()[1]
			out, _ := NewTensor(make([]float32, tLen*1024), []int64{1, tLen, 1024})

			return map[string]*Tensor{"text_embeddings": out}, nil
		},
	}
	prefill := &fakeRunner{
		name: "flow_lm_prefill",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			out := map[string]*Tensor{}

			for i := range numLayers {
				kv, _ := NewTensor(make([]float32, 2*1*int(T)*2*4), []int64{2, 1, T, 2, 4})
				out[fmt.Sprintf("kv_%d", i)] = kv
			}

			off, _ := NewTensor([]int64{T}, []int64{1})
			out["offset"] = off

			return out, nil
		},
	}
	step := &fakeRunner{
		name: "flow_lm_step",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			stepCount++
			newOff := T + int64(stepCount)
			out := map[string]*Tensor{}

			for i := range numLayers {
				kv, _ := NewTensor(make([]float32, 2*1*int(newOff)*2*4), []int64{2, 1, newOff, 2, 4})
				out[fmt.Sprintf("kv_%d", i)] = kv
			}

			off, _ := NewTensor([]int64{newOff}, []int64{1})
			out["offset"] = off
			hidden, _ := NewTensor(make([]float32, 1024), []int64{1, 1024})
			out["last_hidden"] = hidden

			eosVal := float32(-10.0)
			if stepCount >= eosAfterSteps {
				eosVal = 0.0
			}

			eos, _ := NewTensor([]float32{eosVal}, []int64{1, 1})
			out["eos_logits"] = eos

			return out, nil
		},
	}
	flowFlow := &fakeRunner{
		name: "flow_lm_flow",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			out, _ := NewTensor(make([]float32, 32), []int64{1, 32})
			return map[string]*Tensor{"flow_direction": out}, nil
		},
	}
	latentToMimi := &fakeRunner{
		name: "latent_to_mimi",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			tLen := inputs["latent"].Shape()[1]
			out, _ := NewTensor(make([]float32, 512*tLen), []int64{1, 512, tLen})

			return map[string]*Tensor{"mimi_latent": out}, nil
		},
	}
	mimiDecoder := &fakeRunner{
		name: "mimi_decoder",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			tLen := inputs["latent"].Shape()[2]
			out, _ := NewTensor(make([]float32, tLen*480), []int64{1, 1, tLen * 480})

			return map[string]*Tensor{"audio": out}, nil
		},
	}

	return engineWithFakeRunners(map[string]runnerIface{
		"text_conditioner": textCond,
		"flow_lm_prefill":  prefill,
		"flow_lm_step":     step,
		"flow_lm_flow":     flowFlow,
		"latent_to_mimi":   latentToMimi,
		"mimi_decoder":     mimiDecoder,
	})
}

func TestGenerateAudio_StatefulPath_ProducesNonEmptyPCM(t *testing.T) {
	e := fakeStatefulEngine(t, 3)
	cfg := GenerateConfig{Temperature: 0.0, EOSThreshold: -4.0, MaxSteps: 256, SamplerDecodeSteps: 1}

	pcm, err := e.GenerateAudio(context.Background(), []int64{1, 2, 3}, cfg)
	if err != nil {
		t.Fatalf("GenerateAudio (stateful): %v", err)
	}

	if len(pcm) == 0 {
		t.Fatal("expected non-empty PCM from stateful path")
	}
}

func TestGenerateAudio_FallbackToStateless_WhenNoPrefillGraph(t *testing.T) {
	// fakeGenerateEngine only has flow_lm_main (no flow_lm_prefill) → stateless fallback.
	e := fakeGenerateEngine(t, 3)
	cfg := GenerateConfig{Temperature: 0.0, EOSThreshold: -4.0, MaxSteps: 256, SamplerDecodeSteps: 1}

	pcm, err := e.GenerateAudio(context.Background(), []int64{1, 2, 3}, cfg)
	if err != nil {
		t.Fatalf("GenerateAudio (stateless fallback): %v", err)
	}

	if len(pcm) == 0 {
		t.Fatal("expected non-empty PCM from stateless fallback")
	}
}

// ---------------------------------------------------------------------------
// Unit tests for Engine.GenerateAudio — full pipeline orchestration
// ---------------------------------------------------------------------------

// fakeGenerateEngine builds an Engine with fake runners for all 4 graphs.
// The flow_lm_main runner triggers EOS after eosAfterSteps calls.
func fakeGenerateEngine(t *testing.T, eosAfterSteps int) *Engine {
	t.Helper()

	stepCount := 0

	textCond := &fakeRunner{
		name: "text_conditioner",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			// Return fake embeddings [1, T, 1024].
			tokTensor := inputs["tokens"]
			T := tokTensor.Shape()[1]
			data := make([]float32, T*1024)
			out, _ := NewTensor(data, []int64{1, T, 1024})

			return map[string]*Tensor{"text_embeddings": out}, nil
		},
	}

	flowMain := &fakeRunner{
		name: "flow_lm_main",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			stepCount++
			hidden := make([]float32, 1024)
			h, _ := NewTensor(hidden, []int64{1, 1024})

			// Trigger EOS after configured number of steps.
			eosVal := float32(-10.0) // below threshold
			if stepCount >= eosAfterSteps {
				eosVal = 0.0 // above default threshold -4.0
			}

			eos, _ := NewTensor([]float32{eosVal}, []int64{1, 1})

			return map[string]*Tensor{"last_hidden": h, "eos_logits": eos}, nil
		},
	}

	flowFlow := &fakeRunner{
		name: "flow_lm_flow",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			dir := make([]float32, 32)
			for i := range dir {
				dir[i] = 0.5
			}

			out, _ := NewTensor(dir, []int64{1, 32})

			return map[string]*Tensor{"flow_direction": out}, nil
		},
	}

	latentToMimi := &fakeRunner{
		name: "latent_to_mimi",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			T := inputs["latent"].Shape()[1]
			data := make([]float32, 512*T)
			out, _ := NewTensor(data, []int64{1, 512, T})

			return map[string]*Tensor{"mimi_latent": out}, nil
		},
	}

	mimiDecoder := &fakeRunner{
		name: "mimi_decoder",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			T := inputs["latent"].Shape()[2]
			// Each latent frame produces 480 audio samples (24kHz / 50fps).
			nSamples := T * 480

			data := make([]float32, nSamples)
			for i := range data {
				data[i] = 0.1
			}

			out, _ := NewTensor(data, []int64{1, 1, nSamples})

			return map[string]*Tensor{"audio": out}, nil
		},
	}

	return engineWithFakeRunners(map[string]runnerIface{
		"text_conditioner": textCond,
		"flow_lm_main":     flowMain,
		"flow_lm_flow":     flowFlow,
		"latent_to_mimi":   latentToMimi,
		"mimi_decoder":     mimiDecoder,
	})
}

func TestGenerateAudio_ProducesNonEmptyPCM(t *testing.T) {
	e := fakeGenerateEngine(t, 3) // EOS fires on step 3

	cfg := GenerateConfig{
		Temperature:        0.0, // deterministic
		EOSThreshold:       -4.0,
		MaxSteps:           256,
		SamplerDecodeSteps: 1,
	}

	tokens := []int64{1, 2, 3, 4, 5}

	pcm, err := e.GenerateAudio(context.Background(), tokens, cfg)
	if err != nil {
		t.Fatalf("GenerateAudio: %v", err)
	}

	if len(pcm) == 0 {
		t.Fatal("expected non-empty PCM output")
	}
}

func TestGenerateAudio_RespectsMaxSteps(t *testing.T) {
	// EOS never fires (threshold very high) → should stop at MaxSteps.
	stepCount := 0
	flowMain := &fakeRunner{
		name: "flow_lm_main",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			stepCount++
			h, _ := NewTensor(make([]float32, 1024), []int64{1, 1024})
			eos, _ := NewTensor([]float32{-10.0}, []int64{1, 1}) // never fires

			return map[string]*Tensor{"last_hidden": h, "eos_logits": eos}, nil
		},
	}

	e := engineWithFakeRunners(map[string]runnerIface{
		"text_conditioner": &fakeRunner{
			name: "text_conditioner",
			fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
				T := inputs["tokens"].Shape()[1]
				out, _ := NewTensor(make([]float32, T*1024), []int64{1, T, 1024})

				return map[string]*Tensor{"text_embeddings": out}, nil
			},
		},
		"flow_lm_main": flowMain,
		"flow_lm_flow": &fakeRunner{
			name: "flow_lm_flow",
			fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
				out, _ := NewTensor(make([]float32, 32), []int64{1, 32})
				return map[string]*Tensor{"flow_direction": out}, nil
			},
		},
		"latent_to_mimi": &fakeRunner{
			name: "latent_to_mimi",
			fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
				T := inputs["latent"].Shape()[1]
				out, _ := NewTensor(make([]float32, 512*T), []int64{1, 512, T})

				return map[string]*Tensor{"mimi_latent": out}, nil
			},
		},
		"mimi_decoder": &fakeRunner{
			name: "mimi_decoder",
			fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
				T := inputs["latent"].Shape()[2]
				out, _ := NewTensor(make([]float32, T*480), []int64{1, 1, T * 480})

				return map[string]*Tensor{"audio": out}, nil
			},
		},
	})

	cfg := GenerateConfig{
		Temperature:        0.0,
		EOSThreshold:       -4.0,
		MaxSteps:           10,
		SamplerDecodeSteps: 1,
	}

	_, err := e.GenerateAudio(context.Background(), []int64{1, 2, 3}, cfg)
	if err != nil {
		t.Fatalf("GenerateAudio: %v", err)
	}

	if stepCount != 10 {
		t.Errorf("flow_lm_main called %d times, want 10 (MaxSteps)", stepCount)
	}
}

// countRuns wraps the named runner of e and returns its call counter.
func countRuns(e *Engine, name string) *int {
	calls := 0
	inner := e.runners[name]
	e.runners[name] = &fakeRunner{
		name: name,
		fn: func(ctx context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			calls++
			return inner.Run(ctx, inputs)
		},
	}

	return &calls
}

// generatePaths are the two AR loops with the name of their per-step graph.
var generatePaths = []struct {
	name      string
	engine    func(t *testing.T, eosAfterSteps int) *Engine
	stepGraph string
}{
	{"stateless", fakeGenerateEngine, "flow_lm_main"},
	{"stateful", fakeStatefulEngine, "flow_lm_step"},
}

func TestGenerateAudio_EOSStopRule(t *testing.T) {
	// The fakes flag EOS from their third step call (step 2) on. EOS is only
	// accepted from step genloop.MinFramesBeforeEOS = 6, and upstream keeps
	// eos_step + FramesAfterEOS frames: the step that stops the loop runs the
	// backbone but neither the flow head nor the decoder.
	for _, path := range generatePaths {
		for _, tc := range []struct {
			framesAfter int
			wantFrames  int
		}{
			{3, 9},
			{0, 6},
		} {
			t.Run(fmt.Sprintf("%s/frames_after_eos=%d", path.name, tc.framesAfter), func(t *testing.T) {
				e := path.engine(t, 3)
				steps := countRuns(e, path.stepGraph)
				flows := countRuns(e, "flow_lm_flow")

				pcm, err := e.GenerateAudio(context.Background(), []int64{1, 2, 3}, GenerateConfig{
					EOSThreshold:       -4.0,
					MaxSteps:           256,
					SamplerDecodeSteps: 1,
					FramesAfterEOS:     tc.framesAfter,
				})
				if err != nil {
					t.Fatalf("GenerateAudio: %v", err)
				}

				if *flows != tc.wantFrames || len(pcm) != tc.wantFrames*480 {
					t.Errorf("frames = %d (pcm %d samples), want %d", *flows, len(pcm), tc.wantFrames)
				}

				if *steps != tc.wantFrames+1 {
					t.Errorf("%s calls = %d, want %d", path.stepGraph, *steps, tc.wantFrames+1)
				}
			})
		}
	}
}

func TestGenerateAudio_MissingTextConditioner(t *testing.T) {
	e := engineWithFakeRunners(map[string]runnerIface{})
	cfg := GenerateConfig{MaxSteps: 10}

	_, err := e.GenerateAudio(context.Background(), []int64{1}, cfg)
	if err == nil {
		t.Fatal("expected error when text_conditioner is missing")
	}
}

func TestGenerateAudio_EmptyTokens(t *testing.T) {
	e := fakeGenerateEngine(t, 1)
	cfg := GenerateConfig{MaxSteps: 10}

	_, err := e.GenerateAudio(context.Background(), nil, cfg)
	if err == nil {
		t.Fatal("expected error for empty tokens")
	}
}

func TestGenerateAudio_PropagatesFlowLMError(t *testing.T) {
	e := engineWithFakeRunners(map[string]runnerIface{
		"text_conditioner": &fakeRunner{
			name: "text_conditioner",
			fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
				T := inputs["tokens"].Shape()[1]
				out, _ := NewTensor(make([]float32, T*1024), []int64{1, T, 1024})

				return map[string]*Tensor{"text_embeddings": out}, nil
			},
		},
		"flow_lm_main": &fakeRunner{
			name: "flow_lm_main",
			fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
				return nil, errors.New("flow_lm_main exploded")
			},
		},
	})
	cfg := GenerateConfig{MaxSteps: 10, EOSThreshold: -4.0, SamplerDecodeSteps: 1}

	_, err := e.GenerateAudio(context.Background(), []int64{1, 2}, cfg)
	if err == nil {
		t.Fatal("expected error from flow_lm_main")
	}
}

// TestGenerateAudio_NaNFromFlowLM_ProducesDetectableSilence is a regression
// test for the bug where the ONNX flow_lm_main graph was missing the
// torch.where(isnan(sequence), bos_emb, sequence) substitution. The BOS token
// is represented as NaN, and without that substitution all hidden states become
// NaN, which propagates silently through the pipeline and yields a silent WAV.
//
// This test documents the failure mode: NaN hidden state → NaN latent frames →
// NaN/zero mimi output → silence. The test asserts that GenerateAudio detects
// and reports non-finite PCM when fed by a corrupt (NaN-returning) runner.
func TestGenerateAudio_NaNHiddenStateProducesSilence(t *testing.T) {
	// Simulate the buggy ONNX model: flow_lm_main returns NaN for last_hidden,
	// which is what happened before the bos_emb substitution fix.
	nanHidden := make([]float32, 1024)
	for i := range nanHidden {
		nanHidden[i] = float32(math.NaN())
	}

	flowMain := &fakeRunner{
		name: "flow_lm_main",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			h, _ := NewTensor(nanHidden, []int64{1, 1024})
			// NaN logit: NaN > -4.0 is false, so EOS never fires.
			eos, _ := NewTensor([]float32{float32(math.NaN())}, []int64{1, 1})

			return map[string]*Tensor{"last_hidden": h, "eos_logits": eos}, nil
		},
	}
	flowFlow := &fakeRunner{
		name: "flow_lm_flow",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			// NaN condition → NaN flow direction.
			dir := make([]float32, 32)
			for i := range dir {
				dir[i] = float32(math.NaN())
			}

			out, _ := NewTensor(dir, []int64{1, 32})

			return map[string]*Tensor{"flow_direction": out}, nil
		},
	}

	e := engineWithFakeRunners(map[string]runnerIface{
		"text_conditioner": &fakeRunner{
			name: "text_conditioner",
			fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
				T := inputs["tokens"].Shape()[1]
				out, _ := NewTensor(make([]float32, T*1024), []int64{1, T, 1024})

				return map[string]*Tensor{"text_embeddings": out}, nil
			},
		},
		"flow_lm_main": flowMain,
		"flow_lm_flow": flowFlow,
		"latent_to_mimi": &fakeRunner{
			name: "latent_to_mimi",
			fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
				T := inputs["latent"].Shape()[1]
				// NaN latent → NaN mimi_latent.
				data := make([]float32, 512*T)
				for i := range data {
					data[i] = float32(math.NaN())
				}

				out, _ := NewTensor(data, []int64{1, 512, T})

				return map[string]*Tensor{"mimi_latent": out}, nil
			},
		},
		"mimi_decoder": &fakeRunner{
			name: "mimi_decoder",
			fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
				T := inputs["latent"].Shape()[2]
				// NaN mimi_latent → NaN or zero PCM (mimi_decoder clamps/zeros NaN).
				data := make([]float32, T*480)
				for i := range data {
					data[i] = float32(math.NaN())
				}

				out, _ := NewTensor(data, []int64{1, 1, T * 480})

				return map[string]*Tensor{"audio": out}, nil
			},
		},
	})

	cfg := GenerateConfig{
		Temperature:        0.0,
		EOSThreshold:       -4.0,
		MaxSteps:           5, // limit steps; NaN EOS never fires
		SamplerDecodeSteps: 1,
	}

	pcm, err := e.GenerateAudio(context.Background(), []int64{1, 2, 3}, cfg)
	if err != nil {
		t.Fatalf("GenerateAudio: %v", err)
	}

	// With NaN propagation the PCM must contain NaN or have zero RMS (silence).
	// Either symptom indicates the pipeline is corrupted.
	var sumSq float64
	hasNaN := false

	for _, s := range pcm {
		if math.IsNaN(float64(s)) {
			hasNaN = true
			break
		}

		sumSq += float64(s) * float64(s)
	}

	rms := math.Sqrt(sumSq / float64(len(pcm)))

	if !hasNaN && rms >= 0.01 {
		t.Errorf("expected NaN or silence from corrupt pipeline, got rms=%.6f", rms)
	}
	// Log what we got so future readers can see the failure mode.
	t.Logf("NaN pipeline: hasNaN=%v rms=%.6f (expected corrupt/silent output)", hasNaN, rms)
}
