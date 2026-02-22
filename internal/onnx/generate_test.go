package onnx

import (
	"context"
	"fmt"
	"testing"
)

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
		Temperature:    0.0, // deterministic
		EOSThreshold:   -4.0,
		MaxSteps:       256,
		LSDDecodeSteps: 1,
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
		Temperature:    0.0,
		EOSThreshold:   -4.0,
		MaxSteps:       10,
		LSDDecodeSteps: 1,
	}

	_, err := e.GenerateAudio(context.Background(), []int64{1, 2, 3}, cfg)
	if err != nil {
		t.Fatalf("GenerateAudio: %v", err)
	}
	if stepCount != 10 {
		t.Errorf("flow_lm_main called %d times, want 10 (MaxSteps)", stepCount)
	}
}

func TestGenerateAudio_EOSCountdown(t *testing.T) {
	// EOS fires on step 2, framesAfterEOS = 3.
	// Should generate steps: 1(no), 2(EOS, countdown=3), 3(2), 4(1), 5(0→stop).
	// Total = 5 steps.
	stepCount := 0
	flowMain := &fakeRunner{
		name: "flow_lm_main",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			stepCount++
			h, _ := NewTensor(make([]float32, 1024), []int64{1, 1024})
			eosVal := float32(-10.0)
			if stepCount >= 2 {
				eosVal = 0.0 // fires on step 2+
			}
			eos, _ := NewTensor([]float32{eosVal}, []int64{1, 1})
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
		Temperature:    0.0,
		EOSThreshold:   -4.0,
		MaxSteps:       256,
		LSDDecodeSteps: 1,
		FramesAfterEOS: 3,
	}

	_, err := e.GenerateAudio(context.Background(), []int64{1, 2, 3}, cfg)
	if err != nil {
		t.Fatalf("GenerateAudio: %v", err)
	}
	// EOS on step 2, countdown 3 → steps 2,3,4,5 (stop at 5).
	// Total = 5 steps.
	if stepCount != 5 {
		t.Errorf("flow_lm_main called %d times, want 5 (EOS at 2 + 3 after)", stepCount)
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
				return nil, fmt.Errorf("flow_lm_main exploded")
			},
		},
	})
	cfg := GenerateConfig{MaxSteps: 10, EOSThreshold: -4.0, LSDDecodeSteps: 1}

	_, err := e.GenerateAudio(context.Background(), []int64{1, 2}, cfg)
	if err == nil {
		t.Fatal("expected error from flow_lm_main")
	}
}
