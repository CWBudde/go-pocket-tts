package onnx

import (
	"context"
	"testing"
)

// ---------------------------------------------------------------------------
// Unit tests for ConcatTensorsDim1
// ---------------------------------------------------------------------------

func TestConcatTensorsDim1_BasicConcat(t *testing.T) {
	// a [1, 2, 3] + b [1, 3, 3] → [1, 5, 3]
	aData := []float32{1, 2, 3, 4, 5, 6}
	bData := []float32{7, 8, 9, 10, 11, 12, 13, 14, 15}
	a, _ := NewTensor(aData, []int64{1, 2, 3})
	b, _ := NewTensor(bData, []int64{1, 3, 3})

	result, err := ConcatTensorsDim1(a, b)
	if err != nil {
		t.Fatalf("ConcatTensorsDim1: %v", err)
	}

	shape := result.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 5 || shape[2] != 3 {
		t.Errorf("shape = %v, want [1 5 3]", shape)
	}

	data, _ := ExtractFloat32(result)

	want := append(aData, bData...)
	if len(data) != len(want) {
		t.Fatalf("data length = %d, want %d", len(data), len(want))
	}

	for i := range want {
		if data[i] != want[i] {
			t.Errorf("data[%d] = %v, want %v", i, data[i], want[i])
			break
		}
	}
}

func TestConcatTensorsDim1_DifferentDim2_ReturnsError(t *testing.T) {
	// a [1, 2, 3] + b [1, 2, 4] → error (dim2 mismatch)
	a, _ := NewTensor(make([]float32, 6), []int64{1, 2, 3})
	b, _ := NewTensor(make([]float32, 8), []int64{1, 2, 4})

	_, err := ConcatTensorsDim1(a, b)
	if err == nil {
		t.Fatal("expected error for mismatched dim2")
	}
}

func TestConcatTensorsDim1_DifferentBatch_ReturnsError(t *testing.T) {
	// a [1, 2, 3] + b [2, 2, 3] → error (batch mismatch)
	a, _ := NewTensor(make([]float32, 6), []int64{1, 2, 3})
	b, _ := NewTensor(make([]float32, 12), []int64{2, 2, 3})

	_, err := ConcatTensorsDim1(a, b)
	if err == nil {
		t.Fatal("expected error for mismatched batch dim")
	}
}

func TestConcatTensorsDim1_Not3D_ReturnsError(t *testing.T) {
	// 2D tensors → error
	a, _ := NewTensor(make([]float32, 6), []int64{2, 3})
	b, _ := NewTensor(make([]float32, 6), []int64{2, 3})

	_, err := ConcatTensorsDim1(a, b)
	if err == nil {
		t.Fatal("expected error for 2D tensors")
	}
}

// ---------------------------------------------------------------------------
// Unit tests for GenerateAudio with voice embedding
// ---------------------------------------------------------------------------

func TestGenerateAudio_WithVoiceEmbedding_PrependsToTextEmb(t *testing.T) {
	// Verify that when VoiceEmbedding is set, the text_embeddings passed to
	// flow_lm_main have the voice frames prepended.
	//
	// voice: [1, 2, 1024]
	// text:  [1, 5, 1024] (from 5 tokens)
	// expected combined: [1, 7, 1024]
	var capturedTextEmbShape []int64

	textCond := &fakeRunner{
		name: "text_conditioner",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			tokTensor := inputs["tokens"]
			T := tokTensor.Shape()[1]
			data := make([]float32, T*1024)
			out, _ := NewTensor(data, []int64{1, T, 1024})

			return map[string]*Tensor{"text_embeddings": out}, nil
		},
	}

	stepCount := 0
	flowMain := &fakeRunner{
		name: "flow_lm_main",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			stepCount++
			// Capture the text_embeddings shape on first step.
			if stepCount == 1 {
				textEmb := inputs["text_embeddings"]
				capturedTextEmbShape = textEmb.Shape()
			}

			h, _ := NewTensor(make([]float32, 1024), []int64{1, 1024})
			// Fire EOS immediately.
			eos, _ := NewTensor([]float32{0.0}, []int64{1, 1})

			return map[string]*Tensor{"last_hidden": h, "eos_logits": eos}, nil
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
			T := inputs["latent"].Shape()[1]
			out, _ := NewTensor(make([]float32, 512*T), []int64{1, 512, T})

			return map[string]*Tensor{"mimi_latent": out}, nil
		},
	}

	mimiDecoder := &fakeRunner{
		name: "mimi_decoder",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			T := inputs["latent"].Shape()[2]
			out, _ := NewTensor(make([]float32, T*480), []int64{1, 1, T * 480})

			return map[string]*Tensor{"audio": out}, nil
		},
	}

	e := engineWithFakeRunners(map[string]runnerIface{
		"text_conditioner": textCond,
		"flow_lm_main":     flowMain,
		"flow_lm_flow":     flowFlow,
		"latent_to_mimi":   latentToMimi,
		"mimi_decoder":     mimiDecoder,
	})

	// Create a voice embedding [1, 2, 1024].
	voiceData := make([]float32, 2*1024)
	for i := range voiceData {
		voiceData[i] = 99.0 // distinctive value
	}

	voiceEmb, _ := NewTensor(voiceData, []int64{1, 2, 1024})

	cfg := GenerateConfig{
		Temperature:    0.0,
		EOSThreshold:   -4.0,
		MaxSteps:       256,
		LSDDecodeSteps: 1,
		VoiceEmbedding: voiceEmb,
	}

	tokens := []int64{1, 2, 3, 4, 5}

	_, err := e.GenerateAudio(context.Background(), tokens, cfg)
	if err != nil {
		t.Fatalf("GenerateAudio: %v", err)
	}

	// text_embeddings passed to flow_lm_main should be [1, 7, 1024]
	// (2 voice frames + 5 text frames).
	if len(capturedTextEmbShape) != 3 {
		t.Fatalf("text_embeddings not 3D: %v", capturedTextEmbShape)
	}

	if capturedTextEmbShape[1] != 7 {
		t.Errorf("text_embeddings dim1 = %d, want 7 (2 voice + 5 text)", capturedTextEmbShape[1])
	}
}

func TestGenerateAudio_WithoutVoiceEmbedding_Unchanged(t *testing.T) {
	// Without voice embedding, text_embeddings should match the token count.
	var capturedTextEmbShape []int64

	textCond := &fakeRunner{
		name: "text_conditioner",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			T := inputs["tokens"].Shape()[1]
			out, _ := NewTensor(make([]float32, T*1024), []int64{1, T, 1024})

			return map[string]*Tensor{"text_embeddings": out}, nil
		},
	}

	stepCount := 0
	flowMain := &fakeRunner{
		name: "flow_lm_main",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			stepCount++
			if stepCount == 1 {
				capturedTextEmbShape = inputs["text_embeddings"].Shape()
			}

			h, _ := NewTensor(make([]float32, 1024), []int64{1, 1024})
			eos, _ := NewTensor([]float32{0.0}, []int64{1, 1}) // EOS immediately

			return map[string]*Tensor{"last_hidden": h, "eos_logits": eos}, nil
		},
	}

	e := engineWithFakeRunners(map[string]runnerIface{
		"text_conditioner": textCond,
		"flow_lm_main":     flowMain,
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
		// VoiceEmbedding: nil — no voice
	}

	tokens := []int64{1, 2, 3}

	_, err := e.GenerateAudio(context.Background(), tokens, cfg)
	if err != nil {
		t.Fatalf("GenerateAudio: %v", err)
	}

	if capturedTextEmbShape[1] != 3 {
		t.Errorf("text_embeddings dim1 = %d, want 3 (no voice prefix)", capturedTextEmbShape[1])
	}
}
