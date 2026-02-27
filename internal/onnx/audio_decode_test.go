package onnx

import (
	"context"
	"errors"
	"testing"
)

// ---------------------------------------------------------------------------
// Unit tests for StackLatentFrames
// ---------------------------------------------------------------------------

func TestStackLatentFrames_SingleFrame(t *testing.T) {
	frame, _ := NewTensor(make([]float32, 32), []int64{1, 1, 32})

	result, err := StackLatentFrames([]*Tensor{frame})
	if err != nil {
		t.Fatalf("StackLatentFrames: %v", err)
	}

	shape := result.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 32 {
		t.Errorf("shape = %v, want [1 1 32]", shape)
	}
}

func TestStackLatentFrames_MultipleFrames(t *testing.T) {
	frames := make([]*Tensor, 5)
	for i := range frames {
		data := make([]float32, 32)
		for j := range data {
			data[j] = float32(i)
		}

		frames[i], _ = NewTensor(data, []int64{1, 1, 32})
	}

	result, err := StackLatentFrames(frames)
	if err != nil {
		t.Fatalf("StackLatentFrames: %v", err)
	}

	// Shape: [1, 5, 32].
	shape := result.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 5 || shape[2] != 32 {
		t.Errorf("shape = %v, want [1 5 32]", shape)
	}

	// Verify data ordering: frame i has all values = float32(i).
	data, _ := ExtractFloat32(result)

	for i := range 5 {
		for j := range 32 {
			got := data[i*32+j]

			want := float32(i)
			if got != want {
				t.Errorf("data[%d][%d] = %v, want %v", i, j, got, want)
			}
		}
	}
}

func TestStackLatentFrames_Empty(t *testing.T) {
	_, err := StackLatentFrames(nil)
	if err == nil {
		t.Fatal("expected error for empty frames")
	}
}

// ---------------------------------------------------------------------------
// Unit tests for Engine.LatentToMimi
// ---------------------------------------------------------------------------

func TestLatentToMimi_MissingGraph(t *testing.T) {
	e := engineWithFakeRunners(map[string]runnerIface{})
	latent, _ := NewTensor(make([]float32, 32), []int64{1, 1, 32})

	_, err := e.LatentToMimi(context.Background(), latent)
	if err == nil {
		t.Fatal("expected error when latent_to_mimi graph is absent")
	}
}

func TestLatentToMimi_PropagatesRunnerError(t *testing.T) {
	fake := &fakeRunner{
		name: "latent_to_mimi",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			return nil, errors.New("ort failure")
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"latent_to_mimi": fake})
	latent, _ := NewTensor(make([]float32, 32), []int64{1, 1, 32})

	_, err := e.LatentToMimi(context.Background(), latent)
	if err == nil {
		t.Fatal("expected error propagated from runner")
	}
}

func TestLatentToMimi_ReturnsOutput(t *testing.T) {
	const T = 3
	// Fake output: [1, 512, T].
	outputData := make([]float32, 1*512*T)
	fakeOutput, _ := NewTensor(outputData, []int64{1, 512, T})

	fake := &fakeRunner{
		name: "latent_to_mimi",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			if _, ok := inputs["latent"]; !ok {
				t.Error("expected 'latent' input key")
			}

			return map[string]*Tensor{"mimi_latent": fakeOutput}, nil
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"latent_to_mimi": fake})
	latent, _ := NewTensor(make([]float32, T*32), []int64{1, T, 32})

	result, err := e.LatentToMimi(context.Background(), latent)
	if err != nil {
		t.Fatalf("LatentToMimi: %v", err)
	}

	shape := result.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 512 || shape[2] != T {
		t.Errorf("shape = %v, want [1 512 %d]", shape, T)
	}
}

func TestLatentToMimi_MissingOutput(t *testing.T) {
	fake := &fakeRunner{
		name: "latent_to_mimi",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			return map[string]*Tensor{}, nil
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"latent_to_mimi": fake})
	latent, _ := NewTensor(make([]float32, 32), []int64{1, 1, 32})

	_, err := e.LatentToMimi(context.Background(), latent)
	if err == nil {
		t.Fatal("expected error for missing mimi_latent output")
	}
}

// ---------------------------------------------------------------------------
// Unit tests for Engine.MimiDecode
// ---------------------------------------------------------------------------

func TestMimiDecode_MissingGraph(t *testing.T) {
	e := engineWithFakeRunners(map[string]runnerIface{})
	mimiLatent, _ := NewTensor(make([]float32, 512), []int64{1, 512, 1})

	_, err := e.MimiDecode(context.Background(), mimiLatent)
	if err == nil {
		t.Fatal("expected error when mimi_decoder graph is absent")
	}
}

func TestMimiDecode_PropagatesRunnerError(t *testing.T) {
	fake := &fakeRunner{
		name: "mimi_decoder",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			return nil, errors.New("ort failure")
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"mimi_decoder": fake})
	mimiLatent, _ := NewTensor(make([]float32, 512), []int64{1, 512, 1})

	_, err := e.MimiDecode(context.Background(), mimiLatent)
	if err == nil {
		t.Fatal("expected error propagated from runner")
	}
}

func TestMimiDecode_ReturnsAudio(t *testing.T) {
	// Mimi decoder: input "latent" [1, 512, T] → output "audio" [1, 1, N_samples].
	const nSamples = 480

	audioData := make([]float32, nSamples)
	for i := range audioData {
		audioData[i] = float32(i) * 0.001
	}

	fakeAudio, _ := NewTensor(audioData, []int64{1, 1, int64(nSamples)})

	fake := &fakeRunner{
		name: "mimi_decoder",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			if _, ok := inputs["latent"]; !ok {
				t.Error("expected 'latent' input key")
			}

			return map[string]*Tensor{"audio": fakeAudio}, nil
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"mimi_decoder": fake})
	mimiLatent, _ := NewTensor(make([]float32, 512), []int64{1, 512, 1})

	pcm, err := e.MimiDecode(context.Background(), mimiLatent)
	if err != nil {
		t.Fatalf("MimiDecode: %v", err)
	}

	if len(pcm) != nSamples {
		t.Errorf("len(pcm) = %d, want %d", len(pcm), nSamples)
	}
	// Spot-check first sample.
	if pcm[0] != 0.0 {
		t.Errorf("pcm[0] = %v, want 0.0", pcm[0])
	}
}

func TestMimiDecode_MissingAudioOutput(t *testing.T) {
	fake := &fakeRunner{
		name: "mimi_decoder",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			return map[string]*Tensor{}, nil
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"mimi_decoder": fake})
	mimiLatent, _ := NewTensor(make([]float32, 512), []int64{1, 512, 1})

	_, err := e.MimiDecode(context.Background(), mimiLatent)
	if err == nil {
		t.Fatal("expected error for missing audio output")
	}
}
