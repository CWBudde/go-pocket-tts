package onnx

import (
	"context"
	"testing"
)

func TestProjectSpeakerConditioning_KnownValues(t *testing.T) {
	latentData := make([]float32, 2*mimiEncoderLatentDim)
	latentData[0], latentData[1], latentData[2] = 1, 2, 3
	latentData[mimiEncoderLatentDim+0], latentData[mimiEncoderLatentDim+1], latentData[mimiEncoderLatentDim+2] = 4, 5, 6

	latent, err := NewTensor(latentData, []int64{1, 2, mimiEncoderLatentDim})
	if err != nil {
		t.Fatalf("NewTensor latent: %v", err)
	}

	weight := make([]float32, VoiceEmbeddingDim*mimiEncoderLatentDim)
	// Output dim 0 uses [1,0,1].
	weight[0*mimiEncoderLatentDim+0] = 1
	weight[0*mimiEncoderLatentDim+2] = 1
	// Output dim 1 uses [0.5,0.5,0].
	weight[1*mimiEncoderLatentDim+0] = 0.5
	weight[1*mimiEncoderLatentDim+1] = 0.5
	// Output dim 1023 uses [-1,1,1].
	weight[1023*mimiEncoderLatentDim+0] = -1
	weight[1023*mimiEncoderLatentDim+1] = 1
	weight[1023*mimiEncoderLatentDim+2] = 1

	got, err := projectSpeakerConditioning(latent, weight)
	if err != nil {
		t.Fatalf("projectSpeakerConditioning: %v", err)
	}

	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != VoiceEmbeddingDim {
		t.Fatalf("shape = %v, want [1 2 %d]", shape, VoiceEmbeddingDim)
	}

	data, err := ExtractFloat32(got)
	if err != nil {
		t.Fatalf("ExtractFloat32: %v", err)
	}

	// Frame 0 expectations.
	if data[0] != 4 { // 1*1 + 2*0 + 3*1
		t.Fatalf("frame0 dim0 = %v, want 4", data[0])
	}
	if data[1] != 1.5 { // 1*0.5 + 2*0.5
		t.Fatalf("frame0 dim1 = %v, want 1.5", data[1])
	}
	if data[1023] != 4 { // -1 + 2 + 3
		t.Fatalf("frame0 dim1023 = %v, want 4", data[1023])
	}

	// Frame 1 expectations.
	base := VoiceEmbeddingDim
	if data[base+0] != 10 { // 4*1 + 5*0 + 6*1
		t.Fatalf("frame1 dim0 = %v, want 10", data[base+0])
	}
	if data[base+1] != 4.5 { // 4*0.5 + 5*0.5
		t.Fatalf("frame1 dim1 = %v, want 4.5", data[base+1])
	}
	if data[base+1023] != 7 { // -4 + 5 + 6
		t.Fatalf("frame1 dim1023 = %v, want 7", data[base+1023])
	}
}

func TestNormalizeMimiEncoderLatent_TransposesChannelFirstOutput(t *testing.T) {
	raw := make([]float32, mimiEncoderLatentDim*2) // [1, 512, 2]
	for c := range mimiEncoderLatentDim {
		raw[c*2] = float32(c)
		raw[c*2+1] = float32(1000 + c)
	}

	latent, err := NewTensor(raw, []int64{1, mimiEncoderLatentDim, 2})
	if err != nil {
		t.Fatalf("NewTensor latent: %v", err)
	}

	norm, err := normalizeMimiEncoderLatent(latent)
	if err != nil {
		t.Fatalf("normalizeMimiEncoderLatent: %v", err)
	}

	if shape := norm.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != mimiEncoderLatentDim {
		t.Fatalf("shape = %v, want [1 2 %d]", shape, mimiEncoderLatentDim)
	}

	data, err := ExtractFloat32(norm)
	if err != nil {
		t.Fatalf("ExtractFloat32: %v", err)
	}
	if data[0] != 0 || data[mimiEncoderLatentDim-1] != float32(mimiEncoderLatentDim-1) {
		t.Fatalf("first frame data mismatch: got [%v ... %v]", data[0], data[mimiEncoderLatentDim-1])
	}
	if data[mimiEncoderLatentDim] != 1000 || data[mimiEncoderLatentDim*2-1] != float32(1000+mimiEncoderLatentDim-1) {
		t.Fatalf("second frame data mismatch: got [%v ... %v]", data[mimiEncoderLatentDim], data[mimiEncoderLatentDim*2-1])
	}
}

func TestEncodeVoiceSamples_RunsMimiEncoderAndProjection(t *testing.T) {
	latentData := make([]float32, 2*mimiEncoderLatentDim)
	latentData[0], latentData[1] = 2, 3
	latentData[mimiEncoderLatentDim+0], latentData[mimiEncoderLatentDim+1] = 4, 1
	latentTensor, err := NewTensor(latentData, []int64{1, 2, mimiEncoderLatentDim})
	if err != nil {
		t.Fatalf("NewTensor latent: %v", err)
	}

	weight := make([]float32, VoiceEmbeddingDim*mimiEncoderLatentDim)
	weight[0*mimiEncoderLatentDim+0] = 1
	weight[0*mimiEncoderLatentDim+1] = 1
	weight[1*mimiEncoderLatentDim+0] = 1
	weight[1*mimiEncoderLatentDim+1] = -1

	fake := &fakeRunner{
		name: "mimi_encoder",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			audioIn, ok := inputs["audio"]
			if !ok {
				t.Fatal("expected 'audio' input")
			}
			shape := audioIn.Shape()
			if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 3 {
				t.Fatalf("audio input shape = %v, want [1 1 3]", shape)
			}
			return map[string]*Tensor{"latent": latentTensor}, nil
		},
	}

	e := engineWithFakeRunners(map[string]runnerIface{"mimi_encoder": fake})
	e.speakerProjWeight = weight

	got, err := e.encodeVoiceSamples(context.Background(), []float32{0.25, -0.25, 0.5})
	if err != nil {
		t.Fatalf("encodeVoiceSamples: %v", err)
	}

	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != VoiceEmbeddingDim {
		t.Fatalf("shape = %v, want [1 2 %d]", shape, VoiceEmbeddingDim)
	}

	data, err := ExtractFloat32(got)
	if err != nil {
		t.Fatalf("ExtractFloat32: %v", err)
	}
	if data[0] != 5 || data[1] != -1 {
		t.Fatalf("frame0 projected dims = [%v %v], want [5 -1]", data[0], data[1])
	}
	if data[VoiceEmbeddingDim+0] != 5 || data[VoiceEmbeddingDim+1] != 3 {
		t.Fatalf(
			"frame1 projected dims = [%v %v], want [5 3]",
			data[VoiceEmbeddingDim+0],
			data[VoiceEmbeddingDim+1],
		)
	}
}

func TestEncodeVoiceSamples_MissingMimiEncoderGraph(t *testing.T) {
	e := engineWithFakeRunners(map[string]runnerIface{})
	e.speakerProjWeight = make([]float32, VoiceEmbeddingDim*mimiEncoderLatentDim)

	if _, err := e.encodeVoiceSamples(context.Background(), []float32{1}); err == nil {
		t.Fatal("expected error when mimi_encoder graph is missing")
	}
}
