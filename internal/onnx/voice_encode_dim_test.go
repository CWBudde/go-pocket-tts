package onnx

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

// newModelLatentDim is mimi.inner_dim of every upstream model after
// english_2026-01 (german, english_2026-09, …).
const newModelLatentDim = 32

// writeSpeakerProj writes a checkpoint holding only flow_lm.speaker_proj_weight
// [VoiceEmbeddingDim, dim] and returns its path.
func writeSpeakerProj(t *testing.T, weight []float32, dim int) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "model.safetensors")

	err := safetensors.WriteFile(path, []safetensors.Tensor{{
		Name:  "flow_lm.speaker_proj_weight",
		Shape: []int64{VoiceEmbeddingDim, int64(dim)},
		Data:  weight,
	}})
	if err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	return path
}

// TestEncodeVoiceSamples_LatentDimFromConfig runs a new-model encoder: a
// channel-first [1, 32, T] latent projected by speaker_proj_weight [1024, 32]
// loaded from the checkpoint.
func TestEncodeVoiceSamples_LatentDimFromConfig(t *testing.T) {
	const dim = newModelLatentDim

	// Channel-first [1, 32, 2]: frame 0 has channels (1, 2, 0, …), frame 1 (3, 4, 0, …).
	raw := make([]float32, dim*2)
	raw[0*2+0], raw[1*2+0] = 1, 2
	raw[0*2+1], raw[1*2+1] = 3, 4

	latent, err := NewTensor(raw, []int64{1, dim, 2})
	if err != nil {
		t.Fatalf("NewTensor latent: %v", err)
	}

	weight := make([]float32, VoiceEmbeddingDim*dim)
	weight[0*dim+0], weight[0*dim+1] = 1, 1        // out 0 = c0 + c1
	weight[1023*dim+0], weight[1023*dim+1] = 2, -1 // out 1023 = 2·c0 − c1

	fake := &fakeRunner{
		name: "mimi_encoder",
		fn: func(context.Context, map[string]*Tensor) (map[string]*Tensor, error) {
			return map[string]*Tensor{"latent": latent}, nil
		},
	}

	e := engineWithFakeRunners(map[string]runnerIface{"mimi_encoder": fake})
	e.latentDim = dim
	e.modelWeightsPath = writeSpeakerProj(t, weight, dim)

	got, err := e.encodeVoiceSamples(context.Background(), []float32{0.1, 0.2})
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

	for _, c := range []struct {
		idx  int
		want float32
	}{
		{0, 3}, {1023, 0}, // frame 0: 1+2, 2·1−2
		{VoiceEmbeddingDim, 7}, {VoiceEmbeddingDim + 1023, 2}, // frame 1: 3+4, 2·3−4
	} {
		if data[c.idx] != c.want {
			t.Errorf("embedding[%d] = %v, want %v", c.idx, data[c.idx], c.want)
		}
	}
}

// TestSpeakerProjectionWeight_ShapeFollowsLatentDim checks that the checkpoint
// weight is loaded as [1024, latent dim]: a new-model [1024, 32] weight loads
// only with dim 32, the english_2026-01 [1024, 512] one with the default.
func TestSpeakerProjectionWeight_ShapeFollowsLatentDim(t *testing.T) {
	for _, tc := range []struct {
		name      string
		weightDim int
		latentDim int
		wantErr   bool
	}{
		{name: "new model", weightDim: newModelLatentDim, latentDim: newModelLatentDim},
		{name: "english_2026-01 default", weightDim: defaultMimiEncoderLatentDim, latentDim: 0},
		{name: "new model weight, default dim", weightDim: newModelLatentDim, latentDim: 0, wantErr: true},
		{name: "old weight, new model dim", weightDim: defaultMimiEncoderLatentDim, latentDim: newModelLatentDim, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := engineWithFakeRunners(nil)
			e.latentDim = tc.latentDim
			e.modelWeightsPath = writeSpeakerProj(t, make([]float32, VoiceEmbeddingDim*tc.weightDim), tc.weightDim)

			got, err := e.speakerProjectionWeight()
			if tc.wantErr {
				if err == nil || !strings.Contains(err.Error(), "speaker_proj_weight") {
					t.Fatalf("err = %v, want a speaker_proj_weight shape error", err)
				}

				return
			}

			if err != nil {
				t.Fatalf("speakerProjectionWeight: %v", err)
			}

			if len(got) != VoiceEmbeddingDim*tc.weightDim {
				t.Fatalf("weight has %d values, want %d", len(got), VoiceEmbeddingDim*tc.weightDim)
			}
		})
	}
}

// TestEncodeVoiceSamples_LatentDimMismatchNamesShape checks that a 512-wide
// encoder (the english_2026-01 ONNX export) under a new-model config fails
// with the dimension it expected.
func TestEncodeVoiceSamples_LatentDimMismatchNamesShape(t *testing.T) {
	latent, err := NewTensor(make([]float32, 2*defaultMimiEncoderLatentDim), []int64{1, 2, defaultMimiEncoderLatentDim})
	if err != nil {
		t.Fatalf("NewTensor latent: %v", err)
	}

	fake := &fakeRunner{
		name: "mimi_encoder",
		fn: func(context.Context, map[string]*Tensor) (map[string]*Tensor, error) {
			return map[string]*Tensor{"latent": latent}, nil
		},
	}

	e := engineWithFakeRunners(map[string]runnerIface{"mimi_encoder": fake})
	e.latentDim = newModelLatentDim
	e.speakerProjWeight = make([]float32, VoiceEmbeddingDim*newModelLatentDim)

	_, err = e.encodeVoiceSamples(context.Background(), []float32{0.1})
	if err == nil || !strings.Contains(err.Error(), "[1,T,32] or [1,32,T]") {
		t.Fatalf("err = %v, want an unexpected latent shape error naming [1,T,32] or [1,32,T]", err)
	}
}
