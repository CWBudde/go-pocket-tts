package onnx

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/example/go-pocket-tts/internal/audio"
	"github.com/example/go-pocket-tts/internal/safetensors"
)

const (
	mimiEncoderLatentDim = 512
	// VoiceEmbeddingDim is the projected per-frame speaker conditioning width.
	VoiceEmbeddingDim = 1024
)

// EncodeVoice loads a WAV/PCM prompt from audioPath and returns a flattened
// voice embedding tensor with logical shape [1, T, 1024].
func (e *Engine) EncodeVoice(audioPath string) ([]float32, error) {
	samples, err := loadVoiceAudioSamples(audioPath)
	if err != nil {
		return nil, err
	}

	embedding, err := e.encodeVoiceSamples(context.Background(), samples)
	if err != nil {
		return nil, err
	}

	data, err := ExtractFloat32(embedding)
	if err != nil {
		return nil, fmt.Errorf("encode voice: extract embedding: %w", err)
	}

	return data, nil
}

func (e *Engine) encodeVoiceSamples(ctx context.Context, samples []float32) (*Tensor, error) {
	if len(samples) == 0 {
		return nil, errors.New("encode voice: empty audio samples")
	}

	runner, ok := e.runners["mimi_encoder"]
	if !ok {
		return nil, errors.New("mimi_encoder graph not found in manifest")
	}

	audioTensor, err := NewTensor(samples, []int64{1, 1, int64(len(samples))})
	if err != nil {
		return nil, fmt.Errorf("encode voice: build audio tensor: %w", err)
	}

	outputs, err := runner.Run(ctx, map[string]*Tensor{"audio": audioTensor})
	if err != nil {
		return nil, fmt.Errorf("mimi_encoder: run: %w", err)
	}

	latent, ok := outputs["latent"]
	if !ok {
		return nil, errors.New("mimi_encoder: missing 'latent' in output")
	}

	normalizedLatent, err := normalizeMimiEncoderLatent(latent)
	if err != nil {
		return nil, fmt.Errorf("mimi_encoder: normalize latent: %w", err)
	}

	weight, err := e.speakerProjectionWeight()
	if err != nil {
		return nil, err
	}

	return projectSpeakerConditioning(normalizedLatent, weight)
}

func normalizeMimiEncoderLatent(latent *Tensor) (*Tensor, error) {
	shape := latent.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("expected 3D latent, got %v", shape)
	}

	if shape[0] != 1 {
		return nil, fmt.Errorf("expected latent batch dim=1, got %d", shape[0])
	}

	data, err := ExtractFloat32(latent)
	if err != nil {
		return nil, fmt.Errorf("extract latent: %w", err)
	}

	if shape[2] == mimiEncoderLatentDim {
		// Already [1, T, 512].
		return NewTensor(data, []int64{1, shape[1], mimiEncoderLatentDim})
	}

	if shape[1] == mimiEncoderLatentDim {
		// [1, 512, T] -> [1, T, 512].
		T := int(shape[2])
		transposed := make([]float32, len(data))

		for t := range T {
			for c := range mimiEncoderLatentDim {
				src := c*T + t
				dst := t*mimiEncoderLatentDim + c
				transposed[dst] = data[src]
			}
		}

		return NewTensor(transposed, []int64{1, shape[2], mimiEncoderLatentDim})
	}

	return nil, fmt.Errorf("unexpected latent shape %v (need [1,T,512] or [1,512,T])", shape)
}

func projectSpeakerConditioning(latent *Tensor, weight []float32) (*Tensor, error) {
	shape := latent.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[2] != mimiEncoderLatentDim {
		return nil, fmt.Errorf("latent shape must be [1,T,%d], got %v", mimiEncoderLatentDim, shape)
	}

	if len(weight) != VoiceEmbeddingDim*mimiEncoderLatentDim {
		return nil, fmt.Errorf(
			"speaker projection weight has %d values, expected %d",
			len(weight),
			VoiceEmbeddingDim*mimiEncoderLatentDim,
		)
	}

	latentData, err := ExtractFloat32(latent)
	if err != nil {
		return nil, fmt.Errorf("extract latent data: %w", err)
	}

	T := int(shape[1])
	out := make([]float32, T*VoiceEmbeddingDim)

	for t := range T {
		latRow := latentData[t*mimiEncoderLatentDim : (t+1)*mimiEncoderLatentDim]

		outRow := out[t*VoiceEmbeddingDim : (t+1)*VoiceEmbeddingDim]
		for outIdx := range VoiceEmbeddingDim {
			wRow := weight[outIdx*mimiEncoderLatentDim : (outIdx+1)*mimiEncoderLatentDim]

			var sum float32
			for i := range mimiEncoderLatentDim {
				sum += latRow[i] * wRow[i]
			}

			outRow[outIdx] = sum
		}
	}

	return NewTensor(out, []int64{1, int64(T), VoiceEmbeddingDim})
}

func (e *Engine) speakerProjectionWeight() ([]float32, error) {
	if len(e.speakerProjWeight) != 0 {
		return append([]float32(nil), e.speakerProjWeight...), nil
	}

	e.speakerProjOnce.Do(func() {
		modelPath, err := e.resolveModelWeightsPath()
		if err != nil {
			e.speakerProjErr = err
			return
		}

		store, err := safetensors.OpenStore(modelPath, safetensors.StoreOptions{
			KeyMapper: func(name string) (string, bool) {
				if name == "condition_provider.conditioners.speaker_wavs.output_proj.weight" ||
					name == "flow_lm.speaker_proj_weight" {
					return "speaker_proj_weight", true
				}

				return name, true
			},
		})
		if err != nil {
			e.speakerProjErr = fmt.Errorf("open model safetensors %q: %w", modelPath, err)
			return
		}
		defer store.Close()

		tensor, err := store.TensorWithShape("speaker_proj_weight", []int64{VoiceEmbeddingDim, mimiEncoderLatentDim})
		if err != nil {
			e.speakerProjErr = fmt.Errorf("load speaker_proj_weight from %q: %w", modelPath, err)
			return
		}

		e.speakerProjWeight = append([]float32(nil), tensor.Data...)
	})

	if e.speakerProjErr != nil {
		return nil, e.speakerProjErr
	}

	return append([]float32(nil), e.speakerProjWeight...), nil
}

func (e *Engine) resolveModelWeightsPath() (string, error) {
	if p := strings.TrimSpace(e.modelWeightsPath); p != "" {
		// #nosec G703 -- Path is an explicit local file path provided by config; this only validates readability.
		_, err := os.Stat(p)
		if err != nil {
			return "", fmt.Errorf("model safetensors path %q is not readable: %w", p, err)
		}

		return p, nil
	}

	if p := strings.TrimSpace(os.Getenv("POCKETTTS_MODEL_SAFETENSORS")); p != "" {
		_, err := os.Stat(p)
		if err == nil {
			return p, nil
		}
	}

	manifestDir := filepath.Dir(e.manifestPath)
	candidates := []string{
		filepath.Join(manifestDir, "..", "tts_b6369a24.safetensors"),
		filepath.Join(manifestDir, "..", "model.safetensors"),
		filepath.Join(manifestDir, "..", "tts.safetensors"),
		"models/tts_b6369a24.safetensors",
		"models/model.safetensors",
	}

	for _, c := range candidates {
		clean := filepath.Clean(c)

		_, err := os.Stat(clean)
		if err == nil {
			return clean, nil
		}
	}

	return "", errors.New("speaker projection weights not found; set --model-safetensors, RunnerConfig.ModelWeightsPath, or POCKETTTS_MODEL_SAFETENSORS")
}

func loadVoiceAudioSamples(audioPath string) ([]float32, error) {
	if strings.TrimSpace(audioPath) == "" {
		return nil, errors.New("encode voice: audio path must not be empty")
	}

	data, err := os.ReadFile(audioPath)
	if err != nil {
		return nil, fmt.Errorf("encode voice: read audio file %q: %w", audioPath, err)
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("encode voice: audio file %q is empty", audioPath)
	}

	ext := strings.ToLower(filepath.Ext(audioPath))
	if ext == ".wav" {
		samples, err := audio.DecodeWAV(data)
		if err != nil {
			return nil, fmt.Errorf("encode voice: decode WAV %q: %w", audioPath, err)
		}

		return samples, nil
	}

	samples, err := decodePCM16LE(data)
	if err != nil {
		return nil, fmt.Errorf("encode voice: decode raw PCM16 %q: %w", audioPath, err)
	}

	return samples, nil
}

func decodePCM16LE(data []byte) ([]float32, error) {
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("byte length %d is not a multiple of 2", len(data))
	}

	if len(data) == 0 {
		return nil, errors.New("empty PCM buffer")
	}

	out := make([]float32, len(data)/2)
	for i := range out {
		lo := int16(data[i*2])
		hi := int16(data[i*2+1]) << 8
		pcm := hi | lo
		out[i] = float32(pcm) / 32768.0
	}

	return out, nil
}
