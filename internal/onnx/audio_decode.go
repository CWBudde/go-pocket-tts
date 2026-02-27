package onnx

import (
	"context"
	"errors"
	"fmt"
)

// StackLatentFrames concatenates latent frames [1, 1, 32] into [1, T, 32].
func StackLatentFrames(frames []*Tensor) (*Tensor, error) {
	if len(frames) == 0 {
		return nil, errors.New("no latent frames to stack")
	}

	var combined []float32

	for i, f := range frames {
		data, err := ExtractFloat32(f)
		if err != nil {
			return nil, fmt.Errorf("extract frame %d: %w", i, err)
		}

		combined = append(combined, data...)
	}

	T := int64(len(frames))

	return NewTensor(combined, []int64{1, T, int64(latentDim)})
}

// LatentToMimi runs the latent_to_mimi ONNX graph.
//
// Input: latent [1, T_frames, 32]
// Output: mimi_latent [1, 512, T_frames].
func (e *Engine) LatentToMimi(ctx context.Context, latent *Tensor) (*Tensor, error) {
	runner, ok := e.runners["latent_to_mimi"]
	if !ok {
		return nil, errors.New("latent_to_mimi graph not found in manifest")
	}

	outputs, err := runner.Run(ctx, map[string]*Tensor{"latent": latent})
	if err != nil {
		return nil, fmt.Errorf("latent_to_mimi: run: %w", err)
	}

	mimiLatent, ok := outputs["mimi_latent"]
	if !ok {
		return nil, errors.New("latent_to_mimi: missing 'mimi_latent' in output")
	}

	return mimiLatent, nil
}

// MimiDecode runs the mimi_decoder ONNX graph and returns PCM audio samples.
//
// Input: mimiLatent [1, 512, T_frames]
// Output: []float32 PCM samples (24 kHz).
func (e *Engine) MimiDecode(ctx context.Context, mimiLatent *Tensor) ([]float32, error) {
	runner, ok := e.runners["mimi_decoder"]
	if !ok {
		return nil, errors.New("mimi_decoder graph not found in manifest")
	}

	outputs, err := runner.Run(ctx, map[string]*Tensor{"latent": mimiLatent})
	if err != nil {
		return nil, fmt.Errorf("mimi_decoder: run: %w", err)
	}

	audio, ok := outputs["audio"]
	if !ok {
		return nil, errors.New("mimi_decoder: missing 'audio' in output")
	}

	pcm, err := ExtractFloat32(audio)
	if err != nil {
		return nil, fmt.Errorf("mimi_decoder: extract audio: %w", err)
	}

	return pcm, nil
}
