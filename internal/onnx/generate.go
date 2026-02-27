package onnx

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
)

// GenerateConfig holds parameters for the autoregressive generation loop.
type GenerateConfig struct {
	Temperature    float64 // noise scale for flow sampling (default 0.7)
	EOSThreshold   float64 // raw logit threshold for EOS detection (default -4.0)
	MaxSteps       int     // maximum AR steps before forced stop (default 256)
	LSDDecodeSteps int     // Euler integration steps per frame (default 1)
	FramesAfterEOS int     // extra frames to generate after first EOS
	VoiceEmbedding *Tensor // optional voice conditioning [1, T_voice, D]; prepended to text_embeddings
}

// GenerateAudio runs the full TTS generation pipeline:
//
//	text_conditioner → AR loop → latent_to_mimi → mimi_decoder
//
// It dispatches to the stateful path (prefill + per-step KV-cache) when the
// flow_lm_prefill graph is present in the manifest, and falls back to the
// legacy stateless path for older ONNX bundles.
//
// Returns 24 kHz float32 PCM audio samples.
func (e *Engine) GenerateAudio(ctx context.Context, tokens []int64, cfg GenerateConfig) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, errors.New("generate: token slice must not be empty")
	}

	if _, ok := e.runners["flow_lm_prefill"]; ok {
		return e.generateAudioStateful(ctx, tokens, cfg)
	}

	return e.generateAudioStateless(ctx, tokens, cfg)
}

// generateAudioStateful uses the flow_lm_prefill + flow_lm_step graphs with an
// explicit KV-cache maintained across AR steps. This is the correct incremental
// pattern and produces clear audio for all text lengths.
func (e *Engine) generateAudioStateful(ctx context.Context, tokens []int64, cfg GenerateConfig) ([]float32, error) {
	// Step 1: Text conditioning.
	textEmb, err := e.TextConditioner(ctx, tokens)
	if err != nil {
		return nil, fmt.Errorf("generate: %w", err)
	}

	// Optional: prepend voice embedding.
	if cfg.VoiceEmbedding != nil {
		textEmb, err = ConcatTensorsDim1(cfg.VoiceEmbedding, textEmb)
		if err != nil {
			return nil, fmt.Errorf("generate: prepend voice embedding: %w", err)
		}

		slog.Debug("voice conditioning applied", "voice_frames", cfg.VoiceEmbedding.Shape()[1], "total_frames", textEmb.Shape()[1])
	}

	// Step 2: Prefill — run text through transformer, build KV-cache once.
	kvState, err := e.FlowLMPrefill(ctx, textEmb)
	if err != nil {
		return nil, fmt.Errorf("generate: prefill: %w", err)
	}

	// Step 3: Autoregressive generation loop.
	currentFrame := NewBOSSequence() // [1, 1, 32] NaN sentinel for first step
	var latentFrames []*Tensor
	var eosCountdown *int

	for step := range cfg.MaxSteps {
		lastHidden, eosLogits, err := e.FlowLMStepStateful(ctx, currentFrame, kvState)
		if err != nil {
			return nil, fmt.Errorf("generate step %d: %w", step, err)
		}

		if EOSDetected(eosLogits, cfg.EOSThreshold) && eosCountdown == nil {
			countdown := cfg.FramesAfterEOS
			eosCountdown = &countdown
			slog.Debug("EOS detected", "step", step, "frames_after_eos", countdown)
		}

		frame, err := e.FlowLMFlow(ctx, lastHidden, cfg.Temperature, cfg.LSDDecodeSteps)
		if err != nil {
			return nil, fmt.Errorf("generate step %d flow: %w", step, err)
		}

		latentFrames = append(latentFrames, frame)
		currentFrame = frame

		if eosCountdown != nil {
			if *eosCountdown == 0 {
				break
			}

			*eosCountdown--
		}
	}

	slog.Info("generation complete (stateful)", "frames", len(latentFrames))

	return e.decodeLatentsToAudio(ctx, latentFrames)
}

// generateAudioStateless is the legacy stateless AR loop that re-processes the
// full growing sequence on each step. Used as fallback for ONNX bundles that
// lack the flow_lm_prefill graph.
//
// Known limitation: produces garbled audio at the beginning for longer text
// inputs because the KV-cache is reset on every call to flow_lm_main.
func (e *Engine) generateAudioStateless(ctx context.Context, tokens []int64, cfg GenerateConfig) ([]float32, error) {
	// Step 1: Text conditioning.
	textEmb, err := e.TextConditioner(ctx, tokens)
	if err != nil {
		return nil, fmt.Errorf("generate: %w", err)
	}

	if cfg.VoiceEmbedding != nil {
		textEmb, err = ConcatTensorsDim1(cfg.VoiceEmbedding, textEmb)
		if err != nil {
			return nil, fmt.Errorf("generate: prepend voice embedding: %w", err)
		}

		slog.Debug("voice conditioning applied", "voice_frames", cfg.VoiceEmbedding.Shape()[1], "total_frames", textEmb.Shape()[1])
	}

	// Step 2: Autoregressive generation loop (stateless — full sequence each step).
	sequence := NewBOSSequence()
	var latentFrames []*Tensor
	var eosCountdown *int

	for step := range cfg.MaxSteps {
		lastHidden, eosLogits, err := e.FlowLMStep(ctx, sequence, textEmb)
		if err != nil {
			return nil, fmt.Errorf("generate step %d: %w", step, err)
		}

		if EOSDetected(eosLogits, cfg.EOSThreshold) && eosCountdown == nil {
			countdown := cfg.FramesAfterEOS
			eosCountdown = &countdown
			slog.Debug("EOS detected", "step", step, "frames_after_eos", countdown)
		}

		frame, err := e.FlowLMFlow(ctx, lastHidden, cfg.Temperature, cfg.LSDDecodeSteps)
		if err != nil {
			return nil, fmt.Errorf("generate step %d flow: %w", step, err)
		}

		latentFrames = append(latentFrames, frame)

		if eosCountdown != nil {
			if *eosCountdown == 0 {
				break
			}

			*eosCountdown--
		}

		sequence, err = AppendLatentFrame(sequence, frame)
		if err != nil {
			return nil, fmt.Errorf("generate step %d append: %w", step, err)
		}
	}

	slog.Info("generation complete (stateless)", "frames", len(latentFrames))

	return e.decodeLatentsToAudio(ctx, latentFrames)
}

// decodeLatentsToAudio stacks latent frames and runs LatentToMimi + MimiDecode.
func (e *Engine) decodeLatentsToAudio(ctx context.Context, latentFrames []*Tensor) ([]float32, error) {
	latent, err := StackLatentFrames(latentFrames)
	if err != nil {
		return nil, fmt.Errorf("generate: stack latents: %w", err)
	}

	mimiLatent, err := e.LatentToMimi(ctx, latent)
	if err != nil {
		return nil, fmt.Errorf("generate: %w", err)
	}

	pcm, err := e.MimiDecode(ctx, mimiLatent)
	if err != nil {
		return nil, fmt.Errorf("generate: %w", err)
	}

	return pcm, nil
}
