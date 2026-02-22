package onnx

import (
	"context"
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
//	text_conditioner → AR loop (flow_lm_main + flow_lm_flow) → latent_to_mimi → mimi_decoder
//
// Returns 24 kHz float32 PCM audio samples.
func (e *Engine) GenerateAudio(ctx context.Context, tokens []int64, cfg GenerateConfig) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("generate: token slice must not be empty")
	}

	// Step 1: Text conditioning.
	textEmb, err := e.TextConditioner(ctx, tokens)
	if err != nil {
		return nil, fmt.Errorf("generate: %w", err)
	}

	// Optional: prepend voice embedding to text embeddings.
	if cfg.VoiceEmbedding != nil {
		textEmb, err = ConcatTensorsDim1(cfg.VoiceEmbedding, textEmb)
		if err != nil {
			return nil, fmt.Errorf("generate: prepend voice embedding: %w", err)
		}
		slog.Debug("voice conditioning applied", "voice_frames", cfg.VoiceEmbedding.Shape()[1], "total_frames", textEmb.Shape()[1])
	}

	// Step 2: Autoregressive generation loop.
	sequence := NewBOSSequence()
	var latentFrames []*Tensor
	var eosCountdown *int

	for step := range cfg.MaxSteps {
		// Run flow_lm_main.
		lastHidden, eosLogits, err := e.FlowLMStep(ctx, sequence, textEmb)
		if err != nil {
			return nil, fmt.Errorf("generate step %d: %w", step, err)
		}

		// EOS detection with countdown.
		if EOSDetected(eosLogits, cfg.EOSThreshold) && eosCountdown == nil {
			countdown := cfg.FramesAfterEOS
			eosCountdown = &countdown
			slog.Debug("EOS detected", "step", step, "frames_after_eos", countdown)
		}

		// Flow decode: last_hidden → latent frame [1, 1, 32].
		frame, err := e.FlowLMFlow(ctx, lastHidden, cfg.Temperature, cfg.LSDDecodeSteps)
		if err != nil {
			return nil, fmt.Errorf("generate step %d flow: %w", step, err)
		}
		latentFrames = append(latentFrames, frame)

		// Check countdown expiry.
		if eosCountdown != nil {
			if *eosCountdown == 0 {
				break
			}
			*eosCountdown--
		}

		// Grow sequence for next step.
		sequence, err = AppendLatentFrame(sequence, frame)
		if err != nil {
			return nil, fmt.Errorf("generate step %d append: %w", step, err)
		}
	}

	slog.Info("generation complete", "frames", len(latentFrames))

	// Step 3: Stack latent frames and decode to audio.
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
