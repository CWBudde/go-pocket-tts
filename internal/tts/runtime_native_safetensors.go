package tts

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"sync"
	"time"

	nativemodel "github.com/example/go-pocket-tts/internal/native"
	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

const nativeLatentDim = 32

type nativeSafetensorsRuntime struct {
	model *nativemodel.Model

	rngMu sync.Mutex
	rng   *rand.Rand
}

func newNativeSafetensorsRuntime(model *nativemodel.Model) Runtime {
	return &nativeSafetensorsRuntime{
		model: model,
		rng:   rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// NewNativeSafetensorsRuntime creates the pure-Go safetensors runtime for a
// preloaded native model (used by js/wasm and tests that bootstrap from bytes).
func NewNativeSafetensorsRuntime(model *nativemodel.Model) Runtime {
	return newNativeSafetensorsRuntime(model)
}

func (r *nativeSafetensorsRuntime) GenerateAudio(ctx context.Context, tokens []int64, cfg RuntimeGenerateConfig) ([]float32, error) {
	if r == nil || r.model == nil {
		return nil, errors.New("native-safetensors runtime unavailable")
	}

	if len(tokens) == 0 {
		return nil, errors.New("generate: token slice must not be empty")
	}

	maxSteps := cfg.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 256
	}

	decodeSteps := cfg.LSDDecodeSteps
	if decodeSteps <= 0 {
		decodeSteps = 1
	}

	overallStart := time.Now()
	stageStart := overallStart

	slog.Debug(
		"native-safetensors generation start",
		"tokens", len(tokens),
		"max_steps", maxSteps,
		"lsd_steps", decodeSteps,
		"temperature", cfg.Temperature,
		"eos_threshold", cfg.EOSThreshold,
	)

	textEmb, err := r.model.TextEmbeddings(tokens)
	if err != nil {
		return nil, fmt.Errorf("generate: text embeddings: %w", err)
	}

	slog.Debug(
		"native-safetensors text conditioning ready",
		"ms", time.Since(stageStart).Milliseconds(),
		"text_frames", textEmb.Shape()[1],
	)

	if cfg.VoiceEmbedding != nil {
		voiceEmb, err := tensor.New(cfg.VoiceEmbedding.Data, cfg.VoiceEmbedding.Shape)
		if err != nil {
			return nil, fmt.Errorf("generate: build voice tensor: %w", err)
		}

		textEmb, err = tensor.Concat([]*tensor.Tensor{voiceEmb, textEmb}, 1)
		if err != nil {
			return nil, fmt.Errorf("generate: prepend voice embedding: %w", err)
		}

		shape := cfg.VoiceEmbedding.Shape
		if len(shape) >= 2 {
			slog.Debug("voice conditioning applied", "voice_frames", shape[1], "total_frames", textEmb.Shape()[1])
		}
	}

	stageStart = time.Now()

	flowState, err := r.model.NewFlowState()
	if err != nil {
		return nil, fmt.Errorf("generate: init flow state: %w", err)
	}

	if err := r.model.PromptFlow(flowState, textEmb); err != nil {
		return nil, fmt.Errorf("generate: prompt flow state: %w", err)
	}

	slog.Debug("native-safetensors flow prompt complete", "ms", time.Since(stageStart).Milliseconds())

	sequenceFrame, err := newBOSSequenceTensor()
	if err != nil {
		return nil, fmt.Errorf("generate: build bos sequence: %w", err)
	}

	var latentFrames []*tensor.Tensor
	var eosCountdown *int

	stageStart = time.Now()

	for step := range maxSteps {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		r.rngMu.Lock()
		frame, isEOS, err := r.model.SampleNextLatentStateful(
			flowState,
			sequenceFrame,
			decodeSteps,
			float32(cfg.EOSThreshold),
			float32(cfg.Temperature),
			r.rng,
		)
		r.rngMu.Unlock()

		if err != nil {
			return nil, fmt.Errorf("generate step %d: %w", step, err)
		}

		latentFrames = append(latentFrames, frame)

		if isEOS && eosCountdown == nil {
			countdown := cfg.FramesAfterEOS
			eosCountdown = &countdown
			slog.Debug("EOS detected", "step", step, "frames_after_eos", countdown)
		}

		if eosCountdown != nil {
			if *eosCountdown == 0 {
				break
			}

			(*eosCountdown)--
		}

		sequenceFrame = frame

		if step > 0 && step%10 == 0 {
			slog.Debug("native-safetensors generation progress", "step", step, "frames", len(latentFrames))
		}
	}

	slog.Debug("native-safetensors AR loop complete", "ms", time.Since(stageStart).Milliseconds(), "frames", len(latentFrames))

	stageStart = time.Now()

	latent, err := stackLatentFramesTensor(latentFrames)
	if err != nil {
		return nil, fmt.Errorf("generate: stack latents: %w", err)
	}

	mimiLatent, err := r.model.LatentToMimi(latent)
	if err != nil {
		return nil, fmt.Errorf("generate: latent_to_mimi: %w", err)
	}

	audio3D, err := r.model.MimiDecode(mimiLatent)
	if err != nil {
		return nil, fmt.Errorf("generate: mimi_decode: %w", err)
	}

	slog.Debug("native-safetensors decode complete", "ms", time.Since(stageStart).Milliseconds())

	shape := audio3D.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 {
		return nil, fmt.Errorf("generate: unexpected audio shape %v, want [1,1,N]", shape)
	}

	slog.Info(
		"generation complete",
		"backend", "native-safetensors",
		"frames", len(latentFrames),
		"samples", len(audio3D.RawData()),
		"duration_ms", time.Since(overallStart).Milliseconds(),
	)

	return append([]float32(nil), audio3D.RawData()...), nil
}

func (r *nativeSafetensorsRuntime) Close() {
	if r != nil && r.model != nil {
		r.model.Close()
	}
}

func newBOSSequenceTensor() (*tensor.Tensor, error) {
	data := make([]float32, nativeLatentDim)
	for i := range data {
		data[i] = float32(math.NaN())
	}

	return tensor.New(data, []int64{1, 1, nativeLatentDim})
}

func stackLatentFramesTensor(frames []*tensor.Tensor) (*tensor.Tensor, error) {
	if len(frames) == 0 {
		return nil, errors.New("no latent frames to stack")
	}

	return tensor.Concat(frames, 1)
}
