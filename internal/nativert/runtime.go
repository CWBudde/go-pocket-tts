package nativert

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/audio"
	"github.com/cwbudde/go-pocket-tts/internal/genloop"
	nativemodel "github.com/cwbudde/go-pocket-tts/internal/native"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
	"github.com/cwbudde/go-pocket-tts/internal/text"
)

const nativeLatentDim = 32

// VoiceEmbedding is a runtime-neutral voice conditioning tensor payload.
// Shape is expected to be [1, T, D] when present.
type VoiceEmbedding struct {
	Data  []float32
	Shape []int64
}

// Config controls a single chunk generation call.
type Config struct {
	Temperature        float64
	EOSThreshold       float64
	MaxSteps           int
	EstimatedMaxSteps  int
	SamplerDecodeSteps int
	FramesAfterEOS     int
	MimiStepsPerLatent int
	MimiSequenceLength int
	VoiceEmbedding     *VoiceEmbedding
	VoiceModelState    *safetensors.VoiceModelState
	// Rand drives sampling when set, so a caller that seeds it gets
	// reproducible audio; it must not be shared between concurrent calls.
	// Nil uses the runtime's own time-seeded source.
	Rand *rand.Rand
	// StepCallback is called after each AR step with the 1-based step index
	// and the configured maxSteps ceiling. It may be nil.
	StepCallback func(step, maxSteps int)
}

// latentSampler has the signature of Model.SampleNextLatentStateful.
type latentSampler func(
	state *nativemodel.FlowLMState,
	sequenceFrame *tensor.Tensor,
	decodeSteps int,
	eosThreshold, temperature float32,
	rng *rand.Rand,
) (*tensor.Tensor, bool, error)

// Runtime generates audio with a preloaded native model.
type Runtime struct {
	model *nativemodel.Model
	// sample replaces model.SampleNextLatentStateful in tests when set.
	sample latentSampler

	rngMu sync.Mutex
	rng   *rand.Rand
}

// New creates the pure-Go safetensors runtime for a preloaded native model.
func New(model *nativemodel.Model) *Runtime {
	return &Runtime{
		model: model,
		rng:   rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// SampleRate is the rate of the PCM GenerateAudio returns.
func (r *Runtime) SampleRate() int {
	return int(r.model.Mimi().SampleRate())
}

func (r *Runtime) MimiTiming() (float64, float64, int) {
	if r == nil || r.model == nil || r.model.Mimi() == nil {
		cfg := nativemodel.DefaultMimiConfig()
		return cfg.FrameRate, cfg.EncoderFrameRate, cfg.MimiStepsPerLatent()
	}

	mimi := r.model.Mimi()

	return mimi.FrameRate(), mimi.EncoderFrameRate(), mimi.MimiStepsPerLatent()
}

func (r *Runtime) GenerateAudio(ctx context.Context, tokens []int64, cfg Config) ([]float32, error) {
	if r == nil || r.model == nil {
		return nil, errors.New("native-safetensors runtime unavailable")
	}

	if len(tokens) == 0 {
		return nil, errors.New("generate: token slice must not be empty")
	}

	maxSteps := resolveMaxSteps(cfg, len(tokens))
	decodeSteps := resolveDecodeSteps(cfg)

	overallStart := time.Now()
	stageStart := overallStart

	slog.Debug(
		"native-safetensors generation start",
		"tokens", len(tokens),
		"max_steps", maxSteps,
		"estimated_max_steps", cfg.EstimatedMaxSteps,
		"mimi_steps_per_latent", cfg.MimiStepsPerLatent,
		"mimi_sequence_length", cfg.MimiSequenceLength,
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

	flowState, err := r.prepareFlowState(textEmb, cfg)
	if err != nil {
		return nil, err
	}

	sequenceFrame, err := newBOSSequenceTensor()
	if err != nil {
		return nil, fmt.Errorf("generate: build bos sequence: %w", err)
	}

	stageStart = time.Now()

	latentFrames, err := r.runARLoop(ctx, flowState, sequenceFrame, maxSteps, decodeSteps, cfg)
	if err != nil {
		return nil, err
	}

	slog.Debug("native-safetensors AR loop complete", "ms", time.Since(stageStart).Milliseconds(), "frames", len(latentFrames))

	stageStart = time.Now()

	audio3D, err := r.decodeLatents(latentFrames)
	if err != nil {
		return nil, err
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

	pcm := append([]float32(nil), audio3D.RawData()...)
	audio.ChunkFadeIn(pcm, int(r.model.Mimi().SampleRate()))

	return pcm, nil
}

func (r *Runtime) Close() {
	if r != nil && r.model != nil {
		r.model.Close()
	}
}

// CheckVoice primes a fresh flow state with the voice conditioning alone
// (an embedding or a model state, at most one set), the voice half of
// prepareFlowState, and reports why the model rejects it.
func (r *Runtime) CheckVoice(embedding *VoiceEmbedding, modelState *safetensors.VoiceModelState) error {
	if modelState != nil {
		_, err := r.model.NewFlowStateFromVoiceModelState(modelState)
		return err
	}

	if embedding == nil {
		return nil
	}

	voiceEmb, err := tensor.New(embedding.Data, embedding.Shape)
	if err != nil {
		return fmt.Errorf("build voice tensor: %w", err)
	}

	voiceEmb, err = r.model.VoicePrompt(voiceEmb)
	if err != nil {
		return err
	}

	state, err := r.model.NewFlowState()
	if err != nil {
		return err
	}

	return r.model.PromptFlow(state, voiceEmb)
}

// prepareFlowState applies voice conditioning to the text embeddings and
// returns the flow state primed with the conditioning sequence.
func (r *Runtime) prepareFlowState(textEmb *tensor.Tensor, cfg Config) (*nativemodel.FlowLMState, error) {
	if cfg.VoiceEmbedding != nil && cfg.VoiceModelState != nil {
		return nil, errors.New("generate: voice embedding and voice model state are mutually exclusive")
	}

	if cfg.VoiceEmbedding != nil {
		voiceEmb, err := tensor.New(cfg.VoiceEmbedding.Data, cfg.VoiceEmbedding.Shape)
		if err != nil {
			return nil, fmt.Errorf("generate: build voice tensor: %w", err)
		}

		voiceEmb, err = r.model.VoicePrompt(voiceEmb)
		if err != nil {
			return nil, fmt.Errorf("generate: voice prompt: %w", err)
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

	stageStart := time.Now()

	var (
		flowState *nativemodel.FlowLMState
		err       error
	)

	if cfg.VoiceModelState != nil {
		flowState, err = r.model.NewFlowStateFromVoiceModelState(cfg.VoiceModelState)
		if err != nil {
			return nil, fmt.Errorf("generate: load voice model state: %w", err)
		}

		slog.Debug("voice model-state conditioning applied")
	} else {
		flowState, err = r.model.NewFlowState()
		if err != nil {
			return nil, fmt.Errorf("generate: init flow state: %w", err)
		}
	}

	err = r.model.PromptFlow(flowState, textEmb)
	if err != nil {
		return nil, fmt.Errorf("generate: prompt flow state: %w", err)
	}

	slog.Debug("native-safetensors flow prompt complete", "ms", time.Since(stageStart).Milliseconds())

	return flowState, nil
}

// runARLoop runs the autoregressive latent sampling loop until the
// genloop.EOSStop rule ends it (eos_step + FramesAfterEOS frames) or maxSteps
// is reached.
func (r *Runtime) runARLoop(
	ctx context.Context,
	flowState *nativemodel.FlowLMState,
	sequenceFrame *tensor.Tensor,
	maxSteps, decodeSteps int,
	cfg Config,
) ([]*tensor.Tensor, error) {
	var latentFrames []*tensor.Tensor

	sample := r.sample
	if sample == nil {
		sample = r.model.SampleNextLatentStateful
	}

	stop := genloop.EOSStop{FramesAfter: cfg.FramesAfterEOS}

	for step := range maxSteps {
		err := ctx.Err()
		if err != nil {
			return nil, err
		}

		frame, isEOS, err := r.sampleStep(sample, flowState, sequenceFrame, decodeSteps, cfg)
		if err != nil {
			return nil, fmt.Errorf("generate step %d: %w", step, err)
		}

		if stop.Stop(step, isEOS) {
			eosStep, _ := stop.EOSStep()
			slog.Debug("EOS stop", "eos_step", eosStep, "frames_after_eos", cfg.FramesAfterEOS)

			break
		}

		latentFrames = append(latentFrames, frame)
		sequenceFrame = frame

		if cfg.StepCallback != nil {
			cfg.StepCallback(step+1, maxSteps)
		}

		if step > 0 && step%10 == 0 {
			slog.Debug("native-safetensors generation progress", "step", step, "frames", len(latentFrames))
		}
	}

	return latentFrames, nil
}

// sampleStep samples one latent frame with cfg.Rand, or with the shared
// runtime source under its lock when the caller brought none.
func (r *Runtime) sampleStep(
	sample latentSampler,
	flowState *nativemodel.FlowLMState,
	sequenceFrame *tensor.Tensor,
	decodeSteps int,
	cfg Config,
) (*tensor.Tensor, bool, error) {
	eos, temperature := float32(cfg.EOSThreshold), float32(cfg.Temperature)
	if cfg.Rand != nil {
		return sample(flowState, sequenceFrame, decodeSteps, eos, temperature, cfg.Rand)
	}

	r.rngMu.Lock()
	defer r.rngMu.Unlock()

	return sample(flowState, sequenceFrame, decodeSteps, eos, temperature, r.rng)
}

func resolveMaxSteps(cfg Config, tokenCount int) int {
	if cfg.MaxSteps > 0 {
		return cfg.MaxSteps
	}

	if cfg.EstimatedMaxSteps > 0 {
		return cfg.EstimatedMaxSteps
	}

	return text.EstimateMaxFrames(tokenCount, text.DefaultMimiFrameRate)
}

func resolveDecodeSteps(cfg Config) int {
	if cfg.SamplerDecodeSteps > 0 {
		return cfg.SamplerDecodeSteps
	}

	return 1
}

// decodeLatents stacks the latent frames and decodes them to a [1,1,N] audio tensor.
func (r *Runtime) decodeLatents(latentFrames []*tensor.Tensor) (*tensor.Tensor, error) {
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

	return audio3D, nil
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
