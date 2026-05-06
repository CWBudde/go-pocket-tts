package tts

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	nativemodel "github.com/cwbudde/go-pocket-tts/internal/native"
	"github.com/cwbudde/go-pocket-tts/internal/onnx"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/ops"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
	"github.com/cwbudde/go-pocket-tts/internal/text"
	"github.com/cwbudde/go-pocket-tts/internal/tokenizer"
)

// maxTokensPerChunk is the SentencePiece token budget per synthesis chunk,
// matching the reference implementation's 50-token limit.
const maxTokensPerChunk = 50

// Service orchestrates text preprocessing and ONNX inference.
type Service struct {
	runtime   Runtime
	tokenizer tokenizer.Tokenizer
	ttsCfg    config.TTSConfig
}

// NewService initializes the TTS service with the configured native runtime.
func NewService(cfg config.Config) (*Service, error) {
	tok, err := tokenizer.NewSentencePieceTokenizer(cfg.Paths.TokenizerModel)
	if err != nil {
		return nil, fmt.Errorf("init tokenizer: %w", err)
	}

	backend, err := config.NormalizeBackend(cfg.TTS.Backend)
	if err != nil {
		return nil, err
	}

	var rt Runtime

	switch backend {
	case config.BackendNative:
		ops.SetConvWorkers(cfg.Runtime.ConvWorkers)

		if w := cfg.Runtime.ConvWorkers; w > 1 {
			slog.Info("conv parallelism enabled", "workers", w)
		}

		tensorWorkers, source := resolveTensorWorkers(cfg.Runtime)
		tensor.SetWorkers(tensorWorkers)

		if tensorWorkers > 1 {
			slog.Info("tensor parallelism enabled", "workers", tensorWorkers, "source", source)
		}

		modelPath, err := resolveNativeModelPath(cfg)
		if err != nil {
			return nil, err
		}

		model, err := nativemodel.LoadModelFromSafetensors(modelPath, nativemodel.DefaultConfig())
		if err != nil {
			return nil, fmt.Errorf("init safetensors-native model: %w", err)
		}

		slog.Info("loaded safetensors model", "path", modelPath)

		rt = newNativeSafetensorsRuntime(model)

		slog.Info("created native runtime", "backend", config.BackendNative)
	case config.BackendNativeONNX:
		rcfg := onnx.RunnerConfig{
			LibraryPath: cfg.Runtime.ORTLibraryPath,
			APIVersion:  23,
		}
		if rcfg.LibraryPath == "" {
			info, err := onnx.DetectRuntime(cfg.Runtime)
			if err != nil {
				return nil, fmt.Errorf("detect ORT runtime: %w", err)
			}

			rcfg.LibraryPath = info.LibraryPath
		}

		engine, err := onnx.NewEngine(cfg.Paths.ONNXManifest, rcfg)
		if err != nil {
			return nil, fmt.Errorf("init onnx engine: %w", err)
		}

		rt = newONNXRuntime(engine)
	default:
		return nil, fmt.Errorf("unsupported tts service backend %q", backend)
	}

	return &Service{
		runtime:   rt,
		tokenizer: tok,
		ttsCfg:    cfg.TTS,
	}, nil
}

// Synthesize converts text to audio samples.
// Text is preprocessed and split into ≤50-token chunks per the reference
// implementation. Each chunk is generated independently and the resulting
// PCM audio is concatenated.
//
// If voicePath is non-empty, it should point to a .safetensors file containing
// a voice embedding. The embedding is loaded and prepended to the text
// conditioning for each chunk, enabling voice cloning.
func (s *Service) Synthesize(input string, voicePath string) ([]float32, error) {
	return s.SynthesizeCtx(context.Background(), input, voicePath)
}

// SynthesizeCtx is like Synthesize but accepts a context for cancellation and
// deadline propagation from the HTTP handler or CLI.
func (s *Service) SynthesizeCtx(ctx context.Context, input string, voicePath string) ([]float32, error) {
	chunks, err := text.PrepareChunks(input, s.tokenizer, maxTokensPerChunk)
	if err != nil {
		return nil, fmt.Errorf("no tokens produced from input: %w", err)
	}

	voiceConditioning, err := loadVoiceConditioning(voicePath)
	if err != nil {
		return nil, err
	}

	if s.runtime == nil {
		return nil, errors.New("tts runtime unavailable")
	}

	var allAudio []float32

	for i, chunk := range chunks {
		err := ctx.Err()
		if err != nil {
			return nil, err
		}

		cfg := s.generateConfig(chunk)
		voiceConditioning.applyTo(&cfg)

		pcm, err := s.runtime.GenerateAudio(ctx, chunk.TokenIDs, cfg)
		if err != nil {
			return nil, fmt.Errorf("chunk %d: %w", i, err)
		}

		allAudio = append(allAudio, pcm...)
	}

	return allAudio, nil
}

// SynthesizeStream produces audio incrementally, sending one PCMChunk per
// text chunk to out.  The channel is closed before the method returns.
// The caller should range over out in a separate goroutine.
func (s *Service) SynthesizeStream(ctx context.Context, input string, voicePath string, out chan<- PCMChunk) error {
	defer close(out)

	chunks, err := text.PrepareChunks(input, s.tokenizer, maxTokensPerChunk)
	if err != nil {
		return fmt.Errorf("no tokens produced from input: %w", err)
	}

	voiceConditioning, err := loadVoiceConditioning(voicePath)
	if err != nil {
		return err
	}

	if s.runtime == nil {
		return errors.New("tts runtime unavailable")
	}

	for i, chunk := range chunks {
		err := ctx.Err()
		if err != nil {
			return err
		}

		cfg := s.generateConfig(chunk)
		voiceConditioning.applyTo(&cfg)

		pcm, err := s.runtime.GenerateAudio(ctx, chunk.TokenIDs, cfg)
		if err != nil {
			return fmt.Errorf("chunk %d: %w", i, err)
		}

		select {
		case out <- PCMChunk{Samples: pcm, ChunkIndex: i, Final: i == len(chunks)-1}:
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return nil
}

type voiceConditioning struct {
	embedding  *VoiceEmbedding
	modelState *safetensors.VoiceModelState
}

func (v voiceConditioning) applyTo(cfg *RuntimeGenerateConfig) {
	if cfg == nil {
		return
	}

	cfg.VoiceEmbedding = v.embedding
	cfg.VoiceModelState = v.modelState
}

func loadVoiceConditioning(voicePath string) (voiceConditioning, error) {
	if strings.TrimSpace(voicePath) == "" {
		return voiceConditioning{}, nil
	}

	kind, err := safetensors.InspectVoiceFile(voicePath)
	if err != nil {
		return voiceConditioning{}, fmt.Errorf("inspect voice safetensors: %w", err)
	}

	if kind == safetensors.VoiceFileModelState {
		state, err := safetensors.LoadVoiceModelState(voicePath)
		if err != nil {
			return voiceConditioning{}, fmt.Errorf("load voice model state: %w", err)
		}

		return voiceConditioning{modelState: state}, nil
	}

	data, shape, err := safetensors.LoadVoiceEmbedding(voicePath)
	if err != nil {
		return voiceConditioning{}, fmt.Errorf("load voice embedding: %w", err)
	}

	return voiceConditioning{embedding: &VoiceEmbedding{
		Data:  data,
		Shape: shape,
	}}, nil
}

// Close releases engine resources.
func (s *Service) Close() {
	if s.runtime != nil {
		s.runtime.Close()
	}
}

// generateConfig builds a GenerateConfig from the stored TTS config, adding
// upstream per-chunk generation length and Mimi decode sizing estimates.
func (s *Service) generateConfig(chunk text.ChunkMetadata) RuntimeGenerateConfig {
	mimiFrameRate, _, mimiStepsPerLatent := mimiTimingForRuntime(s.runtime)
	estimatedMaxSteps := text.EstimateMaxFrames(chunk.NumTokens, mimiFrameRate)

	return RuntimeGenerateConfig{
		Temperature:        s.ttsCfg.Temperature,
		EOSThreshold:       s.ttsCfg.EOSThreshold,
		MaxSteps:           generationStepLimit(s.ttsCfg.MaxSteps, estimatedMaxSteps),
		EstimatedMaxSteps:  estimatedMaxSteps,
		LSDDecodeSteps:     s.ttsCfg.LSDDecodeSteps,
		FramesAfterEOS:     chunk.FramesAfterEOS(),
		MimiStepsPerLatent: mimiStepsPerLatent,
		MimiSequenceLength: estimatedMaxSteps * mimiStepsPerLatent,
	}
}

func generationStepLimit(configured, estimated int) int {
	defaultMax := config.DefaultConfig().TTS.MaxSteps
	if estimated > 0 && (configured <= 0 || configured == defaultMax) {
		return estimated
	}

	return configured
}

type mimiTimingRuntime interface {
	MimiTiming() (frameRate float64, encoderFrameRate float64, stepsPerLatent int)
}

func mimiTimingForRuntime(rt Runtime) (float64, float64, int) {
	if timed, ok := rt.(mimiTimingRuntime); ok {
		frameRate, encoderFrameRate, steps := timed.MimiTiming()
		if frameRate > 0 && encoderFrameRate > 0 && steps > 0 {
			return frameRate, encoderFrameRate, steps
		}
	}

	const (
		mimiFrameRate        = text.DefaultMimiFrameRate
		mimiEncoderFrameRate = 200.0
	)

	return mimiFrameRate, mimiEncoderFrameRate, int(mimiEncoderFrameRate / mimiFrameRate)
}

func resolveNativeModelPath(cfg config.Config) (string, error) {
	p := strings.TrimSpace(cfg.Paths.ModelPath)
	if p == "" {
		return "", errors.New("safetensors model path is empty; set --paths-model-path")
	}

	if !strings.HasSuffix(strings.ToLower(p), ".safetensors") {
		return "", fmt.Errorf("model path %q does not end in .safetensors; set --paths-model-path to a .safetensors checkpoint", p)
	}

	_, err := os.Stat(p)
	if err != nil {
		return "", fmt.Errorf("safetensors model not found at %q; run 'pockettts model download' or set --paths-model-path", p)
	}

	return p, nil
}

func resolveTensorWorkers(cfg config.RuntimeConfig) (int, string) {
	if cfg.Workers > 0 {
		return cfg.Workers, "runtime-workers"
	}

	if cfg.ConvWorkers > 0 {
		return cfg.ConvWorkers, "conv-workers"
	}

	return 1, "default"
}
