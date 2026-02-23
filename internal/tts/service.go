package tts

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/example/go-pocket-tts/internal/config"
	nativemodel "github.com/example/go-pocket-tts/internal/native"
	"github.com/example/go-pocket-tts/internal/onnx"
	"github.com/example/go-pocket-tts/internal/safetensors"
	"github.com/example/go-pocket-tts/internal/text"
	"github.com/example/go-pocket-tts/internal/tokenizer"
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
	case config.BackendNativeSafetensors:
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
		slog.Info("created native runtime", "backend", config.BackendNativeSafetensors)
	default:
		return nil, fmt.Errorf("unsupported tts service backend %q", backend)
	}
	return &Service{
		runtime:   rt,
		tokenizer: tok,
		ttsCfg:    cfg.TTS,
	}, nil
}

// generateConfig builds a GenerateConfig from the stored TTS config,
// overriding FramesAfterEOS per chunk.
func (s *Service) generateConfig(framesAfterEOS int) RuntimeGenerateConfig {
	return RuntimeGenerateConfig{
		Temperature:    s.ttsCfg.Temperature,
		EOSThreshold:   s.ttsCfg.EOSThreshold,
		MaxSteps:       s.ttsCfg.MaxSteps,
		LSDDecodeSteps: s.ttsCfg.LSDDecodeSteps,
		FramesAfterEOS: framesAfterEOS,
	}
}

// Synthesize converts text to audio samples via ONNX inference.
// Text is preprocessed and split into ≤50-token chunks per the reference
// implementation. Each chunk is generated independently and the resulting
// PCM audio is concatenated.
//
// If voicePath is non-empty, it should point to a .safetensors file containing
// a voice embedding. The embedding is loaded and prepended to the text
// conditioning for each chunk, enabling voice cloning.
func (s *Service) Synthesize(input string, voicePath string) ([]float32, error) {
	chunks, err := text.PrepareChunks(input, s.tokenizer, maxTokensPerChunk)
	if err != nil {
		return nil, fmt.Errorf("no tokens produced from input: %w", err)
	}

	// Load voice embedding if a path was provided.
	voiceEmb, err := loadVoiceEmbedding(voicePath)
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	var allAudio []float32
	if s.runtime == nil {
		return nil, fmt.Errorf("tts runtime unavailable")
	}

	for i, chunk := range chunks {
		cfg := s.generateConfig(chunk.FramesAfterEOS())
		cfg.VoiceEmbedding = voiceEmb

		pcm, err := s.runtime.GenerateAudio(ctx, chunk.TokenIDs, cfg)
		if err != nil {
			return nil, fmt.Errorf("chunk %d: %w", i, err)
		}
		allAudio = append(allAudio, pcm...)
	}

	return allAudio, nil
}

func loadVoiceEmbedding(voicePath string) (*VoiceEmbedding, error) {
	if strings.TrimSpace(voicePath) == "" {
		return nil, nil
	}
	data, shape, err := safetensors.LoadVoiceEmbedding(voicePath)
	if err != nil {
		return nil, fmt.Errorf("load voice embedding: %w", err)
	}
	return &VoiceEmbedding{
		Data:  data,
		Shape: shape,
	}, nil
}

// Close releases engine resources.
func (s *Service) Close() {
	if s.runtime != nil {
		s.runtime.Close()
	}
}

func resolveNativeModelPath(cfg config.Config) (string, error) {
	candidates := []string{}
	if p := strings.TrimSpace(cfg.Paths.ModelPath); strings.HasSuffix(strings.ToLower(p), ".safetensors") {
		candidates = append(candidates, p)
	}
	candidates = append(candidates, "models/tts_b6369a24.safetensors")

	for _, path := range candidates {
		if path == "" {
			continue
		}
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}

	return "", fmt.Errorf(
		"safetensors model not found; set --paths-model-path to a .safetensors checkpoint (tried: %v)",
		candidates,
	)
}
