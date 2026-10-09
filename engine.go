// Package pockettts synthesizes speech with the pure-Go PocketTTS runtime.
//
// The catalog (LoadCatalog) lists every supported model with pinned,
// checksummed downloads. Load builds an Engine from the model's weights and
// tokenizer bytes (e.g. fetched in a browser); LoadDir reads them from a
// directory laid out by the download package. Engine.Synthesize turns text
// into mono float32 PCM at SampleRate.
//
// The package depends on neither ONNX Runtime nor the network, so it builds
// for js/wasm.
package pockettts

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	nativemodel "github.com/cwbudde/go-pocket-tts/internal/native"
	"github.com/cwbudde/go-pocket-tts/internal/nativert"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/ops"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
	"github.com/cwbudde/go-pocket-tts/internal/text"
	"github.com/cwbudde/go-pocket-tts/internal/tokenizer"
)

// SampleRate is the rate of the PCM Synthesize returns.
const SampleRate = 24000

// maxTokensPerChunk is the SentencePiece token budget per synthesis chunk,
// matching the reference implementation's 50-token limit.
const maxTokensPerChunk = 50

// mimiStepsPerLatent is the Mimi decoder steps per generated latent frame.
const mimiStepsPerLatent = 16

// maxSamplerSteps bounds Options.SamplerSteps; more steps only cost time.
const maxSamplerSteps = 64

// Engine synthesizes speech with one loaded model. Its methods must not be
// called concurrently.
type Engine struct {
	name  string
	model *modelcfg.ModelConfig
	tok   tokenizer.Tokenizer
	rt    *nativert.Runtime
}

// Load builds the engine of the catalog model name from its weights and
// tokenizer (a tokenizer.model or tokenizer.json, sniffed from the bytes).
// The weights are decoded and not retained, so the caller may drop them.
func Load(name string, weights, tokenizerData []byte) (*Engine, error) {
	cfg, tok, err := lookup(name, tokenizerData)
	if err != nil {
		return nil, err
	}

	store, err := safetensors.OpenStoreFromBytes(weights, safetensors.StoreOptions{})
	if err != nil {
		return nil, fmt.Errorf("pockettts: open weights: %w", err)
	}
	defer store.Close()

	model, err := nativemodel.LoadModelFromStore(store, nativemodel.ConfigFor(cfg))
	if err != nil {
		return nil, fmt.Errorf("pockettts: load model: %w", err)
	}

	return &Engine{name: name, model: cfg, tok: tok, rt: nativert.New(model)}, nil
}

// LoadDir builds the engine of the catalog model name from the files below
// root at their catalog paths (the layout the download package writes).
func LoadDir(root, name string) (*Engine, error) {
	m, err := LookupModel(name)
	if err != nil {
		return nil, err
	}

	tokenizerData, err := readFile(root, m.Tokenizer)
	if err != nil {
		return nil, err
	}

	cfg, tok, err := lookup(name, tokenizerData)
	if err != nil {
		return nil, err
	}

	weights, err := localPath(root, m.Weights)
	if err != nil {
		return nil, err
	}

	model, err := nativemodel.LoadModelFromSafetensors(weights, nativemodel.ConfigFor(cfg))
	if err != nil {
		return nil, fmt.Errorf("pockettts: load model: %w", err)
	}

	return &Engine{name: name, model: cfg, tok: tok, rt: nativert.New(model)}, nil
}

func lookup(name string, tokenizerData []byte) (*modelcfg.ModelConfig, tokenizer.Tokenizer, error) {
	_, err := LookupModel(name)
	if err != nil {
		return nil, nil, err
	}

	cfg, err := modelcfg.Lookup(name)
	if err != nil {
		return nil, nil, fmt.Errorf("pockettts: model config: %w", err)
	}

	tok, err := tokenizer.LoadBytes(tokenizerData, cfg.FlowLM.LookupTable.NBins)
	if err != nil {
		return nil, nil, fmt.Errorf("pockettts: load tokenizer: %w", err)
	}

	return cfg, tok, nil
}

// Name is the catalog name of the loaded model.
func (e *Engine) Name() string {
	return e.name
}

// Close releases the model.
func (e *Engine) Close() {
	if e != nil && e.rt != nil {
		e.rt.Close()
	}
}

// Voice is a parsed voice: a voice embedding or a voice model state.
type Voice struct {
	embedding *nativert.VoiceEmbedding
	state     *safetensors.VoiceModelState
}

// ParseVoice parses a voice .safetensors file, either kind.
func ParseVoice(data []byte) (*Voice, error) {
	kind, err := safetensors.InspectVoiceFileBytes(data)
	if err != nil {
		return nil, fmt.Errorf("pockettts: inspect voice: %w", err)
	}

	if kind == safetensors.VoiceFileModelState {
		state, err := safetensors.LoadVoiceModelStateFromBytes(data)
		if err != nil {
			return nil, fmt.Errorf("pockettts: load voice model state: %w", err)
		}

		return &Voice{state: state}, nil
	}

	values, shape, err := safetensors.LoadVoiceEmbeddingFromBytes(data)
	if err != nil {
		return nil, fmt.Errorf("pockettts: load voice embedding: %w", err)
	}

	return &Voice{embedding: &nativert.VoiceEmbedding{Data: values, Shape: shape}}, nil
}

// LoadVoiceDir parses the predefined voice id of the catalog model name from
// below root.
func LoadVoiceDir(root, name, id string) (*Voice, error) {
	m, err := LookupModel(name)
	if err != nil {
		return nil, err
	}

	v, ok := m.Voice(id)
	if !ok {
		return nil, fmt.Errorf("pockettts: model %s has no voice %q", name, id)
	}

	data, err := readFile(root, v.File)
	if err != nil {
		return nil, err
	}

	return ParseVoice(data)
}

// Options controls one synthesis. Start from DefaultOptions.
type Options struct {
	// Temperature scales sampling noise; 0 is deterministic and flat.
	Temperature float64
	// EOSThreshold is the end-of-speech logit a chunk stops at; lower
	// values stop earlier.
	EOSThreshold float64
	// SamplerSteps is the flow decode steps per frame (1 to 64).
	SamplerSteps int
	// Seed makes the output reproducible for one build and platform.
	Seed uint64
	// Progress, when set, is called after each generated frame.
	Progress func(Progress)
}

// Progress reports synthesis progress: chunk is 1-based, step counts the
// frames of that chunk against its frame budget.
type Progress struct {
	Chunk, Chunks  int
	Step, MaxSteps int
}

// DefaultOptions returns the defaults of the loaded model.
func (e *Engine) DefaultOptions() Options {
	return Options{Temperature: e.model.DefaultTemperature, EOSThreshold: -4, SamplerSteps: 1}
}

func (o Options) validate() error {
	switch {
	case math.IsNaN(o.Temperature) || math.IsInf(o.Temperature, 0) || o.Temperature < 0 || o.Temperature > 2:
		return errors.New("temperature must be in [0, 2]")
	case math.IsNaN(o.EOSThreshold) || math.IsInf(o.EOSThreshold, 0):
		return errors.New("EOS threshold must be finite")
	case o.SamplerSteps < 1 || o.SamplerSteps > maxSamplerSteps:
		return fmt.Errorf("sampler steps must be in [1, %d]", maxSamplerSteps)
	}

	return nil
}

// Synthesize speaks text with voice (nil uses the model's unconditioned
// voice). The text is split into sentence chunks of at most 50 tokens that
// are generated one after another; ctx is checked before every frame.
func (e *Engine) Synthesize(ctx context.Context, input string, voice *Voice, opts Options) ([]float32, error) {
	err := opts.validate()
	if err != nil {
		return nil, fmt.Errorf("pockettts: %w", err)
	}

	normalized, err := text.Normalize(input)
	if err != nil {
		return nil, fmt.Errorf("pockettts: %w", err)
	}

	chunks, err := text.PrepareChunks(normalized, e.tok, maxTokensPerChunk, text.OptionsFor(e.model))
	if err != nil {
		return nil, fmt.Errorf("pockettts: prepare text: %w", err)
	}

	if len(chunks) == 0 {
		return nil, errors.New("pockettts: text produced no chunks")
	}

	rng := rand.New(rand.NewSource(int64(opts.Seed))) // #nosec G115 G404 -- the seed's bits select a reproducible sampling stream, not a secret.

	var pcm []float32

	for i, chunk := range chunks {
		maxSteps := text.EstimateMaxFrames(len(chunk.TokenIDs), text.DefaultMimiFrameRate)
		cfg := nativert.Config{
			Temperature:        opts.Temperature,
			EOSThreshold:       opts.EOSThreshold,
			MaxSteps:           maxSteps,
			EstimatedMaxSteps:  maxSteps,
			SamplerDecodeSteps: opts.SamplerSteps,
			FramesAfterEOS:     e.model.FramesAfterEOS(chunk.FramesAfterEOS()),
			MimiStepsPerLatent: mimiStepsPerLatent,
			MimiSequenceLength: maxSteps * mimiStepsPerLatent,
			Rand:               rng,
		}

		if voice != nil {
			cfg.VoiceEmbedding, cfg.VoiceModelState = voice.embedding, voice.state
		}

		if opts.Progress != nil {
			progress := Progress{Chunk: i + 1, Chunks: len(chunks), MaxSteps: maxSteps}
			opts.Progress(progress)

			cfg.StepCallback = func(step, _ int) {
				progress.Step = step
				opts.Progress(progress)
			}
		}

		audio, err := e.rt.GenerateAudio(ctx, chunk.TokenIDs, cfg)
		if err != nil {
			return nil, fmt.Errorf("pockettts: chunk %d of %d: %w", i+1, len(chunks), err)
		}

		pcm = append(pcm, audio...)
	}

	if len(pcm) == 0 {
		return nil, errors.New("pockettts: synthesis produced no samples")
	}

	return pcm, nil
}

// SetWorkers sets the goroutines the native kernels use (default 1). The
// summation order follows the worker count, so a seed reproduces its audio
// only at the same count. It is process-wide; call it before Load.
func SetWorkers(n int) {
	if n < 1 {
		n = 1
	}

	ops.SetConvWorkers(n)
	tensor.SetWorkers(n)
}

func localPath(root string, f File) (string, error) {
	p := filepath.Join(root, filepath.FromSlash(f.Path))

	info, err := os.Stat(p)
	if err != nil {
		return "", fmt.Errorf("pockettts: %w (download the model first)", err)
	}

	if info.Size() != f.Size {
		return "", fmt.Errorf("pockettts: %s has %d bytes, want %d (download the model again)", p, info.Size(), f.Size)
	}

	return p, nil
}

func readFile(root string, f File) ([]byte, error) {
	p, err := localPath(root, f)
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(p) // #nosec G304 -- the path is a catalog path below the caller's model root.
	if err != nil {
		return nil, fmt.Errorf("pockettts: %w", err)
	}

	return data, nil
}
