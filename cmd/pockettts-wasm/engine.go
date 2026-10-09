package main

import (
	"errors"
	"fmt"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	nativemodel "github.com/cwbudde/go-pocket-tts/internal/native"
	"github.com/cwbudde/go-pocket-tts/internal/nativert"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
	"github.com/cwbudde/go-pocket-tts/internal/tokenizer"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
)

type nativeEngine struct {
	runtime   tts.Runtime
	tokenizer tokenizer.Tokenizer
	// name and model are the model config the page selected.
	name  string
	model *modelcfg.ModelConfig
	// voice clones voices from audio prompts; nil when the checkpoint's Mimi
	// encoder weights are zeroed (the ungated checkpoints).
	voice *nativemodel.VoiceEncoder
}

// loadModelFromStore decodes the FlowLM and the Mimi decoder; tests replace
// it to run newEngine on checkpoints holding only the voice encoder.
var loadModelFromStore = nativemodel.LoadModelFromStore

// emitFunc reports load progress; nil reports nothing.
type emitFunc func(stage string, current, total int, detail string)

// newEngine builds the engine of model config name ("" selects
// config.DefaultLanguage) from its safetensors checkpoint and tokenizer, a
// SentencePiece tokenizer.model or a tokenizer.json (sniffed from the bytes).
func newEngine(modelSafetensors, tokenizerBytes []byte, name string, emit emitFunc) (*nativeEngine, error) {
	if emit == nil {
		emit = func(string, int, int, string) {}
	}

	if name == "" {
		name = config.DefaultLanguage
	}

	mc, err := modelcfg.Lookup(name)
	if err != nil {
		return nil, fmt.Errorf("model config: %w", err)
	}

	emit("tokenizer", 5, 100, "loading tokenizer")

	tok, err := tokenizer.LoadBytes(tokenizerBytes, mc.FlowLM.LookupTable.NBins)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	emit("load", 20, 100, "opening safetensors checkpoint")

	store, err := safetensors.OpenStoreFromBytes(modelSafetensors, safetensors.StoreOptions{})
	if err != nil {
		return nil, fmt.Errorf("open model safetensors: %w", err)
	}

	emit("load", 50, 100, "building native model")

	cfg := nativemodel.ConfigFor(mc)

	model, err := loadModelFromStore(store, cfg)
	if err != nil {
		store.Close()

		return nil, fmt.Errorf("load native model: %w", err)
	}

	emit("load", 80, 100, "loading voice encoder")

	// The encoder is built from the same store, before it is closed. An
	// ungated checkpoint has zeroed encoder weights: it still synthesizes,
	// only cloning is unavailable.
	voice, err := nativemodel.LoadVoiceEncoder(nativemodel.NewVarBuilder(store), cfg.Mimi)
	// The weights are decoded; drop the checkpoint bytes so they are not held
	// next to them (WASM memory is capped at 4 GB).
	store.Close()

	if errors.Is(err, nativemodel.ErrMimiEncoderWeightsZeroed) {
		voice, err = nil, nil
	}

	if err != nil {
		return nil, fmt.Errorf("load voice encoder: %w", err)
	}

	return &nativeEngine{
		runtime:   nativert.New(model),
		tokenizer: tok,
		name:      name,
		model:     mc,
		voice:     voice,
	}, nil
}
