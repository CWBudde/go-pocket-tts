package main

import (
	"fmt"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	nativemodel "github.com/cwbudde/go-pocket-tts/internal/native"
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
}

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

	model, err := nativemodel.LoadModelFromStore(store, nativemodel.ConfigFor(mc))
	// The weights are decoded; drop the checkpoint bytes so they are not held
	// next to them (WASM memory is capped at 4 GB).
	store.Close()

	if err != nil {
		return nil, fmt.Errorf("load native model: %w", err)
	}

	return &nativeEngine{
		runtime:   tts.NewNativeSafetensorsRuntime(model),
		tokenizer: tok,
		name:      name,
		model:     mc,
	}, nil
}
