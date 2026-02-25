//go:build js && wasm

package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"math"
	"sync"
	"syscall/js"

	"github.com/example/go-pocket-tts/internal/audio"
	"github.com/example/go-pocket-tts/internal/config"
	nativemodel "github.com/example/go-pocket-tts/internal/native"
	"github.com/example/go-pocket-tts/internal/safetensors"
	"github.com/example/go-pocket-tts/internal/text"
	"github.com/example/go-pocket-tts/internal/tts"
)

const (
	maxTokensPerChunk = 50
)

type progressReporter struct {
	cb js.Value
}

func (p *progressReporter) Emit(stage string, current, total int, detail string) {
	if p == nil || p.cb.IsUndefined() || p.cb.IsNull() {
		return
	}
	percent := 0.0
	if total > 0 {
		percent = (float64(current) / float64(total)) * 100.0
		if percent < 0 {
			percent = 0
		}
		if percent > 100 {
			percent = 100
		}
	}
	payload := map[string]any{
		"stage":   stage,
		"current": current,
		"total":   total,
		"percent": percent,
		"detail":  detail,
	}
	defer func() {
		_ = recover()
	}()
	p.cb.Invoke(js.ValueOf(payload))
}

type synthesizeOptions struct {
	Temperature      float64
	EOSThreshold     float64
	MaxSteps         int
	LSDDecodeSteps   int
	VoiceSafetensors []byte
}

type preprocessTokenizer struct {
	prep *text.Preprocessor
}

func (t preprocessTokenizer) Encode(input string) ([]int64, error) {
	tokens := t.prep.Preprocess(input)
	out := make([]int64, len(tokens))
	for i, tok := range tokens {
		out[i] = int64(tok)
	}
	return out, nil
}

type nativeEngine struct {
	runtime tts.Runtime
}

var (
	defaults = config.DefaultConfig()
	engineMu sync.RWMutex
	engine   *nativeEngine
)

func main() {
	kernel := map[string]any{
		"version":    "0.4.0-wasm",
		"sampleRate": audio.ExpectedSampleRate,
		"loadModel":  js.FuncOf(loadModelAsync),
		"normalize":  js.FuncOf(normalizeText),
		"tokenize":   js.FuncOf(tokenizeText),
		"synthesize": js.FuncOf(synthesizeAsync),
	}

	js.Global().Set("PocketTTSKernel", js.ValueOf(kernel))
	println("PocketTTS wasm kernel loaded")
	select {}
}

func normalizeText(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errResult("missing text argument")
	}

	normalized, err := text.Normalize(args[0].String())
	if err != nil {
		return errResult(err.Error())
	}

	return okResult(map[string]any{"text": normalized})
}

func tokenizeText(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errResult("missing text argument")
	}

	normalized, err := text.Normalize(args[0].String())
	if err != nil {
		return errResult(err.Error())
	}

	tok := preprocessTokenizer{prep: text.NewPreprocessor()}
	chunks, err := text.PrepareChunks(normalized, tok, maxTokensPerChunk)
	if err != nil {
		return errResult(err.Error())
	}

	flat := make([]any, 0, len(chunks)*8)
	for _, c := range chunks {
		for _, id := range c.TokenIDs {
			flat = append(flat, id)
		}
	}

	return okResult(map[string]any{
		"text":   normalized,
		"tokens": flat,
		"chunks": len(chunks),
	})
}

func loadModelAsync(_ js.Value, args []js.Value) any {
	promiseCtor := js.Global().Get("Promise")
	var handler js.Func
	handler = js.FuncOf(func(_ js.Value, pArgs []js.Value) any {
		defer handler.Release()
		resolve := pArgs[0]
		reject := pArgs[1]

		if len(args) < 1 {
			reject.Invoke("missing model safetensors bytes argument")
			return nil
		}

		modelBytes, ok := copyJSBytes(args[0])
		if !ok || len(modelBytes) == 0 {
			reject.Invoke("model safetensors bytes must be a non-empty Uint8Array/ArrayBuffer")
			return nil
		}

		var progress progressReporter
		if len(args) > 1 && args[1].Type() == js.TypeFunction {
			progress.cb = args[1]
		}

		go func() {
			res, err := loadModel(modelBytes, &progress)
			if err != nil {
				reject.Invoke(err.Error())
				return
			}
			resolve.Invoke(js.ValueOf(res))
		}()

		return nil
	})

	return promiseCtor.New(handler)
}

func loadModel(modelSafetensors []byte, progress *progressReporter) (map[string]any, error) {
	progress.Emit("load", 5, 100, "opening safetensors checkpoint")
	store, err := safetensors.OpenStoreFromBytes(modelSafetensors, safetensors.StoreOptions{})
	if err != nil {
		return nil, fmt.Errorf("open model safetensors: %w", err)
	}

	progress.Emit("load", 35, 100, "building native model")
	model, err := nativemodel.LoadModelFromStore(store, nativemodel.DefaultConfig())
	if err != nil {
		store.Close()
		return nil, fmt.Errorf("load native model: %w", err)
	}
	runtime := tts.NewNativeSafetensorsRuntime(model)

	newEngine := &nativeEngine{runtime: runtime}
	engineMu.Lock()
	oldEngine := engine
	engine = newEngine
	engineMu.Unlock()

	if oldEngine != nil && oldEngine.runtime != nil {
		oldEngine.runtime.Close()
	}

	progress.Emit("load", 100, 100, "model ready")
	return okResult(map[string]any{
		"model_bytes": len(modelSafetensors),
	}), nil
}

func parseSynthOptions(args []js.Value) synthesizeOptions {
	opts := synthesizeOptions{
		Temperature:    defaults.TTS.Temperature,
		EOSThreshold:   defaults.TTS.EOSThreshold,
		MaxSteps:       defaults.TTS.MaxSteps,
		LSDDecodeSteps: defaults.TTS.LSDDecodeSteps,
	}

	if len(args) < 3 {
		return opts
	}
	optVal := args[2]
	if optVal.IsUndefined() || optVal.IsNull() {
		return opts
	}

	if v := optVal.Get("temperature"); !v.IsUndefined() && !v.IsNull() {
		temp := v.Float()
		if !math.IsNaN(temp) && !math.IsInf(temp, 0) && temp >= 0 {
			opts.Temperature = temp
		}
	}
	if v := optVal.Get("eosThreshold"); !v.IsUndefined() && !v.IsNull() {
		eos := v.Float()
		if !math.IsNaN(eos) && !math.IsInf(eos, 0) {
			opts.EOSThreshold = eos
		}
	}
	if v := optVal.Get("maxSteps"); !v.IsUndefined() && !v.IsNull() {
		steps := v.Int()
		if steps > 0 {
			opts.MaxSteps = steps
		}
	}
	if v := optVal.Get("lsdSteps"); !v.IsUndefined() && !v.IsNull() {
		steps := v.Int()
		if steps > 0 {
			opts.LSDDecodeSteps = steps
		}
	}

	if v := optVal.Get("voiceSafetensors"); !v.IsUndefined() && !v.IsNull() {
		if b, ok := copyJSBytes(v); ok {
			opts.VoiceSafetensors = b
		}
	}

	return opts
}

func synthesizeAsync(_ js.Value, args []js.Value) any {
	promiseCtor := js.Global().Get("Promise")
	var handler js.Func
	handler = js.FuncOf(func(_ js.Value, pArgs []js.Value) any {
		defer handler.Release()
		resolve := pArgs[0]
		reject := pArgs[1]

		textArg := ""
		var progress progressReporter
		if len(args) > 0 {
			textArg = args[0].String()
		}
		if len(args) > 1 && args[1].Type() == js.TypeFunction {
			progress.cb = args[1]
		}
		opts := parseSynthOptions(args)

		go func() {
			res, err := synthesize(textArg, &progress, opts)
			if err != nil {
				reject.Invoke(err.Error())
				return
			}
			resolve.Invoke(js.ValueOf(res))
		}()

		return nil
	})

	return promiseCtor.New(handler)
}

func synthesize(input string, progress *progressReporter, opts synthesizeOptions) (map[string]any, error) {
	engineMu.RLock()
	currentEngine := engine
	if currentEngine == nil || currentEngine.runtime == nil {
		engineMu.RUnlock()
		return nil, fmt.Errorf("model is not loaded; call loadModel first")
	}
	defer engineMu.RUnlock()

	progress.Emit("prepare", 0, 100, "normalizing and chunking input")
	normalized, err := text.Normalize(input)
	if err != nil {
		return nil, err
	}

	tok := preprocessTokenizer{prep: text.NewPreprocessor()}
	chunks, err := text.PrepareChunks(normalized, tok, maxTokensPerChunk)
	if err != nil {
		return nil, err
	}
	if len(chunks) == 0 {
		return nil, fmt.Errorf("no chunks produced")
	}
	progress.Emit("prepare", 10, 100, fmt.Sprintf("prepared %d chunks", len(chunks)))

	var voiceEmb *tts.VoiceEmbedding
	if len(opts.VoiceSafetensors) > 0 {
		vData, vShape, vErr := safetensors.LoadVoiceEmbeddingFromBytes(opts.VoiceSafetensors)
		if vErr != nil {
			return nil, fmt.Errorf("load voice embedding: %w", vErr)
		}
		voiceEmb = &tts.VoiceEmbedding{Data: vData, Shape: vShape}
		progress.Emit("voice", 15, 100, fmt.Sprintf("loaded voice embedding (%d frames)", vShape[1]))
	}

	allAudio := make([]float32, 0, audio.ExpectedSampleRate)
	totalTokens := 0
	for i, chunk := range chunks {
		pct := 20 + int((float64(i)/float64(len(chunks)))*70)
		progress.Emit("synthesize", pct, 100, fmt.Sprintf("chunk %d/%d", i+1, len(chunks)))

		cfg := tts.RuntimeGenerateConfig{
			Temperature:    opts.Temperature,
			EOSThreshold:   opts.EOSThreshold,
			MaxSteps:       opts.MaxSteps,
			LSDDecodeSteps: opts.LSDDecodeSteps,
			FramesAfterEOS: chunk.FramesAfterEOS(),
			VoiceEmbedding: voiceEmb,
		}

		pcm, genErr := currentEngine.runtime.GenerateAudio(context.Background(), chunk.TokenIDs, cfg)
		if genErr != nil {
			return nil, fmt.Errorf("chunk %d synthesis failed: %w", i+1, genErr)
		}
		allAudio = append(allAudio, pcm...)
		totalTokens += len(chunk.TokenIDs)
	}
	if len(allAudio) == 0 {
		return nil, fmt.Errorf("synthesis produced no samples")
	}

	progress.Emit("encode", 95, 100, "encoding WAV")
	wav, err := audio.EncodeWAV(allAudio)
	if err != nil {
		return nil, fmt.Errorf("encode wav: %w", err)
	}

	result := okResult(map[string]any{
		"text":         normalized,
		"token_count":  totalTokens,
		"chunk_count":  len(chunks),
		"sample_count": len(allAudio),
		"sample_rate":  audio.ExpectedSampleRate,
		"wav_base64":   base64.StdEncoding.EncodeToString(wav),
	})
	progress.Emit("done", 100, 100, "synthesis complete")
	return result, nil
}

func copyJSBytes(v js.Value) ([]byte, bool) {
	if v.IsUndefined() || v.IsNull() {
		return nil, false
	}

	uint8Array := js.Global().Get("Uint8Array")
	if !uint8Array.IsUndefined() && v.InstanceOf(uint8Array) {
		buf := make([]byte, v.Get("length").Int())
		n := js.CopyBytesToGo(buf, v)
		return buf[:n], true
	}

	arrayBuffer := js.Global().Get("ArrayBuffer")
	if !arrayBuffer.IsUndefined() && v.InstanceOf(arrayBuffer) {
		wrapped := uint8Array.New(v)
		buf := make([]byte, wrapped.Get("length").Int())
		n := js.CopyBytesToGo(buf, wrapped)
		return buf[:n], true
	}

	return nil, false
}

func okResult(payload map[string]any) map[string]any {
	payload["ok"] = true
	return payload
}

func errResult(msg string) map[string]any {
	return map[string]any{
		"ok":    false,
		"error": msg,
	}
}
