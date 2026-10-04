//go:build js && wasm

package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
	"sync"
	"syscall/js"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/audio"
	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
	"github.com/cwbudde/go-pocket-tts/internal/text"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
)

const maxTokensPerChunk = 50

// heapLimit keeps the Go heap below the 4 GB of 32-bit WASM memory: near it,
// the GC collects instead of growing the heap to twice the live weights.
const heapLimit = 3584 << 20

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
	Temperature        float64
	EOSThreshold       float64
	MaxSteps           int
	SamplerDecodeSteps int
	VoiceSafetensors   []byte
}

var (
	defaults = config.DefaultConfig()
	engineMu sync.RWMutex
	engine   *nativeEngine
)

func main() {
	debug.SetMemoryLimit(heapLimit)

	kernel := map[string]any{
		"version":     "0.6.0-wasm",
		"sampleRate":  audio.ExpectedSampleRate,
		"loadModel":   js.FuncOf(loadModelAsync),
		"unloadModel": js.FuncOf(unloadModel),
		"normalize":   js.FuncOf(normalizeText),
		"tokenize":    js.FuncOf(tokenizeText),
		"synthesize":  js.FuncOf(synthesizeAsync),
		"cloneVoice":  js.FuncOf(cloneVoiceAsync),
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

	engineMu.RLock()
	currentEngine := engine
	engineMu.RUnlock()

	if currentEngine == nil || currentEngine.tokenizer == nil {
		return errResult("tokenizer not ready; call loadModel first")
	}

	normalized, err := text.Normalize(args[0].String())
	if err != nil {
		return errResult(err.Error())
	}

	chunks, err := text.PrepareChunks(normalized, currentEngine.tokenizer, maxTokensPerChunk, text.OptionsFor(currentEngine.model))
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

// loadModelAsync is the JS-facing wrapper. It expects:
//
//	args[0] – Uint8Array: safetensors model bytes
//	args[1] – Uint8Array: tokenizer bytes, either a SentencePiece model
//	          (tokenizer.model) or a Hugging Face tokenizer.json; the format
//	          is sniffed from the bytes
//	args[2] – Function (optional): progress callback
//	args[3] – Object (optional): {config: "<model config name>"}, the
//	          embedded model config the checkpoint belongs to; omitted or ""
//	          selects config.DefaultLanguage
func loadModelAsync(_ js.Value, args []js.Value) any {
	promiseCtor := js.Global().Get("Promise")
	var handler js.Func
	handler = js.FuncOf(func(_ js.Value, pArgs []js.Value) any {
		defer handler.Release()
		resolve := pArgs[0]
		reject := pArgs[1]

		if len(args) < 2 {
			reject.Invoke("loadModel requires model bytes and tokenizer bytes arguments")
			return nil
		}

		modelBytes, ok := copyJSBytes(args[0])
		if !ok || len(modelBytes) == 0 {
			reject.Invoke("model safetensors bytes must be a non-empty Uint8Array/ArrayBuffer")
			return nil
		}

		tokBytes, ok := copyJSBytes(args[1])
		if !ok || len(tokBytes) == 0 {
			reject.Invoke("tokenizer model bytes must be a non-empty Uint8Array/ArrayBuffer")
			return nil
		}

		var progress progressReporter
		if len(args) > 2 && args[2].Type() == js.TypeFunction {
			progress.cb = args[2]
		}

		configName := ""
		if len(args) > 3 && args[3].Type() == js.TypeObject {
			if v := args[3].Get("config"); v.Type() == js.TypeString {
				configName = v.String()
			}
		}

		go func() {
			// Hand loadModel the only reference to the checkpoint copy, so
			// it is freed once the weights are decoded.
			data := modelBytes
			modelBytes = nil

			res, err := loadModel(data, tokBytes, configName, &progress)
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

func loadModel(modelSafetensors, tokenizerBytes []byte, configName string, progress *progressReporter) (map[string]any, error) {
	modelSize := len(modelSafetensors)

	loaded, err := newEngine(modelSafetensors, tokenizerBytes, configName, progress.Emit)
	if err != nil {
		return nil, err
	}

	engineMu.Lock()
	oldEngine := engine
	engine = loaded
	engineMu.Unlock()

	if oldEngine != nil && oldEngine.runtime != nil {
		oldEngine.runtime.Close()
	}

	// Free the checkpoint copy now, so synthesis reuses its memory instead of
	// growing the WASM heap.
	runtime.GC()

	progress.Emit("load", 100, 100, "model ready")
	return okResult(map[string]any{
		"config":      loaded.name,
		"model_bytes": modelSize,
		// can_clone is false for an ungated checkpoint, whose Mimi encoder
		// weights are zeroed; cloneVoice then fails.
		"can_clone": loaded.canClone(),
	}), nil
}

// loadedEngine returns the engine loadModel installed, nil before the first
// load and after unloadModel.
func loadedEngine() *nativeEngine {
	engineMu.RLock()
	defer engineMu.RUnlock()

	return engine
}

// cloneVoiceAsync is the JS-facing wrapper of cloneVoice. It expects:
//
//	args[0] – Uint8Array: a WAV voice prompt (any sample rate and channel
//	          count; the first 30 s are used)
//
// It resolves to {ok, voice_safetensors: Uint8Array, frames} and rejects when
// the loaded checkpoint cannot clone (an ungated one) or the WAV is invalid.
// voice_safetensors is an audio_prompt embedding for synthesize's
// voiceSafetensors option.
func cloneVoiceAsync(_ js.Value, args []js.Value) any {
	promiseCtor := js.Global().Get("Promise")
	var handler js.Func
	handler = js.FuncOf(func(_ js.Value, pArgs []js.Value) any {
		defer handler.Release()

		resolve, reject := pArgs[0], pArgs[1]

		if len(args) < 1 {
			reject.Invoke("cloneVoice requires WAV bytes")
			return nil
		}

		wav, ok := copyJSBytes(args[0])
		if !ok || len(wav) == 0 {
			reject.Invoke("voice prompt WAV bytes must be a non-empty Uint8Array/ArrayBuffer")
			return nil
		}

		go func() {
			browserYield() // let the page show the cloning status first

			currentEngine := loadedEngine()
			if currentEngine == nil {
				reject.Invoke("model is not loaded; call loadModel first")
				return
			}

			blob, frames, err := currentEngine.cloneVoice(wav)
			if err != nil {
				reject.Invoke(err.Error())
				return
			}

			out := js.Global().Get("Uint8Array").New(len(blob))
			js.CopyBytesToJS(out, blob)
			resolve.Invoke(js.ValueOf(okResult(map[string]any{
				"voice_safetensors": out,
				"frames":            frames,
			})))
		}()

		return nil
	})

	return promiseCtor.New(handler)
}

// unloadModel drops the loaded model, so the page can free its memory before
// it loads another config.
func unloadModel(_ js.Value, _ []js.Value) any {
	engineMu.Lock()
	oldEngine := engine
	engine = nil
	engineMu.Unlock()

	if oldEngine != nil && oldEngine.runtime != nil {
		oldEngine.runtime.Close()
	}

	runtime.GC()

	return okResult(map[string]any{})
}

func parseSynthOptions(args []js.Value) synthesizeOptions {
	opts := synthesizeOptions{
		Temperature:        defaults.TTS.Temperature,
		EOSThreshold:       defaults.TTS.EOSThreshold,
		MaxSteps:           defaults.TTS.MaxSteps,
		SamplerDecodeSteps: defaults.TTS.SamplerDecodeSteps,
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

	// samplerSteps replaces the deprecated lsdSteps option, which is still
	// honoured when samplerSteps is absent.
	stepsVal := optVal.Get("samplerSteps")
	if stepsVal.IsUndefined() || stepsVal.IsNull() {
		stepsVal = optVal.Get("lsdSteps")
	}

	if !stepsVal.IsUndefined() && !stepsVal.IsNull() {
		steps := stepsVal.Int()
		if steps > 0 {
			opts.SamplerDecodeSteps = steps
		}
	}

	if v := optVal.Get("voiceSafetensors"); !v.IsUndefined() && !v.IsNull() {
		if b, ok := copyJSBytes(v); ok {
			opts.VoiceSafetensors = b
		}
	}

	return opts
}

// browserYield schedules a setTimeout(0) via time.Sleep, handing the main
// browser thread back to the event loop so pending UI repaints can happen
// before the Go goroutine resumes.
func browserYield() {
	time.Sleep(time.Millisecond)
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
			browserYield() // let the browser repaint the spinner before any work
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
	engineMu.RUnlock()

	if currentEngine == nil || currentEngine.runtime == nil {
		return nil, fmt.Errorf("model is not loaded; call loadModel first")
	}

	progress.Emit("prepare", 0, 100, "normalizing and chunking input")
	browserYield()
	normalized, err := text.Normalize(input)
	if err != nil {
		return nil, err
	}

	chunks, err := text.PrepareChunks(normalized, currentEngine.tokenizer, maxTokensPerChunk, text.OptionsFor(currentEngine.model))
	if err != nil {
		return nil, err
	}
	if len(chunks) == 0 {
		return nil, fmt.Errorf("no chunks produced")
	}
	progress.Emit("prepare", 10, 100, fmt.Sprintf("prepared %d chunks", len(chunks)))

	var voiceEmb *tts.VoiceEmbedding
	var voiceState *safetensors.VoiceModelState
	if len(opts.VoiceSafetensors) > 0 {
		browserYield()
		kind, vErr := safetensors.InspectVoiceFileBytes(opts.VoiceSafetensors)
		if vErr != nil {
			return nil, fmt.Errorf("inspect voice safetensors: %w", vErr)
		}
		if kind == safetensors.VoiceFileModelState {
			voiceState, vErr = safetensors.LoadVoiceModelStateFromBytes(opts.VoiceSafetensors)
			if vErr != nil {
				return nil, fmt.Errorf("load voice model state: %w", vErr)
			}
			progress.Emit("voice", 15, 100, "loaded voice model state")
		} else {
			vData, vShape, vErr := safetensors.LoadVoiceEmbeddingFromBytes(opts.VoiceSafetensors)
			if vErr != nil {
				return nil, fmt.Errorf("load voice embedding: %w", vErr)
			}
			voiceEmb = &tts.VoiceEmbedding{Data: vData, Shape: vShape}
			progress.Emit("voice", 15, 100, fmt.Sprintf("loaded voice embedding (%d frames)", vShape[1]))
		}
	}

	allAudio := make([]float32, 0, audio.ExpectedSampleRate)
	totalTokens := 0
	nChunks := len(chunks)
	for i, chunk := range chunks {
		chunkStart := 20 + int((float64(i)/float64(nChunks))*70)
		chunkWidth := int(70.0 / float64(nChunks))

		progress.Emit("synthesize", chunkStart, 100, fmt.Sprintf("chunk %d/%d · step 0", i+1, nChunks))
		browserYield() // repaint before TextEmbeddings + PromptFlow block the thread

		estimatedMaxSteps := text.EstimateMaxFrames(len(chunk.TokenIDs), text.DefaultMimiFrameRate)
		maxSteps := wasmGenerationStepLimit(opts.MaxSteps, estimatedMaxSteps)
		mimiStepsPerLatent := 16

		cfg := tts.RuntimeGenerateConfig{
			Temperature:        opts.Temperature,
			EOSThreshold:       opts.EOSThreshold,
			MaxSteps:           maxSteps,
			EstimatedMaxSteps:  estimatedMaxSteps,
			SamplerDecodeSteps: opts.SamplerDecodeSteps,
			FramesAfterEOS:     currentEngine.model.FramesAfterEOS(chunk.FramesAfterEOS()),
			MimiStepsPerLatent: mimiStepsPerLatent,
			MimiSequenceLength: estimatedMaxSteps * mimiStepsPerLatent,
			VoiceEmbedding:     voiceEmb,
			VoiceModelState:    voiceState,
			StepCallback: func(step, _ int) {
				stepPct := 0
				if maxSteps > 0 {
					stepPct = int((float64(step) / float64(maxSteps)) * float64(chunkWidth))
				}
				pct := chunkStart + stepPct
				detail := fmt.Sprintf("chunk %d/%d · step %d", i+1, nChunks, step)
				progress.Emit("synthesize", pct, 100, detail)
				if step%10 == 0 {
					browserYield()
				}
			},
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

func wasmGenerationStepLimit(configured, estimated int) int {
	if estimated > 0 && (configured <= 0 || configured == defaults.TTS.MaxSteps) {
		return estimated
	}

	return configured
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
