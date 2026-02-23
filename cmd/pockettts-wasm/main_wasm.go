//go:build js && wasm

package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"syscall/js"
	"time"

	"github.com/example/go-pocket-tts/internal/audio"
	"github.com/example/go-pocket-tts/internal/text"
)

const (
	modelSampleRate    = 24000
	flowTemperature    = 0.7
	flowDecodeSteps    = 1
	flowEOSThreshold   = -4.0
	flowFramesAfterEOS = 2
	flowMinSteps       = 16
	flowMaxSteps       = 256
)

type tensorPayload struct {
	DType string    `json:"dtype"`
	Shape []int64   `json:"shape"`
	F32   []float32 `json:"f32,omitempty"`
	I64   []int64   `json:"i64,omitempty"`
}

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

var rng = rand.New(rand.NewSource(time.Now().UnixNano()))

func main() {
	kernel := map[string]any{
		"version":         "0.2.0-wasm",
		"sampleRate":      modelSampleRate,
		"normalize":       js.FuncOf(normalizeText),
		"tokenize":        js.FuncOf(tokenizeText),
		"synthesizeModel": js.FuncOf(synthesizeModelAsync),
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

	prep := text.NewPreprocessor()
	tokens := prep.Preprocess(normalized)

	out := make([]any, len(tokens))
	for i, v := range tokens {
		out[i] = v
	}

	return okResult(map[string]any{
		"text":   normalized,
		"tokens": out,
	})
}

type synthesizeOptions struct {
	Temperature    float64
	VoiceEmbedding *tensorPayload
}

func parseSynthOptions(args []js.Value) synthesizeOptions {
	opts := synthesizeOptions{Temperature: flowTemperature}

	// args[2] is an optional options object: { temperature?: number, voiceEmbedding?: {dtype, shape, f32} }
	if len(args) < 3 {
		return opts
	}
	optVal := args[2]
	if optVal.IsUndefined() || optVal.IsNull() {
		return opts
	}

	temp := optVal.Get("temperature")
	if !temp.IsUndefined() && !temp.IsNull() {
		opts.Temperature = temp.Float()
	}

	ve := optVal.Get("voiceEmbedding")
	if !ve.IsUndefined() && !ve.IsNull() {
		veJSON := js.Global().Get("JSON").Call("stringify", ve).String()
		var tp tensorPayload
		if err := json.Unmarshal([]byte(veJSON), &tp); err == nil && len(tp.F32) > 0 {
			opts.VoiceEmbedding = &tp
		}
	}

	return opts
}

func synthesizeModelAsync(_ js.Value, args []js.Value) any {
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
			res, err := synthesizeModel(textArg, &progress, opts)
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

func synthesizeModel(input string, progress *progressReporter, opts synthesizeOptions) (map[string]any, error) {
	progress.Emit("prepare", 0, 100, "normalizing and tokenizing input")
	normalized := normalizePromptForModel(input)
	if normalized == "" {
		return nil, fmt.Errorf("text is empty")
	}

	prep := text.NewPreprocessor()
	tokens := prep.Preprocess(normalized)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens produced after preprocessing")
	}
	progress.Emit("prepare", 5, 100, fmt.Sprintf("tokenized %d units", len(tokens)))

	progress.Emit("text_conditioner", 10, 100, "running text conditioner")
	textCondOutputs, err := runGraph("text_conditioner", map[string]tensorPayload{
		"tokens": {
			DType: "int64",
			Shape: []int64{1, int64(len(tokens))},
			I64:   intsToI64(tokens),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("text_conditioner: %w", err)
	}
	textEmb, err := pickOutput(textCondOutputs, "text_embeddings")
	if err != nil {
		return nil, fmt.Errorf("text_conditioner output: %w", err)
	}
	progress.Emit("text_conditioner", 15, 100, "text embeddings ready")

	if opts.VoiceEmbedding != nil {
		textEmb = prependVoiceEmb(opts.VoiceEmbedding, &textEmb)
		progress.Emit("voice", 17, 100, fmt.Sprintf("prepended voice embedding (%d frames)", opts.VoiceEmbedding.Shape[1]))
	}

	latentDim := int64(32)
	condDim := int64(1024)

	sequenceFrames := make([][]float32, 0, 256)
	bos := make([]float32, latentDim)
	// Keep seed finite because feeds are serialized through JSON for the JS ORT bridge.
	// NaN is not representable in JSON and would fail marshaling.
	for i := range bos {
		bos[i] = 0
	}
	sequenceFrames = append(sequenceFrames, bos)

	generated := make([][]float32, 0, 256)
	eosStep := -1
	maxSteps := flowMaxSteps
	if maxSteps < len(tokens)*5 {
		maxSteps = len(tokens) * 5
	}
	if maxSteps > flowMaxSteps {
		maxSteps = flowMaxSteps
	}
	progress.Emit("autoregressive", 20, 100, fmt.Sprintf("starting generation loop (%d max steps)", maxSteps))

	for step := 0; step < maxSteps; step++ {
		if step == 0 || step%2 == 0 {
			progress.Emit(
				"autoregressive",
				20+int((float64(step)/float64(maxSteps))*60.0),
				100,
				fmt.Sprintf("step %d/%d", step+1, maxSteps),
			)
		}
		seqFlat := flattenFrames(sequenceFrames, int(latentDim))

		mainOutputs, err := runGraph("flow_lm_main", map[string]tensorPayload{
			"sequence": {
				DType: "float32",
				Shape: []int64{1, int64(len(sequenceFrames)), latentDim},
				F32:   seqFlat,
			},
			"text_embeddings": textEmb,
		})
		if err != nil {
			return nil, fmt.Errorf("flow_lm_main: %w", err)
		}

		hidden, err := pickOutput(mainOutputs, "last_hidden")
		if err != nil {
			return nil, fmt.Errorf("flow_lm_main hidden: %w", err)
		}
		eos, err := pickOutput(mainOutputs, "eos_logits")
		if err != nil {
			return nil, fmt.Errorf("flow_lm_main eos: %w", err)
		}

		condition := copyOrTile(hidden.F32, int(condDim))
		x := make([]float32, latentDim)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * math.Sqrt(opts.Temperature))
		}

		for k := 0; k < flowDecodeSteps; k++ {
			s := float32(float64(k) / float64(flowDecodeSteps))
			t := float32(float64(k+1) / float64(flowDecodeSteps))
			flowOut, err := runGraph("flow_lm_flow", map[string]tensorPayload{
				"condition": {
					DType: "float32",
					Shape: []int64{1, condDim},
					F32:   condition,
				},
				"s": {
					DType: "float32",
					Shape: []int64{1, 1},
					F32:   []float32{s},
				},
				"t": {
					DType: "float32",
					Shape: []int64{1, 1},
					F32:   []float32{t},
				},
				"x": {
					DType: "float32",
					Shape: []int64{1, latentDim},
					F32:   x,
				},
			})
			if err != nil {
				return nil, fmt.Errorf("flow_lm_flow: %w", err)
			}
			dir, err := pickOutput(flowOut, "flow_direction")
			if err != nil {
				return nil, fmt.Errorf("flow_lm_flow output: %w", err)
			}
			dt := float32(1.0 / float64(flowDecodeSteps))
			for i := range x {
				x[i] += dir.F32[i%len(dir.F32)] * dt
			}
		}

		next := make([]float32, latentDim)
		copy(next, x)
		sequenceFrames = append(sequenceFrames, next)
		generated = append(generated, next)

		eosVal := float32(-10)
		if len(eos.F32) > 0 {
			eosVal = eos.F32[0]
		}
		if eosVal > flowEOSThreshold && eosStep < 0 {
			eosStep = step
		}
		if eosStep >= 0 && step >= eosStep+flowFramesAfterEOS && step+1 >= flowMinSteps {
			progress.Emit(
				"autoregressive",
				80,
				100,
				fmt.Sprintf("stopped at step %d (EOS+tail)", step+1),
			)
			break
		}
	}

	if len(generated) == 0 {
		return nil, fmt.Errorf("model loop produced no latents")
	}

	steps := int64(len(generated))
	latentSeq := flattenFrames(generated, int(latentDim))
	progress.Emit("latent_to_mimi", 85, 100, "projecting latent for Mimi decoder")

	var mimiInput tensorPayload
	latentToMimiOutputs, err := runGraph("latent_to_mimi", map[string]tensorPayload{
		"latent": {
			DType: "float32",
			Shape: []int64{1, steps, latentDim},
			F32:   latentSeq,
		},
	})
	if err == nil {
		mimiInput, err = pickOutput(latentToMimiOutputs, "mimi_latent")
		if err != nil {
			return nil, fmt.Errorf("latent_to_mimi output: %w", err)
		}
	} else {
		// Fallback for manifests without latent_to_mimi graph.
		mimiDim := int64(512)
		fallback := make([]float32, mimiDim*steps)
		for t := int64(0); t < steps; t++ {
			src := generated[t]
			for c := int64(0); c < mimiDim; c++ {
				fallback[c*steps+t] = src[c%int64(len(src))]
			}
		}
		mimiInput = tensorPayload{
			DType: "float32",
			Shape: []int64{1, mimiDim, steps},
			F32:   fallback,
		}
	}

	mimiOutputs, err := runGraph("mimi_decoder", map[string]tensorPayload{
		"latent": mimiInput,
	})
	if err != nil {
		return nil, fmt.Errorf("mimi_decoder: %w", err)
	}
	audioOut, err := pickOutput(mimiOutputs, "audio")
	if err != nil {
		return nil, fmt.Errorf("mimi_decoder output: %w", err)
	}
	if len(audioOut.F32) == 0 {
		return nil, fmt.Errorf("decoder returned empty audio")
	}
	progress.Emit("mimi_decoder", 95, 100, "encoding WAV")

	wav, err := audio.EncodeWAVPCM16(audioOut.F32, modelSampleRate)
	if err != nil {
		return nil, fmt.Errorf("encode wav: %w", err)
	}

	res := okResult(map[string]any{
		"text":        normalized,
		"token_count": len(tokens),
		"frames":      len(generated),
		"sample_rate": modelSampleRate,
		"wav_base64":  base64.StdEncoding.EncodeToString(wav),
	})
	progress.Emit("done", 100, 100, "synthesis complete")
	return res, nil
}

func runGraph(graphName string, feeds map[string]tensorPayload) (map[string]tensorPayload, error) {
	bridge := js.Global().Get("PocketTTSBridge")
	if bridge.IsUndefined() || bridge.IsNull() {
		return nil, fmt.Errorf("PocketTTSBridge is not available")
	}

	payload, err := json.Marshal(feeds)
	if err != nil {
		return nil, fmt.Errorf("marshal feeds: %w", err)
	}

	promise := bridge.Call("runGraph", graphName, string(payload))
	outputJSON, err := awaitPromiseString(promise)
	if err != nil {
		return nil, err
	}

	var outputs map[string]tensorPayload
	if err := json.Unmarshal([]byte(outputJSON), &outputs); err != nil {
		return nil, fmt.Errorf("decode bridge output: %w", err)
	}
	return outputs, nil
}

func awaitPromiseString(promise js.Value) (string, error) {
	type result struct {
		value string
		err   error
	}
	ch := make(chan result, 1)

	then := js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) > 0 {
			ch <- result{value: args[0].String()}
		} else {
			ch <- result{value: ""}
		}
		return nil
	})
	catch := js.FuncOf(func(_ js.Value, args []js.Value) any {
		msg := "promise rejected"
		if len(args) > 0 {
			v := args[0]
			// JS Error objects have a .message property; .String() gives "[object Object]".
			if m := v.Get("message"); !m.IsUndefined() && !m.IsNull() {
				msg = m.String()
			} else {
				msg = v.Call("toString").String()
			}
		}
		ch <- result{err: fmt.Errorf(msg)}
		return nil
	})

	promise.Call("then", then)
	promise.Call("catch", catch)
	res := <-ch
	then.Release()
	catch.Release()
	return res.value, res.err
}

func prependVoiceEmb(voice *tensorPayload, text *tensorPayload) tensorPayload {
	combined := make([]float32, 0, len(voice.F32)+len(text.F32))
	combined = append(combined, voice.F32...)
	combined = append(combined, text.F32...)
	return tensorPayload{
		DType: "float32",
		Shape: []int64{1, voice.Shape[1] + text.Shape[1], voice.Shape[2]},
		F32:   combined,
	}
}

func pickOutput(outputs map[string]tensorPayload, preferred string) (tensorPayload, error) {
	if out, ok := outputs[preferred]; ok {
		return out, nil
	}
	for _, out := range outputs {
		return out, nil
	}
	return tensorPayload{}, fmt.Errorf("no outputs")
}

func flattenFrames(frames [][]float32, frameDim int) []float32 {
	out := make([]float32, 0, len(frames)*frameDim)
	for _, f := range frames {
		out = append(out, copyOrTile(f, frameDim)...)
	}
	return out
}

func copyOrTile(src []float32, n int) []float32 {
	if n <= 0 {
		return nil
	}
	if len(src) == n {
		out := make([]float32, n)
		copy(out, src)
		return out
	}
	out := make([]float32, n)
	if len(src) == 0 {
		return out
	}
	for i := range out {
		out[i] = src[i%len(src)]
	}
	return out
}

func normalizePromptForModel(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ")
	if s == "" {
		return s
	}
	last := s[len(s)-1]
	if last != '.' && last != '!' && last != '?' {
		s += "."
	}
	return s
}

func intsToI64(in []int) []int64 {
	out := make([]int64, len(in))
	for i, v := range in {
		out[i] = int64(v)
	}
	return out
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
