//go:build js && wasm

package main

import (
	"encoding/base64"
	"fmt"
	"math"
	"syscall/js"

	"github.com/example/go-pocket-tts/internal/audio"
	"github.com/example/go-pocket-tts/internal/text"
)

func main() {
	kernel := map[string]any{
		"version":       "0.1.0-wasm",
		"normalize":     js.FuncOf(normalizeText),
		"tokenize":      js.FuncOf(tokenizeText),
		"synthesizeWav": js.FuncOf(synthesizeWav),
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

func synthesizeWav(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errResult("missing text argument")
	}

	normalized, err := text.Normalize(args[0].String())
	if err != nil {
		return errResult(err.Error())
	}

	prep := text.NewPreprocessor()
	tokens := prep.Preprocess(normalized)
	if len(tokens) == 0 {
		return errResult("no tokens produced after preprocessing")
	}

	const sampleRate = 24000
	samples := synthTokenTone(tokens, sampleRate)
	samples = audio.PeakNormalize(samples)
	samples = audio.FadeIn(samples, sampleRate, 8)
	samples = audio.FadeOut(samples, sampleRate, 12)

	wav, err := audio.EncodeWAVPCM16(samples, sampleRate)
	if err != nil {
		return errResult(fmt.Sprintf("encode wav: %v", err))
	}

	b64 := base64.StdEncoding.EncodeToString(wav)
	return okResult(map[string]any{
		"text":        normalized,
		"sample_rate": sampleRate,
		"wav_base64":  b64,
	})
}

func synthTokenTone(tokens []int, sampleRate int) []float32 {
	segment := sampleRate / 14
	total := segment * len(tokens)
	samples := make([]float32, total)

	for i, token := range tokens {
		freq := 140.0 + float64(token%80)*6.0
		start := i * segment
		end := start + segment
		for j := start; j < end; j++ {
			t := float64(j-start) / float64(sampleRate)
			v := 0.25*math.Sin(2*math.Pi*freq*t) + 0.1*math.Sin(2*math.Pi*(freq*0.5)*t)
			samples[j] += float32(v)
		}
	}

	return samples
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
