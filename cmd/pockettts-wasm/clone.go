package main

import (
	"errors"
	"fmt"

	"github.com/cwbudde/go-pocket-tts/internal/audio"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

// errCloningUnavailable reports an engine loaded from an ungated checkpoint,
// whose Mimi encoder weights are zeroed.
var errCloningUnavailable = errors.New("voice cloning unavailable: the loaded checkpoint has zeroed Mimi encoder " +
	"weights (ungated kyutai/pocket-tts-without-voice-cloning). Accept the terms of the gated kyutai/pocket-tts " +
	"repo on Hugging Face, download the model.safetensors of this model config there and pick it locally " +
	"with \"Use local checkpoint\"")

// canClone reports whether the engine can clone voices from audio prompts.
func (e *nativeEngine) canClone() bool { return e != nil && e.voice != nil }

// cloneVoice encodes a WAV voice prompt (any sample rate and channel count)
// like pockettts export-voice: the prompt is cut to 30 s, resampled to 24 kHz
// and ended on a pause, then encoded to the [1, T, d_model] conditioning. It
// returns that conditioning as an audio_prompt safetensors embedding, which
// synthesize accepts as voiceSafetensors, and its frame count T.
func (e *nativeEngine) cloneVoice(wav []byte) (blob []byte, frames int64, err error) {
	if !e.canClone() {
		return nil, 0, errCloningUnavailable
	}

	samples, rate, err := audio.DecodePromptWAV(wav)
	if err != nil {
		return nil, 0, fmt.Errorf("clone voice: decode WAV: %w", err)
	}

	samples, err = audio.PrepareVoicePrompt(samples, rate)
	if err != nil {
		return nil, 0, fmt.Errorf("clone voice: %w", err)
	}

	cond, err := e.voice.Encode(samples)
	if err != nil {
		return nil, 0, fmt.Errorf("clone voice: %w", err)
	}

	blob, err = safetensors.EncodeTensors([]safetensors.Tensor{{
		Name:  "audio_prompt",
		Shape: cond.Shape(),
		Data:  cond.RawData(),
	}})
	if err != nil {
		return nil, 0, fmt.Errorf("clone voice: encode embedding: %w", err)
	}

	return blob, cond.Shape()[1], nil
}
