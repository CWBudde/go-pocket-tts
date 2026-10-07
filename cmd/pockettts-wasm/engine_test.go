package main

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/audio"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	nativemodel "github.com/cwbudde/go-pocket-tts/internal/native"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
	"github.com/cwbudde/go-pocket-tts/internal/tokenizer"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
)

func TestNewEngine_UnknownConfig(t *testing.T) {
	_, err := newEngine([]byte("model"), []byte("tokenizer"), "klingon", nil)
	if err == nil || !strings.Contains(err.Error(), "klingon") {
		t.Fatalf("err = %v, want an unknown config error naming klingon", err)
	}
}

func TestNewEngine_RejectsBadTokenizerBeforeModel(t *testing.T) {
	// "" selects the default config, whose tokenizer check runs before the
	// model is opened.
	_, err := newEngine([]byte("model"), []byte("not a tokenizer"), "", nil)
	if err == nil || !strings.Contains(err.Error(), "load tokenizer") {
		t.Fatalf("err = %v, want a tokenizer error", err)
	}
}

// TestNewEngine_German builds the engine from the local german download
// (pockettts model download --language german) and checks that the page's
// config choice reaches the engine.
func TestNewEngine_German(t *testing.T) {
	const dir = "../../models/german/"

	modelBytes, err := os.ReadFile(dir + "model.safetensors")
	if err != nil {
		t.Skipf("german model not downloaded: %v", err)
	}

	tokBytes, err := os.ReadFile(dir + "tokenizer.json")
	if err != nil {
		t.Skipf("german tokenizer not downloaded: %v", err)
	}

	var stages []string

	e, err := newEngine(modelBytes, tokBytes, "german", func(stage string, _, _ int, _ string) {
		stages = append(stages, stage)
	})
	if err != nil {
		t.Fatalf("newEngine: %v", err)
	}
	defer e.runtime.Close()

	if e.name != "german" || !strings.Contains(e.model.FlowLM.LookupTable.TokenizerPath, "/german/") {
		t.Fatalf("engine config = %q (tokenizer %s), want german", e.name, e.model.FlowLM.LookupTable.TokenizerPath)
	}

	if !slices.Contains(stages, "tokenizer") || !slices.Contains(stages, "load") {
		t.Errorf("progress stages = %v", stages)
	}

	cfg, err := modelcfg.Lookup("german")
	if err != nil {
		t.Fatal(err)
	}

	want, err := tokenizer.Load(dir+"tokenizer.json", cfg.FlowLM.LookupTable.NBins)
	if err != nil {
		t.Fatal(err)
	}

	wantIDs, err := want.Encode(cfg.DefaultText)
	if err != nil {
		t.Fatal(err)
	}

	gotIDs, err := e.tokenizer.Encode(cfg.DefaultText)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(gotIDs, wantIDs) {
		t.Errorf("tokens = %v, want %v", gotIDs, wantIDs)
	}
}

// TestNewEngine_ReleasesCheckpointBytes checks that the engine drops the
// checkpoint bytes once the weights are decoded, so the 24-layer models fit in
// 4 GB of WASM memory. The gated checkpoint also loads the voice encoder,
// which must not keep them either.
func TestNewEngine_ReleasesCheckpointBytes(t *testing.T) {
	for _, dir := range []string{"../../models/german/", "../../models/gated/german/"} {
		t.Run(dir, func(t *testing.T) {
			modelBytes, err := os.ReadFile(dir + "model.safetensors")
			if err != nil {
				t.Skipf("german model not downloaded: %v", err)
			}

			tokBytes, err := os.ReadFile(dir + "tokenizer.json")
			if err != nil {
				t.Skipf("german tokenizer not downloaded: %v", err)
			}

			released := make(chan struct{})
			runtime.AddCleanup(&modelBytes[0], func(ch chan struct{}) { close(ch) }, released)

			e, err := newEngine(modelBytes, tokBytes, "german", nil)
			if err != nil {
				t.Fatalf("newEngine: %v", err)
			}
			defer e.runtime.Close()

			modelBytes = nil //nolint:wastedassign // drop the test's own reference

			for range 20 {
				runtime.GC()

				select {
				case <-released:
					return
				case <-time.After(50 * time.Millisecond):
				}
			}

			t.Fatal("the engine keeps the checkpoint bytes alive")
		})
	}
}

// Widths of the synthetic voice encoder: a latent of syntheticInnerDim,
// projected to the 1024-wide FlowLM conditioning.
const syntheticInnerDim, syntheticCondDim = 4, 1024

// syntheticTokenizerJSON returns a Unigram tokenizer.json whose vocab size
// matches the german config's n_bins, as the tokenizer loader requires.
func syntheticTokenizerJSON(t *testing.T) []byte {
	t.Helper()

	mc, err := modelcfg.Lookup("german")
	if err != nil {
		t.Fatal(err)
	}

	vocab := make([][]any, mc.FlowLM.LookupTable.NBins)
	vocab[0] = []any{"<unk>", 0}

	for i := 1; i < len(vocab); i++ {
		vocab[i] = []any{"p" + strconv.Itoa(i), -float64(i)}
	}

	blob, err := json.Marshal(map[string]any{
		"model":         map[string]any{"type": "Unigram", "unk_id": 0, "vocab": vocab},
		"pre_tokenizer": map[string]any{"type": "Metaspace", "replacement": "▁"},
	})
	if err != nil {
		t.Fatal(err)
	}

	return blob
}

// syntheticEncoderCheckpoint returns a scaled-down checkpoint holding only the
// voice-encoder tensors (mimi.encoder, mimi.encoder_transformer,
// mimi.downsample and flow_lm.speaker_proj_weight): SEANet channels 2..16, a
// 16-wide transformer. zeroed makes every value 0, like the ungated
// kyutai/pocket-tts-without-voice-cloning checkpoints.
func syntheticEncoderCheckpoint(t *testing.T, zeroed bool) []byte {
	t.Helper()

	const nf = 2

	d, ff := int64(8*nf), int64(16*nf)
	seed := 0

	mk := func(name string, shape ...int64) safetensors.Tensor {
		n := int64(1)
		for _, s := range shape {
			n *= s
		}

		data := make([]float32, n)
		if !zeroed {
			for i := range data {
				seed++
				data[i] = float32(0.3 * math.Sin(float64(seed)*0.7))
			}
		}

		return safetensors.Tensor{Name: name, Shape: shape, Data: data}
	}
	ones := func(name string, n int64) safetensors.Tensor {
		data := make([]float32, n)
		for i := range data {
			data[i] = 1
		}

		return safetensors.Tensor{Name: name, Shape: []int64{n}, Data: data}
	}
	conv := func(idx string, out, in, k int64) []safetensors.Tensor {
		p := "mimi.encoder.model." + idx + ".conv."

		return []safetensors.Tensor{mk(p+"weight", out, in, k), mk(p+"bias", out)}
	}
	res := func(idx string, ch int64) []safetensors.Tensor {
		p := "mimi.encoder.model." + idx + ".block."

		return []safetensors.Tensor{
			mk(p+"1.conv.weight", ch/2, ch, 3), mk(p+"1.conv.bias", ch/2),
			mk(p+"3.conv.weight", ch, ch/2, 1), mk(p+"3.conv.bias", ch),
		}
	}

	ts := make([]safetensors.Tensor, 0, 64)
	ts = append(ts, conv("0", nf, 1, 7)...)
	ts = append(ts, res("1", nf)...)
	ts = append(ts, conv("3", 2*nf, nf, 8)...)
	ts = append(ts, res("4", 2*nf)...)
	ts = append(ts, conv("6", 4*nf, 2*nf, 10)...)
	ts = append(ts, res("7", 4*nf)...)
	ts = append(ts, conv("9", d, 4*nf, 12)...)
	ts = append(ts, conv("11", d, d, 3)...)

	for i := range 2 {
		p := "mimi.encoder_transformer.transformer.layers." + strconv.Itoa(i) + "."
		ts = append(
			ts,
			ones(p+"norm1.weight", d), mk(p+"norm1.bias", d),
			ones(p+"norm2.weight", d), mk(p+"norm2.bias", d),
			mk(p+"self_attn.in_proj.weight", 3*d, d),
			mk(p+"self_attn.out_proj.weight", d, d),
			mk(p+"linear1.weight", ff, d),
			mk(p+"linear2.weight", d, ff),
			mk(p+"layer_scale_1.scale", d),
			mk(p+"layer_scale_2.scale", d),
		)
	}

	ts = append(
		ts,
		mk("mimi.downsample.conv.conv.weight", syntheticInnerDim, d, 32),
		mk("flow_lm.speaker_proj_weight", syntheticCondDim, syntheticInnerDim),
	)

	blob, err := safetensors.EncodeTensors(ts)
	if err != nil {
		t.Fatalf("encode synthetic checkpoint: %v", err)
	}

	return blob
}

// stubModelLoader makes newEngine skip the FlowLM and Mimi decoder, which the
// synthetic checkpoints do not hold. The store is still open while the stub
// runs, like for the real loader.
func stubModelLoader(t *testing.T) {
	t.Helper()

	orig := loadModelFromStore
	loadModelFromStore = func(*safetensors.Store, nativemodel.Config) (*nativemodel.Model, error) {
		return &nativemodel.Model{}, nil
	}

	t.Cleanup(func() { loadModelFromStore = orig })
}

// newSyntheticEngine builds the engine from a synthetic encoder checkpoint.
func newSyntheticEngine(t *testing.T, zeroed bool) *nativeEngine {
	t.Helper()
	stubModelLoader(t)

	e, err := newEngine(syntheticEncoderCheckpoint(t, zeroed), syntheticTokenizerJSON(t), "german", nil)
	if err != nil {
		t.Fatalf("newEngine: %v", err)
	}

	return e
}

// promptWAV returns seconds of a 220 Hz tone at sampleRate as a PCM16 WAV.
func promptWAV(t *testing.T, sampleRate int, seconds float64) []byte {
	t.Helper()

	samples := make([]float32, int(seconds*float64(sampleRate)))
	for i := range samples {
		samples[i] = float32(0.4 * math.Sin(2*math.Pi*220*float64(i)/float64(sampleRate)))
	}

	wav, err := audio.EncodeWAVPCM16(samples, sampleRate)
	if err != nil {
		t.Fatalf("encode prompt WAV: %v", err)
	}

	return wav
}

func TestNewEngine_GatedCheckpointLoadsVoiceEncoder(t *testing.T) {
	e := newSyntheticEngine(t, false)

	if !e.canClone() {
		t.Fatal("canClone() = false for a checkpoint with encoder weights")
	}

	cond, err := e.voice.Encode(make([]float32, 4000))
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	if got := cond.Shape(); !slices.Equal(got, []int64{1, 3, syntheticCondDim}) {
		t.Errorf("conditioning shape = %v, want [1 3 %d]", got, syntheticCondDim)
	}
}

func TestNewEngine_ZeroedEncoderLoadsWithoutCloning(t *testing.T) {
	e := newSyntheticEngine(t, true)

	if e.canClone() {
		t.Fatal("canClone() = true for a checkpoint with zeroed encoder weights")
	}

	_, _, err := e.cloneVoice(promptWAV(t, audio.ExpectedSampleRate, 1))
	if !errors.Is(err, errCloningUnavailable) {
		t.Fatalf("cloneVoice err = %v, want errCloningUnavailable", err)
	}

	for _, want := range []string{"kyutai/pocket-tts ", "locally"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("cloneVoice err = %q, want it to name %q", err, want)
		}
	}
}

// TestCloneVoice_AudioPromptEmbedding checks that a 16 kHz prompt is prepared
// like the CLI's (audio.PrepareVoicePrompt resamples it to 24 kHz) and that
// the result is the projected conditioning as an audio_prompt embedding.
func TestCloneVoice_AudioPromptEmbedding(t *testing.T) {
	e := newSyntheticEngine(t, false)

	const rate = 16000

	wav := promptWAV(t, rate, 1.5)

	got, gotFrames, err := e.cloneVoice(wav)
	if err != nil {
		t.Fatalf("cloneVoice: %v", err)
	}

	kind, err := safetensors.InspectVoiceFileBytes(got)
	if err != nil || kind == safetensors.VoiceFileModelState {
		t.Fatalf("voice file kind = %q (err %v), want a legacy audio_prompt embedding", kind, err)
	}

	store, err := safetensors.OpenStoreFromBytes(got, safetensors.StoreOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if names := store.Names(); !slices.Equal(names, []string{"audio_prompt"}) {
		t.Errorf("tensors = %v, want [audio_prompt]", names)
	}

	store.Close()

	data, shape, err := safetensors.LoadVoiceEmbeddingFromBytes(got)
	if err != nil {
		t.Fatalf("LoadVoiceEmbeddingFromBytes: %v", err)
	}

	samples, sampleRate, err := audio.DecodePromptWAV(wav)
	if err != nil {
		t.Fatal(err)
	}

	prepared, err := audio.PrepareVoicePrompt(samples, sampleRate)
	if err != nil {
		t.Fatal(err)
	}

	// Mimi frames are 1920 samples at 24 kHz. Resampled, the 1.5 s prompt
	// spans at least 19 frames, the raw 16 kHz samples only 13.
	frames := int64((len(prepared) + 1919) / 1920)
	if frames < 19 {
		t.Fatalf("prepared prompt has %d frames, want at least 19", frames)
	}

	if !slices.Equal(shape, []int64{1, frames, syntheticCondDim}) {
		t.Fatalf("embedding shape = %v, want [1 %d %d]", shape, frames, syntheticCondDim)
	}

	if gotFrames != frames {
		t.Errorf("cloneVoice frames = %d, want %d", gotFrames, frames)
	}

	want, err := e.voice.Encode(prepared)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(data, want.RawData()) {
		t.Error("embedding differs from the conditioning of the prepared prompt")
	}
}

func TestCloneVoice_RejectsBadWAV(t *testing.T) {
	e := newSyntheticEngine(t, false)

	_, _, err := e.cloneVoice([]byte("not a wav"))
	if err == nil || !strings.Contains(err.Error(), "WAV") {
		t.Fatalf("cloneVoice err = %v, want a WAV decode error", err)
	}
}

// generateSteps runs a few generation steps of text on e, with voice as the
// audio prompt when it is not nil, and returns the samples.
func generateSteps(t *testing.T, e *nativeEngine, input string, voice *tts.VoiceEmbedding) []float32 {
	t.Helper()

	ids, err := e.tokenizer.Encode(input)
	if err != nil {
		t.Fatal(err)
	}

	pcm, err := e.runtime.GenerateAudio(context.Background(), ids, tts.RuntimeGenerateConfig{
		Temperature:        0.3,
		EOSThreshold:       -4,
		MaxSteps:           8,
		SamplerDecodeSteps: 1,
		MimiStepsPerLatent: 16,
		VoiceEmbedding:     voice,
	})
	if err != nil {
		t.Fatalf("GenerateAudio: %v", err)
	}

	if len(pcm) == 0 {
		t.Fatal("GenerateAudio produced no samples")
	}

	return pcm
}

// loadGermanEngine builds the engine from the german checkpoint in dir,
// skipping when it is not downloaded.
func loadGermanEngine(t *testing.T, dir string) *nativeEngine {
	t.Helper()

	modelBytes, err := os.ReadFile(dir + "model.safetensors")
	if err != nil {
		t.Skipf("checkpoint not downloaded: %v", err)
	}

	tokBytes, err := os.ReadFile(dir + "tokenizer.json")
	if err != nil {
		t.Skipf("tokenizer not downloaded: %v", err)
	}

	e, err := newEngine(modelBytes, tokBytes, "german", nil)
	if err != nil {
		t.Fatalf("newEngine: %v", err)
	}

	t.Cleanup(e.runtime.Close)

	return e
}

// TestNewEngine_UngatedGermanSynthesizesWithoutCloning loads the ungated
// download (pockettts model download --language german).
func TestNewEngine_UngatedGermanSynthesizesWithoutCloning(t *testing.T) {
	e := loadGermanEngine(t, "../../models/german/")

	if e.canClone() {
		t.Fatal("canClone() = true for the ungated checkpoint")
	}

	generateSteps(t, e, "Hallo Welt.", nil)
}

// TestNewEngine_GatedGermanClonesVoice loads the gated download
// (pockettts model download --language german --hf-token ... --out-dir
// models/gated/german), clones the parity prompt and synthesizes with it.
func TestNewEngine_GatedGermanClonesVoice(t *testing.T) {
	e := loadGermanEngine(t, "../../models/gated/german/")

	if !e.canClone() {
		t.Fatal("canClone() = false for the gated checkpoint")
	}

	wav, err := os.ReadFile("../../internal/native/testdata/python_parity/voice_prompt.wav")
	if err != nil {
		t.Fatal(err)
	}

	blob, _, err := e.cloneVoice(wav)
	if err != nil {
		t.Fatalf("cloneVoice: %v", err)
	}

	data, shape, err := safetensors.LoadVoiceEmbeddingFromBytes(blob)
	if err != nil {
		t.Fatalf("LoadVoiceEmbeddingFromBytes: %v", err)
	}

	if len(shape) != 3 || shape[0] != 1 || shape[1] == 0 || shape[2] != 1024 {
		t.Fatalf("embedding shape = %v, want [1 T 1024]", shape)
	}

	generateSteps(t, e, "Hallo Welt.", &tts.VoiceEmbedding{Data: data, Shape: shape})
}
