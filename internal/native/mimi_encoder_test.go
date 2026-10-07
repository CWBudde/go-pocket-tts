package native

import (
	"errors"
	"math"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

// syntheticEncoderTensors returns the encoder-side tensors of a scaled-down
// checkpoint: SEANet channels nf, 2nf, 4nf, 8nf, a transformer of width 8nf
// (8 heads), a downsample conv to innerDim and a speaker projection to
// condDim. Values are small and deterministic; zero makes every one 0.
func syntheticEncoderTensors(nf, innerDim, condDim int64, zero bool) []safetensors.Tensor {
	d := 8 * nf
	ff := 2 * d
	seed := 0

	mk := func(name string, shape ...int64) safetensors.Tensor {
		n := int64(1)
		for _, s := range shape {
			n *= s
		}

		data := make([]float32, n)
		if !zero {
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
		mk("mimi.downsample.conv.conv.weight", innerDim, d, 32),
		mk("flow_lm.speaker_proj_weight", condDim, innerDim),
	)

	return ts
}

func writeSyntheticEncoder(t *testing.T, tensors []safetensors.Tensor) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "encoder.safetensors")

	err := safetensors.WriteFile(path, tensors)
	if err != nil {
		t.Fatalf("write synthetic encoder: %v", err)
	}

	return path
}

// Latent and conditioning widths of the synthetic encoder.
const syntheticInnerDim, syntheticCondDim = 3, 6

func loadSyntheticVoiceEncoder(t *testing.T) *VoiceEncoder {
	t.Helper()

	path := writeSyntheticEncoder(t, syntheticEncoderTensors(2, syntheticInnerDim, syntheticCondDim, false))

	enc, err := LoadVoiceEncoderFromSafetensors(path, DefaultMimiConfig())
	if err != nil {
		t.Fatalf("load voice encoder: %v", err)
	}

	return enc
}

func testSamples(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(0.5 * math.Sin(float64(i)*0.013))
	}

	return s
}

func TestMimiEncoder_FrameCountAndShape(t *testing.T) {
	t.Parallel()

	enc := loadSyntheticVoiceEncoder(t)

	// Upstream pads the prompt to whole 1920-sample frames (24 kHz / 12.5 Hz).
	for _, tc := range []struct{ samples, frames int64 }{
		{1, 1}, {1920, 1}, {1921, 2}, {5000, 3},
	} {
		latent, err := enc.EncodeToLatent(testSamples(int(tc.samples)))
		if err != nil {
			t.Fatalf("%d samples: %v", tc.samples, err)
		}

		if got, want := latent.Shape(), []int64{1, tc.frames, syntheticInnerDim}; !equalShape(got, want) {
			t.Errorf("%d samples: latent shape %v, want %v", tc.samples, got, want)
		}
	}
}

func TestMimiEncoder_RejectsEmptyPrompt(t *testing.T) {
	t.Parallel()

	enc := loadSyntheticVoiceEncoder(t)

	_, err := enc.EncodeToLatent(nil)
	if err == nil {
		t.Fatal("EncodeToLatent(nil) succeeded")
	}
}

func TestLoadMimiEncoder_ConvStrides(t *testing.T) {
	t.Parallel()

	enc := loadSyntheticVoiceEncoder(t)
	m := enc.mimi

	// SEANet ratios [6, 5, 4] run reversed in the encoder; the downsample
	// conv goes from 200 Hz to 12.5 Hz.
	got := []int64{
		m.initConv.stride, m.down1.stride, m.down2.stride, m.down3.stride,
		m.finalConv.stride, m.downsample.stride,
	}
	want := []int64{1, 4, 5, 6, 1, 16}

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("conv strides %v, want %v", got, want)
		}
	}
}

func TestConv1DReplicateLeftPad(t *testing.T) {
	t.Parallel()

	// A kernel that picks the first tap shows the padded columns.
	w, _ := tensor.New([]float32{1, 0, 0}, []int64{1, 1, 3})
	x, _ := tensor.New([]float32{5, 6, 7}, []int64{1, 1, 3})
	conv := &conv1dLayer{weight: w, stride: 1, dilation: 1, groups: 1}

	out, err := conv.forwardReplicateLeftPad(x)
	if err != nil {
		t.Fatalf("forwardReplicateLeftPad: %v", err)
	}

	// Upstream pad_mode="replicate": copies of the first frame, not zeros.
	got, want := out.RawData(), []float32{5, 5, 5}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("output %v, want %v", got, want)
		}
	}
}

func TestVoiceEncoder_ProjectsLatentWithSpeakerProj(t *testing.T) {
	t.Parallel()

	const innerDim, condDim = syntheticInnerDim, syntheticCondDim

	enc := loadSyntheticVoiceEncoder(t)
	samples := testSamples(4000)

	latent, err := enc.EncodeToLatent(samples)
	if err != nil {
		t.Fatalf("EncodeToLatent: %v", err)
	}

	cond, err := enc.Encode(samples)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	frames := latent.Shape()[1]
	if got, want := cond.Shape(), []int64{1, frames, condDim}; !equalShape(got, want) {
		t.Fatalf("conditioning shape %v, want %v", got, want)
	}

	// _encode_audio: F.linear(latent, speaker_proj_weight), no bias.
	w := enc.speakerProj.RawData()
	l, c := latent.RawData(), cond.RawData()

	for f := range frames {
		for o := range int64(condDim) {
			var want float64
			for i := range int64(innerDim) {
				want += float64(l[f*innerDim+i]) * float64(w[o*innerDim+i])
			}

			if got := float64(c[f*condDim+o]); math.Abs(got-want) > 1e-5 {
				t.Fatalf("conditioning[%d,%d] = %g, want %g", f, o, got, want)
			}
		}
	}
}

func TestLoadVoiceEncoder_ZeroedEncoderWeights(t *testing.T) {
	t.Parallel()

	tensors := syntheticEncoderTensors(2, syntheticInnerDim, syntheticCondDim, true)
	// The ungated checkpoints keep a real downsample and speaker projection.
	for i := range tensors {
		if tensors[i].Name == "mimi.downsample.conv.conv.weight" || tensors[i].Name == "flow_lm.speaker_proj_weight" {
			for j := range tensors[i].Data {
				tensors[i].Data[j] = 0.1
			}
		}
	}

	_, err := LoadVoiceEncoderFromSafetensors(writeSyntheticEncoder(t, tensors), DefaultMimiConfig())
	if !errors.Is(err, ErrMimiEncoderWeightsZeroed) {
		t.Fatalf("err = %v, want %v", err, ErrMimiEncoderWeightsZeroed)
	}
}

func TestLoadVoiceEncoder_SpeakerProjMismatch(t *testing.T) {
	t.Parallel()

	tensors := syntheticEncoderTensors(2, syntheticInnerDim, syntheticCondDim, false)
	for i := range tensors {
		if tensors[i].Name == "flow_lm.speaker_proj_weight" {
			tensors[i].Shape = []int64{6, 4}
			tensors[i].Data = make([]float32, 24)
		}
	}

	_, err := LoadVoiceEncoderFromSafetensors(writeSyntheticEncoder(t, tensors), DefaultMimiConfig())
	if err == nil {
		t.Fatal("a [6,4] speaker projection for a 3-dim latent loaded")
	}
}
