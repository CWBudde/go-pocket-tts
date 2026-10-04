package native

import (
	"errors"
	"fmt"

	"github.com/cwbudde/go-pocket-tts/internal/runtime/ops"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

// ErrMimiEncoderWeightsZeroed reports a checkpoint from the ungated
// kyutai/pocket-tts-without-voice-cloning repo: its Mimi encoder weights are
// zero, so it would encode every prompt to the same latent.
var ErrMimiEncoderWeightsZeroed = errors.New("native: the checkpoint's Mimi encoder weights are zeroed " +
	"(ungated kyutai/pocket-tts-without-voice-cloning); cloning a voice from audio needs the gated " +
	"kyutai/pocket-tts weights: accept the terms on Hugging Face, then run pockettts model download " +
	"with --hf-token or HF_TOKEN")

// MimiEncoder ports upstream MimiModel.encode_to_latent: SEANet encoder,
// encoder transformer and the downsample conv to the 12.5 Hz frame rate.
type MimiEncoder struct {
	cfg MimiConfig

	initConv  *conv1dLayer // encoder.model.0
	res1      *seanetResBlock
	down1     *conv1dLayer // stride 4
	res2      *seanetResBlock
	down2     *conv1dLayer // stride 5
	res3      *seanetResBlock
	down3     *conv1dLayer // stride 6
	finalConv *conv1dLayer // encoder.model.11

	transformer *mimiDecoderTransformer
	downsample  *conv1dLayer // pad_mode="replicate"
}

// LoadMimiEncoder loads mimi.encoder, mimi.encoder_transformer and
// mimi.downsample. The SEANet layout is the same in every upstream config
// (ratios [6, 5, 4], reversed in the encoder); mimi.inner_dim only sets the
// downsample conv's output channels, which are read from its weight. It fails
// with ErrMimiEncoderWeightsZeroed when every SEANet weight is zero.
func LoadMimiEncoder(vb *VarBuilder, cfg MimiConfig) (*MimiEncoder, error) {
	cfg = cfg.withDefaults()
	mimi := vb.Path("mimi")
	seanet := mimi.Path("encoder", "model")

	m := &MimiEncoder{cfg: cfg}

	convs := []struct {
		dst    **conv1dLayer
		idx    string
		stride int64
	}{
		{&m.initConv, "0", 1},
		{&m.down1, "3", 4},
		{&m.down2, "6", 5},
		{&m.down3, "9", 6},
		{&m.finalConv, "11", 1},
	}
	for _, c := range convs {
		conv, err := loadConv1D(seanet.Path(c.idx, "conv"), c.stride, true)
		if err != nil {
			return nil, fmt.Errorf("native: mimi encoder: %w", err)
		}

		*c.dst = conv
	}

	blocks := []struct {
		dst **seanetResBlock
		idx string
	}{
		{&m.res1, "1"},
		{&m.res2, "4"},
		{&m.res3, "7"},
	}
	for _, b := range blocks {
		block, err := loadSEANetResBlock(seanet.Path(b.idx))
		if err != nil {
			return nil, fmt.Errorf("native: mimi encoder: %w", err)
		}

		*b.dst = block
	}

	if m.seanetWeightsZero() {
		return nil, ErrMimiEncoderWeightsZeroed
	}

	transformer, err := loadMimiTransformer(mimi, "encoder_transformer", cfg)
	if err != nil {
		return nil, err
	}

	m.transformer = transformer

	m.downsample, err = loadConv1D(mimi.Path("downsample", "conv", "conv"), int64(cfg.MimiStepsPerLatent()), false)
	if err != nil {
		return nil, fmt.Errorf("native: mimi encoder: %w", err)
	}

	return m, nil
}

// InnerDim is the latent width (mimi.inner_dim): 512 for english_2026-01, 32
// for the newer configs.
func (m *MimiEncoder) InnerDim() int64 { return m.downsample.weight.Shape()[0] }

// EncodeToLatent maps mono 24 kHz audio [B, 1, N] to the unquantized latent
// [B, T, inner_dim] with T = ceil(N / frameSize). Like upstream, the audio is
// right-padded with zeros to whole frames and every conv starts from zero
// state (causal left padding).
func (m *MimiEncoder) EncodeToLatent(audio *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil {
		return nil, errors.New("native: mimi encoder is nil")
	}

	shape := audio.Shape()
	if len(shape) != 3 || shape[1] != 1 || shape[2] == 0 {
		return nil, fmt.Errorf("native: mimi encoder expects audio [B,1,N] with N > 0, got %v", shape)
	}

	frame := m.frameSize()
	frames := (shape[2] + frame - 1) / frame

	encoderSteps := frames * int64(m.cfg.MimiStepsPerLatent())
	if encoderSteps > mimiRoPEPositions {
		return nil, fmt.Errorf("native: mimi encoder: %d samples exceed the %d supported encoder steps", shape[2], mimiRoPEPositions)
	}

	x, err := padRight(audio, frames*frame)
	if err != nil {
		return nil, err
	}

	x, err = m.seanet(x)
	if err != nil {
		return nil, err
	}

	x, err = m.transformer.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("native: mimi encoder transformer: %w", err)
	}

	x, err = m.downsample.forwardReplicateLeftPad(x)
	if err != nil {
		return nil, fmt.Errorf("native: mimi downsample: %w", err)
	}

	return x.Transpose(1, 2)
}

// seanet runs SEANetEncoder: [B, 1, N] -> [B, 512, N/120].
func (m *MimiEncoder) seanet(x *tensor.Tensor) (*tensor.Tensor, error) {
	steps := []func(*tensor.Tensor) (*tensor.Tensor, error){
		m.initConv.forwardStreamingOnce,
		m.res1.Forward,
		elu,
		m.down1.forwardStreamingOnce,
		m.res2.Forward,
		elu,
		m.down2.forwardStreamingOnce,
		m.res3.Forward,
		elu,
		m.down3.forwardStreamingOnce,
		elu,
		m.finalConv.forwardStreamingOnce,
	}

	var err error
	for i, step := range steps {
		x, err = step(x)
		if err != nil {
			return nil, fmt.Errorf("native: mimi encoder step %d: %w", i, err)
		}
	}

	return x, nil
}

func (m *MimiEncoder) seanetWeightsZero() bool {
	ws := []*tensor.Tensor{
		m.initConv.weight, m.down1.weight, m.down2.weight, m.down3.weight, m.finalConv.weight,
		m.res1.conv1.weight, m.res1.conv2.weight, m.res2.conv1.weight, m.res2.conv2.weight,
		m.res3.conv1.weight, m.res3.conv2.weight,
	}
	for _, w := range ws {
		for _, v := range w.RawData() {
			if v != 0 {
				return false
			}
		}
	}

	return true
}

// frameSize is the number of samples per latent frame (1920 at 24 kHz and
// 12.5 Hz).
func (m *MimiEncoder) frameSize() int64 {
	return int64(float64(m.cfg.SampleRate) / m.cfg.FrameRate)
}

func elu(x *tensor.Tensor) (*tensor.Tensor, error) { return eluTensorInPlace(x), nil }

// padRight zero-pads the last dim of x [B, C, T] to n.
func padRight(x *tensor.Tensor, n int64) (*tensor.Tensor, error) {
	shape := x.Shape()

	t := shape[len(shape)-1]
	if n == t {
		return x, nil
	}

	rows := int64(x.ElemCount()) / t
	src := x.RawData()
	out := make([]float32, rows*n)

	for r := range rows {
		copy(out[r*n:r*n+t], src[r*t:(r+1)*t])
	}

	shape[len(shape)-1] = n

	return tensor.New(out, shape)
}

// forwardReplicateLeftPad is forwardStreamingOnce with upstream's
// pad_mode="replicate": the causal left padding repeats the first frame
// instead of being zero.
func (c *conv1dLayer) forwardReplicateLeftPad(x *tensor.Tensor) (*tensor.Tensor, error) {
	shape := x.Shape()
	if len(shape) != 3 || shape[2] == 0 {
		return nil, fmt.Errorf("native: replicate-padded conv expects [B,C,T] with T > 0, got %v", shape)
	}

	effKernel := (c.weight.Shape()[2]-1)*c.dilation + 1
	pad := max(effKernel-c.stride, 0)

	rows, t := shape[0]*shape[1], shape[2]
	src := x.RawData()
	out := make([]float32, rows*(t+pad))

	for r := range rows {
		row := src[r*t : (r+1)*t]
		dst := out[r*(t+pad) : (r+1)*(t+pad)]

		for i := range pad {
			dst[i] = row[0]
		}

		copy(dst[pad:], row)
	}

	padded, err := tensor.New(out, []int64{shape[0], shape[1], t + pad})
	if err != nil {
		return nil, err
	}

	return ops.Conv1D(padded, c.weight, c.bias, c.stride, 0, c.dilation, c.groups)
}

// VoiceEncoder turns a voice prompt into the FlowLM audio conditioning, like
// upstream TTSModel._encode_audio: the Mimi latent projected by
// flow_lm.speaker_proj_weight. The prompt must already be prepared
// (audio.PrepareVoicePrompt); the BOS before the voice is not included.
type VoiceEncoder struct {
	mimi        *MimiEncoder
	speakerProj *tensor.Tensor // [d_model, inner_dim]
}

// LoadVoiceEncoder loads the Mimi encoder and the speaker projection.
func LoadVoiceEncoder(vb *VarBuilder, cfg MimiConfig) (*VoiceEncoder, error) {
	mimi, err := LoadMimiEncoder(vb, cfg)
	if err != nil {
		return nil, err
	}

	proj, err := vb.Tensor("flow_lm.speaker_proj_weight")
	if err != nil {
		return nil, fmt.Errorf("native: voice encoder: %w", err)
	}

	if shape := proj.Shape(); len(shape) != 2 || shape[1] != mimi.InnerDim() {
		return nil, fmt.Errorf("native: flow_lm.speaker_proj_weight shape %v, want [d_model, %d] for the encoder's inner_dim",
			shape, mimi.InnerDim())
	}

	return &VoiceEncoder{mimi: mimi, speakerProj: proj}, nil
}

// LoadVoiceEncoderFromSafetensors opens path and loads only the voice-encoder
// tensors from it.
func LoadVoiceEncoderFromSafetensors(path string, cfg MimiConfig) (*VoiceEncoder, error) {
	store, err := safetensors.OpenStore(path, safetensors.StoreOptions{})
	if err != nil {
		return nil, err
	}
	defer store.Close()

	return LoadVoiceEncoder(NewVarBuilder(store), cfg)
}

// EncodeToLatent encodes prepared 24 kHz mono samples to the Mimi latent
// [1, T, inner_dim].
func (e *VoiceEncoder) EncodeToLatent(samples []float32) (*tensor.Tensor, error) {
	if len(samples) == 0 {
		return nil, errors.New("native: voice prompt is empty")
	}

	audio, err := tensor.New(samples, []int64{1, 1, int64(len(samples))})
	if err != nil {
		return nil, err
	}

	return e.mimi.EncodeToLatent(audio)
}

// Condition projects a latent [1, T, inner_dim] to the conditioning
// [1, T, d_model] (F.linear without bias).
func (e *VoiceEncoder) Condition(latent *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.Linear(latent, e.speakerProj, nil)
}

// Encode returns the conditioning [1, T, d_model] for prepared samples.
func (e *VoiceEncoder) Encode(samples []float32) (*tensor.Tensor, error) {
	latent, err := e.EncodeToLatent(samples)
	if err != nil {
		return nil, err
	}

	return e.Condition(latent)
}
