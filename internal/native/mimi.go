package native

import (
	"errors"
	"fmt"
	"math"
	"strconv"

	"github.com/example/go-pocket-tts/internal/runtime/ops"
	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

var ErrMimiEncoderNotImplemented = errors.New("native mimi encoder is not implemented")

type MimiConfig struct {
	SampleRate int64
	NumHeads   int64
	MaxPeriod  float64
}

func DefaultMimiConfig() MimiConfig {
	return MimiConfig{SampleRate: 24000, NumHeads: 8, MaxPeriod: 10000.0}
}

type conv1dLayer struct {
	weight   *tensor.Tensor
	bias     *tensor.Tensor
	stride   int64
	dilation int64
	groups   int64
}

func loadConv1D(vb *VarBuilder, withBias bool) (*conv1dLayer, error) {
	w, err := vb.Tensor("weight")
	if err != nil {
		return nil, err
	}

	if len(w.Shape()) != 3 {
		return nil, fmt.Errorf("native: conv1d weight must be rank-3, got %v", w.Shape())
	}
	var b *tensor.Tensor

	if withBias {
		t, ok, err := vb.TensorMaybe("bias")
		if err != nil {
			return nil, err
		}

		if ok {
			b = t
		}
	}

	return &conv1dLayer{weight: w, bias: b, stride: 1, dilation: 1, groups: 1}, nil
}

func (c *conv1dLayer) forwardStreamingOnce(x *tensor.Tensor) (*tensor.Tensor, error) {
	k := c.weight.Shape()[2]
	effKernel := (k-1)*c.dilation + 1

	leftPad := max(effKernel-c.stride, 0)

	return ops.Conv1DLeftPad(x, c.weight, c.bias, c.stride, leftPad, c.dilation, c.groups)
}

type convTr1dLayer struct {
	weight  *tensor.Tensor
	bias    *tensor.Tensor
	stride  int64
	groups  int64
	kernelT []float32 // pre-packed [kSize, outCh, inCh] for groups=1; nil otherwise
}

func loadConvTr1D(vb *VarBuilder, stride, groups int64, withBias bool) (*convTr1dLayer, error) {
	w, err := vb.Tensor("weight")
	if err != nil {
		return nil, err
	}

	if len(w.Shape()) != 3 {
		return nil, fmt.Errorf("native: convtranspose1d weight must be rank-3, got %v", w.Shape())
	}
	var b *tensor.Tensor

	if withBias {
		t, ok, err := vb.TensorMaybe("bias")
		if err != nil {
			return nil, err
		}

		if ok {
			b = t
		}
	}
	// Pre-pack the kernel for groups=1 to avoid per-call repack overhead.
	var kernelT []float32
	if groups == 1 {
		kernelT = ops.RepackConvTransposeKernel(w)
	}

	return &convTr1dLayer{weight: w, bias: b, stride: stride, groups: groups, kernelT: kernelT}, nil
}

func (c *convTr1dLayer) forwardStreamingOnce(x *tensor.Tensor) (*tensor.Tensor, error) {
	k := c.weight.Shape()[2]

	pt := k - c.stride
	if c.kernelT != nil {
		return ops.ConvTranspose1DPrePackedRightTrim(x, c.weight, c.bias, c.kernelT, c.stride, 0, 0, 1, c.groups, pt)
	}

	return ops.ConvTranspose1DRightTrim(x, c.weight, c.bias, c.stride, 0, 0, 1, c.groups, pt)
}

type seanetResBlock struct {
	conv1 *conv1dLayer
	conv2 *conv1dLayer
}

func loadSEANetResBlock(vb *VarBuilder) (*seanetResBlock, error) {
	conv1, err := loadConv1D(vb.Path("block", "1", "conv"), true)
	if err != nil {
		return nil, err
	}

	conv2, err := loadConv1D(vb.Path("block", "3", "conv"), true)
	if err != nil {
		return nil, err
	}

	return &seanetResBlock{conv1: conv1, conv2: conv2}, nil
}

func (rb *seanetResBlock) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h := eluTensor(x)
	var err error

	h, err = rb.conv1.forwardStreamingOnce(h)
	if err != nil {
		return nil, err
	}

	h = eluTensor(h)

	h, err = rb.conv2.forwardStreamingOnce(h)
	if err != nil {
		return nil, err
	}

	return addSameShape(x, h)
}

type mimiTransformerLayer struct {
	norm1       *LayerNorm
	norm2       *LayerNorm
	inProj      *Linear
	outProj     *Linear
	linear1     *Linear
	linear2     *Linear
	layerScale1 *tensor.Tensor // optional [d]
	layerScale2 *tensor.Tensor // optional [d]
	nHeads      int64
	headDim     int64
}

func loadMimiTransformerLayer(vb *VarBuilder, nHeads int64) (*mimiTransformerLayer, error) {
	norm1, err := loadLayerNorm(vb, "norm1", 1e-5)
	if err != nil {
		return nil, err
	}

	norm2, err := loadLayerNorm(vb, "norm2", 1e-5)
	if err != nil {
		return nil, err
	}

	inProj, err := loadLinear(vb, "self_attn.in_proj", false)
	if err != nil {
		return nil, err
	}

	outProj, err := loadLinear(vb, "self_attn.out_proj", false)
	if err != nil {
		return nil, err
	}

	linear1, err := loadLinear(vb, "linear1", false)
	if err != nil {
		return nil, err
	}

	linear2, err := loadLinear(vb, "linear2", false)
	if err != nil {
		return nil, err
	}

	ls1, _, err := vb.TensorMaybe("layer_scale_1.scale")
	if err != nil {
		return nil, err
	}

	ls2, _, err := vb.TensorMaybe("layer_scale_2.scale")
	if err != nil {
		return nil, err
	}

	dModel := outProj.Weight.Shape()[0]
	if dModel%nHeads != 0 {
		return nil, fmt.Errorf("native: mimi d_model %d not divisible by heads %d", dModel, nHeads)
	}

	return &mimiTransformerLayer{
		norm1:       norm1,
		norm2:       norm2,
		inProj:      inProj,
		outProj:     outProj,
		linear1:     linear1,
		linear2:     linear2,
		layerScale1: ls1,
		layerScale2: ls2,
		nHeads:      nHeads,
		headDim:     dModel / nHeads,
	}, nil
}

func (l *mimiTransformerLayer) forward(x, ropeCos, ropeSin *tensor.Tensor) (*tensor.Tensor, error) {
	n1, err := l.norm1.Forward(x)
	if err != nil {
		return nil, err
	}

	attn, err := l.selfAttention(n1, ropeCos, ropeSin)
	if err != nil {
		return nil, err
	}

	if l.layerScale1 != nil {
		attn, err = tensor.BroadcastMul(attn, l.layerScale1)
		if err != nil {
			return nil, err
		}
	}

	x, err = addSameShape(x, attn)
	if err != nil {
		return nil, err
	}

	n2, err := l.norm2.Forward(x)
	if err != nil {
		return nil, err
	}

	ff, err := l.linear1.Forward(n2)
	if err != nil {
		return nil, err
	}

	ff = geluErfTensor(ff)

	ff, err = l.linear2.Forward(ff)
	if err != nil {
		return nil, err
	}

	if l.layerScale2 != nil {
		ff, err = tensor.BroadcastMul(ff, l.layerScale2)
		if err != nil {
			return nil, err
		}
	}

	return addSameShape(x, ff)
}

func (l *mimiTransformerLayer) selfAttention(x, ropeCos, ropeSin *tensor.Tensor) (*tensor.Tensor, error) {
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("native: mimi selfAttention expects [B,T,D], got %v", shape)
	}

	b, t, d := shape[0], shape[1], shape[2]

	qkv, err := l.inProj.Forward(x)
	if err != nil {
		return nil, err
	}

	q, k, v, err := splitLastDim3(qkv)
	if err != nil {
		return nil, err
	}

	q, _ = q.Reshape([]int64{b, t, l.nHeads, l.headDim})
	k, _ = k.Reshape([]int64{b, t, l.nHeads, l.headDim})
	v, _ = v.Reshape([]int64{b, t, l.nHeads, l.headDim})
	q, _ = q.Transpose(1, 2)
	k, _ = k.Transpose(1, 2)
	v, _ = v.Transpose(1, 2)

	q, err = ops.RoPE(q, ropeCos, ropeSin, 0)
	if err != nil {
		return nil, err
	}

	k, err = ops.RoPE(k, ropeCos, ropeSin, 0)
	if err != nil {
		return nil, err
	}

	a, err := ops.Attention(q, k, v, true, 0)
	if err != nil {
		return nil, err
	}

	a, _ = a.Transpose(1, 2)
	a, _ = a.Reshape([]int64{b, t, d})

	return l.outProj.Forward(a)
}

type mimiDecoderTransformer struct {
	layers []*mimiTransformerLayer
	cos    *tensor.Tensor
	sin    *tensor.Tensor
}

func loadMimiDecoderTransformer(vb *VarBuilder, cfg MimiConfig) (*mimiDecoderTransformer, error) {
	layers := make([]*mimiTransformerLayer, 0, 4)

	for i := 0; ; i++ {
		layerPath := vb.Path("decoder_transformer", "transformer", "layers", strconv.Itoa(i))
		if !layerPath.Has("norm1.weight") {
			break
		}

		layer, err := loadMimiTransformerLayer(layerPath, cfg.NumHeads)
		if err != nil {
			return nil, fmt.Errorf("native: load mimi transformer layer %d: %w", i, err)
		}

		layers = append(layers, layer)
	}

	if len(layers) == 0 {
		return nil, errors.New("native: no mimi decoder transformer layers found")
	}

	cos, sin, err := buildRoPE(8192, layers[0].headDim, cfg.MaxPeriod)
	if err != nil {
		return nil, err
	}

	return &mimiDecoderTransformer{layers: layers, cos: cos, sin: sin}, nil
}

func (mt *mimiDecoderTransformer) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// [B, C, T] -> [B, T, C]
	x, err := x.Transpose(1, 2)
	if err != nil {
		return nil, err
	}

	for i, layer := range mt.layers {
		x, err = layer.forward(x, mt.cos, mt.sin)
		if err != nil {
			return nil, fmt.Errorf("native: mimi decoder transformer layer %d: %w", i, err)
		}
	}

	return x.Transpose(1, 2)
}

// MimiModel rebuilds latent_to_mimi + mimi_decoder modules.
type MimiModel struct {
	cfg MimiConfig

	quantizerOutProj *conv1dLayer // mimi.quantizer.output_proj
	upsample         *convTr1dLayer
	transformer      *mimiDecoderTransformer

	initConv  *conv1dLayer
	up1       *convTr1dLayer
	res1      *seanetResBlock
	up2       *convTr1dLayer
	res2      *seanetResBlock
	up3       *convTr1dLayer
	res3      *seanetResBlock
	finalConv *conv1dLayer
}

func LoadMimiModel(vb *VarBuilder, cfg MimiConfig) (*MimiModel, error) {
	if cfg.SampleRate == 0 {
		cfg = DefaultMimiConfig()
	}

	mimi := vb.Path("mimi")

	quant, err := loadConv1D(mimi.Path("quantizer", "output_proj"), false)
	if err != nil {
		return nil, err
	}

	upsample, err := loadConvTr1D(mimi.Path("upsample", "convtr", "convtr"), 16, 512, false)
	if err != nil {
		return nil, err
	}

	transformer, err := loadMimiDecoderTransformer(mimi, cfg)
	if err != nil {
		return nil, err
	}

	initConv, err := loadConv1D(mimi.Path("decoder", "model", "0", "conv"), true)
	if err != nil {
		return nil, err
	}

	up1, err := loadConvTr1D(mimi.Path("decoder", "model", "2", "convtr"), 6, 1, true)
	if err != nil {
		return nil, err
	}

	res1, err := loadSEANetResBlock(mimi.Path("decoder", "model", "3"))
	if err != nil {
		return nil, err
	}

	up2, err := loadConvTr1D(mimi.Path("decoder", "model", "5", "convtr"), 5, 1, true)
	if err != nil {
		return nil, err
	}

	res2, err := loadSEANetResBlock(mimi.Path("decoder", "model", "6"))
	if err != nil {
		return nil, err
	}

	up3, err := loadConvTr1D(mimi.Path("decoder", "model", "8", "convtr"), 4, 1, true)
	if err != nil {
		return nil, err
	}

	res3, err := loadSEANetResBlock(mimi.Path("decoder", "model", "9"))
	if err != nil {
		return nil, err
	}

	finalConv, err := loadConv1D(mimi.Path("decoder", "model", "11", "conv"), true)
	if err != nil {
		return nil, err
	}

	return &MimiModel{
		cfg:              cfg,
		quantizerOutProj: quant,
		upsample:         upsample,
		transformer:      transformer,
		initConv:         initConv,
		up1:              up1,
		res1:             res1,
		up2:              up2,
		res2:             res2,
		up3:              up3,
		res3:             res3,
		finalConv:        finalConv,
	}, nil
}

func (m *MimiModel) SampleRate() int64 { return m.cfg.SampleRate }

// QuantizerProject maps [B, 32, T] -> [B, 512, T].
func (m *MimiModel) QuantizerProject(latentBCT *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil || m.quantizerOutProj == nil {
		return nil, errors.New("native: mimi model is not initialized")
	}

	return ops.Conv1D(latentBCT, m.quantizerOutProj.weight, m.quantizerOutProj.bias, 1, 0, 1, 1)
}

// DecodeFromLatent maps [B, 512, T] to PCM-like output [B, 1, N].
func (m *MimiModel) DecodeFromLatent(latent *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil {
		return nil, errors.New("native: mimi model is nil")
	}
	var err error
	x := latent

	x, err = m.upsample.forwardStreamingOnce(x)
	if err != nil {
		return nil, err
	}

	x, err = m.transformer.Forward(x)
	if err != nil {
		return nil, err
	}

	x, err = m.initConv.forwardStreamingOnce(x)
	if err != nil {
		return nil, err
	}

	x = eluTensor(x)

	x, err = m.up1.forwardStreamingOnce(x)
	if err != nil {
		return nil, err
	}

	x, err = m.res1.Forward(x)
	if err != nil {
		return nil, err
	}

	x = eluTensor(x)

	x, err = m.up2.forwardStreamingOnce(x)
	if err != nil {
		return nil, err
	}

	x, err = m.res2.Forward(x)
	if err != nil {
		return nil, err
	}

	x = eluTensor(x)

	x, err = m.up3.forwardStreamingOnce(x)
	if err != nil {
		return nil, err
	}

	x, err = m.res3.Forward(x)
	if err != nil {
		return nil, err
	}

	x = eluTensor(x)

	x, err = m.finalConv.forwardStreamingOnce(x)
	if err != nil {
		return nil, err
	}

	return x, nil
}

// EncodeToLatent is intentionally left as a hook for Phase 20 voice encoding.
func (m *MimiModel) EncodeToLatent(_ *tensor.Tensor) (*tensor.Tensor, error) {
	return nil, ErrMimiEncoderNotImplemented
}

func cosf(x float32) float64 { return math.Cos(float64(x)) }
func sinf(x float32) float64 { return math.Sin(float64(x)) }
