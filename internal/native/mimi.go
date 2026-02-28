package native

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"sync"

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
	h := x.Clone()
	h = eluTensorInPlace(h)
	var err error

	h, err = rb.conv1.forwardStreamingOnce(h)
	if err != nil {
		return nil, err
	}

	h = eluTensorInPlace(h)

	h, err = rb.conv2.forwardStreamingOnce(h)
	if err != nil {
		return nil, err
	}

	return addSameShapeInPlace(x, h)
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
	return l.forwardWithScratch(x, ropeCos, ropeSin, nil)
}

func (l *mimiTransformerLayer) forwardWithScratch(x, ropeCos, ropeSin *tensor.Tensor, scratch *mimiDecodeScratch) (*tensor.Tensor, error) {
	var (
		n1  *tensor.Tensor
		err error
	)

	if scratch != nil {
		n1Out, ensureErr := scratch.ensure(&scratch.norm1, x.Shape())
		if ensureErr != nil {
			return nil, ensureErr
		}

		err = l.norm1.forwardIntoTrusted(x, n1Out)
		if err != nil {
			return nil, err
		}

		n1 = n1Out
	} else {
		n1, err = l.norm1.Forward(x)
		if err != nil {
			return nil, err
		}
	}

	attn, err := l.selfAttentionWithScratch(n1, ropeCos, ropeSin, scratch)
	if err != nil {
		return nil, err
	}

	if l.layerScale1 != nil {
		attn, err = mulLastDimInPlace(attn, l.layerScale1)
		if err != nil {
			return nil, err
		}
	}

	x, err = addSameShapeInPlace(x, attn)
	if err != nil {
		return nil, err
	}

	var n2 *tensor.Tensor
	if scratch != nil {
		n2Out, ensureErr := scratch.ensure(&scratch.norm2, x.Shape())
		if ensureErr != nil {
			return nil, ensureErr
		}

		err = l.norm2.forwardIntoTrusted(x, n2Out)
		if err != nil {
			return nil, err
		}

		n2 = n2Out
	} else {
		n2, err = l.norm2.Forward(x)
		if err != nil {
			return nil, err
		}
	}

	var ff *tensor.Tensor
	if scratch != nil {
		ff1Shape := append([]int64(nil), n2.Shape()...)
		ff1Shape[len(ff1Shape)-1] = l.linear1.outDim

		ff1Out, ensureErr := scratch.ensure(&scratch.ff1, ff1Shape)
		if ensureErr != nil {
			return nil, ensureErr
		}

		err = l.linear1.forwardIntoTrusted(n2, ff1Out)
		if err != nil {
			return nil, err
		}

		ff = ff1Out
	} else {
		ff, err = l.linear1.Forward(n2)
		if err != nil {
			return nil, err
		}
	}

	ff = geluErfTensorInPlace(ff)

	if scratch != nil {
		ff2Out, ensureErr := scratch.ensure(&scratch.ff2, x.Shape())
		if ensureErr != nil {
			return nil, ensureErr
		}

		err = l.linear2.forwardIntoTrusted(ff, ff2Out)
		if err != nil {
			return nil, err
		}

		ff = ff2Out
	} else {
		ff, err = l.linear2.Forward(ff)
		if err != nil {
			return nil, err
		}
	}

	if l.layerScale2 != nil {
		ff, err = mulLastDimInPlace(ff, l.layerScale2)
		if err != nil {
			return nil, err
		}
	}

	return addSameShapeInPlace(x, ff)
}

func (l *mimiTransformerLayer) selfAttention(x, ropeCos, ropeSin *tensor.Tensor) (*tensor.Tensor, error) {
	return l.selfAttentionWithScratch(x, ropeCos, ropeSin, nil)
}

func (l *mimiTransformerLayer) selfAttentionWithScratch(x, ropeCos, ropeSin *tensor.Tensor, scratch *mimiDecodeScratch) (*tensor.Tensor, error) {
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("native: mimi selfAttention expects [B,T,D], got %v", shape)
	}

	b, t, d := shape[0], shape[1], shape[2]

	var qkv *tensor.Tensor
	var err error
	if scratch != nil {
		qkvShape := []int64{b, t, l.inProj.outDim}
		qkvOut, ensureErr := scratch.ensure(&scratch.qkv, qkvShape)
		if ensureErr != nil {
			return nil, ensureErr
		}

		err = l.inProj.forwardIntoTrusted(x, qkvOut)
		if err != nil {
			return nil, err
		}

		qkv = qkvOut
	} else {
		qkv, err = l.inProj.Forward(x)
		if err != nil {
			return nil, err
		}
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

	if scratch != nil {
		attnOut, ensureErr := scratch.ensure(&scratch.attnOut, []int64{b, t, d})
		if ensureErr != nil {
			return nil, ensureErr
		}

		err = l.outProj.forwardIntoTrusted(a, attnOut)
		if err != nil {
			return nil, err
		}

		return attnOut, nil
	}

	return l.outProj.Forward(a)
}

type mimiDecoderTransformer struct {
	layers []*mimiTransformerLayer
	cos    *tensor.Tensor
	sin    *tensor.Tensor
}

type mimiDecodeScratch struct {
	norm1   *tensor.Tensor
	norm2   *tensor.Tensor
	ff1     *tensor.Tensor
	ff2     *tensor.Tensor
	qkv     *tensor.Tensor
	attnOut *tensor.Tensor
}

func (s *mimiDecodeScratch) ensure(slot **tensor.Tensor, shape []int64) (*tensor.Tensor, error) {
	if s == nil {
		return nil, errors.New("native: decode scratch is nil")
	}

	if *slot != nil && equalShape((*slot).Shape(), shape) {
		return *slot, nil
	}

	t, err := tensor.Zeros(shape)
	if err != nil {
		return nil, err
	}

	*slot = t

	return t, nil
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
	return mt.ForwardWithScratch(x, nil)
}

func (mt *mimiDecoderTransformer) ForwardWithScratch(x *tensor.Tensor, scratch *mimiDecodeScratch) (*tensor.Tensor, error) {
	// [B, C, T] -> [B, T, C]
	x, err := x.Transpose(1, 2)
	if err != nil {
		return nil, err
	}

	for i, layer := range mt.layers {
		x, err = layer.forwardWithScratch(x, mt.cos, mt.sin, scratch)
		if err != nil {
			return nil, fmt.Errorf("native: mimi decoder transformer layer %d: %w", i, err)
		}
	}

	return x.Transpose(1, 2)
}

// MimiModel rebuilds latent_to_mimi + mimi_decoder modules.
type MimiModel struct {
	cfg MimiConfig

	quantizerOutProj  *conv1dLayer // mimi.quantizer.output_proj
	upsample          *convTr1dLayer
	transformer       *mimiDecoderTransformer
	decodeScratchPool sync.Pool

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

	model := &MimiModel{
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
	}

	model.decodeScratchPool.New = func() any {
		return &mimiDecodeScratch{}
	}

	return model, nil
}

func (m *MimiModel) SampleRate() int64 { return m.cfg.SampleRate }

// QuantizerProject maps [B, 32, T] -> [B, 512, T].
func (m *MimiModel) QuantizerProject(latentBCT *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil || m.quantizerOutProj == nil {
		return nil, errors.New("native: mimi model is not initialized")
	}

	return ops.Conv1D(latentBCT, m.quantizerOutProj.weight, m.quantizerOutProj.bias, 1, 0, 1, 1)
}

func (m *MimiModel) acquireDecodeScratch() *mimiDecodeScratch {
	if m == nil {
		return &mimiDecodeScratch{}
	}

	raw := m.decodeScratchPool.Get()
	if scratch, ok := raw.(*mimiDecodeScratch); ok && scratch != nil {
		return scratch
	}

	return &mimiDecodeScratch{}
}

func (m *MimiModel) releaseDecodeScratch(scratch *mimiDecodeScratch) {
	if m == nil || scratch == nil {
		return
	}

	m.decodeScratchPool.Put(scratch)
}

// DecodeFromLatent maps [B, 512, T] to PCM-like output [B, 1, N].
func (m *MimiModel) DecodeFromLatent(latent *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil {
		return nil, errors.New("native: mimi model is nil")
	}

	scratch := m.acquireDecodeScratch()
	defer m.releaseDecodeScratch(scratch)

	var err error
	x := latent

	x, err = m.upsample.forwardStreamingOnce(x)
	if err != nil {
		return nil, err
	}

	x, err = m.transformer.ForwardWithScratch(x, scratch)
	if err != nil {
		return nil, err
	}

	x, err = m.initConv.forwardStreamingOnce(x)
	if err != nil {
		return nil, err
	}

	x = eluTensorInPlace(x)

	x, err = m.up1.forwardStreamingOnce(x)
	if err != nil {
		return nil, err
	}

	x, err = m.res1.Forward(x)
	if err != nil {
		return nil, err
	}

	x = eluTensorInPlace(x)

	x, err = m.up2.forwardStreamingOnce(x)
	if err != nil {
		return nil, err
	}

	x, err = m.res2.Forward(x)
	if err != nil {
		return nil, err
	}

	x = eluTensorInPlace(x)

	x, err = m.up3.forwardStreamingOnce(x)
	if err != nil {
		return nil, err
	}

	x, err = m.res3.Forward(x)
	if err != nil {
		return nil, err
	}

	x = eluTensorInPlace(x)

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
