package native

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/example/go-pocket-tts/internal/runtime/ops"
	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

type flowTransformerLayer struct {
	norm1   *LayerNorm
	norm2   *LayerNorm
	inProj  *Linear // self_attn.in_proj
	outProj *Linear // self_attn.out_proj
	linear1 *Linear
	linear2 *Linear
	nHeads  int64
	headDim int64
}

func loadFlowTransformerLayer(vb *VarBuilder, nHeads int64) (*flowTransformerLayer, error) {
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

	dModel := outProj.Weight.Shape()[0]
	if dModel%nHeads != 0 {
		return nil, fmt.Errorf("native: d_model %d not divisible by num_heads %d", dModel, nHeads)
	}
	return &flowTransformerLayer{
		norm1:   norm1,
		norm2:   norm2,
		inProj:  inProj,
		outProj: outProj,
		linear1: linear1,
		linear2: linear2,
		nHeads:  nHeads,
		headDim: dModel / nHeads,
	}, nil
}

func (l *flowTransformerLayer) forward(x, ropeCos, ropeSin *tensor.Tensor) (*tensor.Tensor, error) {
	n1, err := l.norm1.Forward(x)
	if err != nil {
		return nil, err
	}
	attn, err := l.selfAttention(n1, ropeCos, ropeSin)
	if err != nil {
		return nil, err
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
	return addSameShape(x, ff)
}

func (l *flowTransformerLayer) selfAttention(x, ropeCos, ropeSin *tensor.Tensor) (*tensor.Tensor, error) {
	shape := x.Shape() // [B, T, D]
	if len(shape) != 3 {
		return nil, fmt.Errorf("native: selfAttention expects [B, T, D], got %v", shape)
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
	q, err = q.Reshape([]int64{b, t, l.nHeads, l.headDim})
	if err != nil {
		return nil, err
	}
	k, err = k.Reshape([]int64{b, t, l.nHeads, l.headDim})
	if err != nil {
		return nil, err
	}
	v, err = v.Reshape([]int64{b, t, l.nHeads, l.headDim})
	if err != nil {
		return nil, err
	}

	q, err = q.Transpose(1, 2) // [B, H, T, Dh]
	if err != nil {
		return nil, err
	}
	k, err = k.Transpose(1, 2)
	if err != nil {
		return nil, err
	}
	v, err = v.Transpose(1, 2)
	if err != nil {
		return nil, err
	}

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
	a, err = a.Transpose(1, 2) // [B, T, H, Dh]
	if err != nil {
		return nil, err
	}
	a, err = a.Reshape([]int64{b, t, d})
	if err != nil {
		return nil, err
	}
	return l.outProj.Forward(a)
}

type flowTransformer struct {
	layers  []*flowTransformerLayer
	ropeCos *tensor.Tensor // [max_seq, head_dim/2]
	ropeSin *tensor.Tensor // [max_seq, head_dim/2]
}

func loadFlowTransformer(vb *VarBuilder, nHeads int64, maxPeriod float64) (*flowTransformer, error) {
	layers := make([]*flowTransformerLayer, 0, 8)
	for i := 0; ; i++ {
		layerPath := vb.Path("transformer", "layers", strconv.Itoa(i))
		if !layerPath.Has("norm1.weight") {
			break
		}
		layer, err := loadFlowTransformerLayer(layerPath, nHeads)
		if err != nil {
			return nil, fmt.Errorf("native: load flow transformer layer %d: %w", i, err)
		}
		layers = append(layers, layer)
	}
	if len(layers) == 0 {
		return nil, fmt.Errorf("native: no flow_lm transformer layers found")
	}

	headDim := layers[0].headDim
	cos, sin, err := buildRoPE(int64(8192), headDim, maxPeriod)
	if err != nil {
		return nil, err
	}
	return &flowTransformer{layers: layers, ropeCos: cos, ropeSin: sin}, nil
}

func (t *flowTransformer) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("native: flow transformer is nil")
	}
	var err error
	for i, layer := range t.layers {
		x, err = layer.forward(x, t.ropeCos, t.ropeSin)
		if err != nil {
			return nil, fmt.Errorf("native: transformer layer %d: %w", i, err)
		}
	}
	return x, nil
}

func buildRoPE(maxSeq, headDim int64, maxPeriod float64) (*tensor.Tensor, *tensor.Tensor, error) {
	if headDim%2 != 0 {
		return nil, nil, fmt.Errorf("native: rope head dim must be even, got %d", headDim)
	}
	half := int(headDim / 2)
	invFreq := make([]float64, half)
	for i := 0; i < half; i++ {
		invFreq[i] = 1.0 / math.Pow(maxPeriod, float64(i)/float64(half))
	}

	cos := make([]float32, int(maxSeq)*half)
	sin := make([]float32, int(maxSeq)*half)
	for pos := int64(0); pos < maxSeq; pos++ {
		base := int(pos) * half
		for i, f := range invFreq {
			angle := float64(pos) * f
			cos[base+i] = float32(math.Cos(angle))
			sin[base+i] = float32(math.Sin(angle))
		}
	}
	cosT, err := tensor.New(cos, []int64{maxSeq, headDim / 2})
	if err != nil {
		return nil, nil, err
	}
	sinT, err := tensor.New(sin, []int64{maxSeq, headDim / 2})
	if err != nil {
		return nil, nil, err
	}
	return cosT, sinT, nil
}

func detectNumHeads(vb *VarBuilder, fallback int64) int64 {
	if vb == nil {
		return fallback
	}
	first := vb.Path("transformer", "layers", "0")
	w, err := first.Tensor("self_attn.in_proj.weight")
	if err != nil {
		return fallback
	}
	shape := w.Shape()
	if len(shape) != 2 {
		return fallback
	}
	dModel := shape[1]
	if dModel <= 0 {
		return fallback
	}

	// Heuristic from known PocketTTS configs.
	for _, n := range []int64{16, 8, 4, 2, 1} {
		if dModel%n == 0 {
			return n
		}
	}
	return fallback
}

func trimPrefix(s, p string) string {
	return strings.TrimPrefix(strings.TrimSpace(s), strings.TrimSpace(p))
}
