package native

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
)

type timestepEmbedder struct {
	freqs   *tensor.Tensor // [freq/2]
	linear1 *Linear        // mlp.0
	linear2 *Linear        // mlp.2
	alpha   *tensor.Tensor // mlp.3.alpha
}

func loadTimestepEmbedder(vb *VarBuilder) (*timestepEmbedder, error) {
	freqs, err := vb.Tensor("freqs")
	if err != nil {
		return nil, err
	}

	linear1, err := loadLinear(vb, "mlp.0", true)
	if err != nil {
		return nil, err
	}

	linear2, err := loadLinear(vb, "mlp.2", true)
	if err != nil {
		return nil, err
	}

	alpha, err := vb.Tensor("mlp.3.alpha")
	if err != nil {
		return nil, err
	}

	return &timestepEmbedder{freqs: freqs, linear1: linear1, linear2: linear2, alpha: alpha}, nil
}

func (te *timestepEmbedder) Forward(t *tensor.Tensor) (*tensor.Tensor, error) {
	if te == nil {
		return nil, errors.New("native: timestep embedder is nil")
	}

	shape := t.Shape()
	if len(shape) != 2 || shape[1] != 1 {
		return nil, fmt.Errorf("native: timestep input must have shape [B,1], got %v", shape)
	}

	args, err := tensor.BroadcastMul(t, te.freqs)
	if err != nil {
		return nil, err
	}

	cos := args.Clone()
	sin := args.Clone()

	for i, v := range cos.RawData() {
		cos.RawData()[i] = float32(cosf(v))
		sin.RawData()[i] = float32(sinf(v))
	}

	emb, err := tensor.Concat([]*tensor.Tensor{cos, sin}, -1)
	if err != nil {
		return nil, err
	}

	x, err := te.linear1.Forward(emb)
	if err != nil {
		return nil, err
	}

	x = siluTensor(x)

	x, err = te.linear2.Forward(x)
	if err != nil {
		return nil, err
	}

	return rmsNormWithAlpha(x, te.alpha, 1e-5)
}

type flowResBlock struct {
	inLN  *LayerNorm
	mlp0  *Linear
	mlp2  *Linear
	adaLN *Linear // maps condition -> 3*channels
}

func loadFlowResBlock(vb *VarBuilder) (*flowResBlock, error) {
	inLN, err := loadLayerNorm(vb, "in_ln", 1e-6)
	if err != nil {
		return nil, err
	}

	mlp0, err := loadLinear(vb, "mlp.0", true)
	if err != nil {
		return nil, err
	}

	mlp2, err := loadLinear(vb, "mlp.2", true)
	if err != nil {
		return nil, err
	}

	adaLN, err := loadLinear(vb, "adaLN_modulation.1", true)
	if err != nil {
		return nil, err
	}

	return &flowResBlock{inLN: inLN, mlp0: mlp0, mlp2: mlp2, adaLN: adaLN}, nil
}

func (rb *flowResBlock) Forward(x, y *tensor.Tensor) (*tensor.Tensor, error) {
	ada, err := rb.adaLN.Forward(siluTensor(y))
	if err != nil {
		return nil, err
	}

	shape := ada.Shape()
	if len(shape) != 2 || shape[1]%3 != 0 {
		return nil, fmt.Errorf("native: flow resblock modulation shape invalid: %v", shape)
	}

	channels := shape[1] / 3

	shift, err := ada.Narrow(1, 0, channels)
	if err != nil {
		return nil, err
	}

	scale, err := ada.Narrow(1, channels, channels)
	if err != nil {
		return nil, err
	}

	gate, err := ada.Narrow(1, 2*channels, channels)
	if err != nil {
		return nil, err
	}

	h, err := rb.inLN.Forward(x)
	if err != nil {
		return nil, err
	}

	h, err = modulate(h, shift, scale)
	if err != nil {
		return nil, err
	}

	h, err = rb.mlp0.Forward(h)
	if err != nil {
		return nil, err
	}

	h = siluTensor(h)

	h, err = rb.mlp2.Forward(h)
	if err != nil {
		return nil, err
	}

	h, err = mulSameShape(h, gate)
	if err != nil {
		return nil, err
	}

	return addSameShape(x, h)
}

type flowFinalLayer struct {
	linear *Linear
	adaLN  *Linear
	ones   *tensor.Tensor
	zeros  *tensor.Tensor
}

func loadFlowFinalLayer(vb *VarBuilder, channels int64) (*flowFinalLayer, error) {
	linear, err := loadLinear(vb, "linear", true)
	if err != nil {
		return nil, err
	}

	adaLN, err := loadLinear(vb, "adaLN_modulation.1", true)
	if err != nil {
		return nil, err
	}

	ones, err := tensor.Full([]int64{channels}, 1.0)
	if err != nil {
		return nil, err
	}

	zeros, err := tensor.Zeros([]int64{channels})
	if err != nil {
		return nil, err
	}

	return &flowFinalLayer{linear: linear, adaLN: adaLN, ones: ones, zeros: zeros}, nil
}

func (fl *flowFinalLayer) Forward(x, c *tensor.Tensor) (*tensor.Tensor, error) {
	ada, err := fl.adaLN.Forward(siluTensor(c))
	if err != nil {
		return nil, err
	}

	shape := ada.Shape()
	if len(shape) != 2 || shape[1]%2 != 0 {
		return nil, fmt.Errorf("native: flow final modulation shape invalid: %v", shape)
	}

	channels := shape[1] / 2

	shift, err := ada.Narrow(1, 0, channels)
	if err != nil {
		return nil, err
	}

	scale, err := ada.Narrow(1, channels, channels)
	if err != nil {
		return nil, err
	}

	x, err = tensor.LayerNorm(x, fl.ones, fl.zeros, 1e-6)
	if err != nil {
		return nil, err
	}

	x, err = modulate(x, shift, scale)
	if err != nil {
		return nil, err
	}

	return fl.linear.Forward(x)
}

// flowNet implements flow_lm_flow equivalent.
type flowNet struct {
	// timeEmbeds has one embedder per time condition (upstream
	// num_time_conds): 2 for lsd (s and t), 1 for flow_matching, 0 for drifting.
	timeEmbeds []*timestepEmbedder
	condEmbed  *Linear
	inputProj  *Linear
	resBlocks  []*flowResBlock
	finalLayer *flowFinalLayer
}

func loadFlowNet(vb *VarBuilder) (*flowNet, error) {
	var timeEmbeds []*timestepEmbedder

	for i := 0; ; i++ {
		tePath := vb.Path("time_embed", strconv.Itoa(i))
		if !tePath.Has("freqs") {
			break
		}

		te, err := loadTimestepEmbedder(tePath)
		if err != nil {
			return nil, fmt.Errorf("native: load flow time embedding %d: %w", i, err)
		}

		timeEmbeds = append(timeEmbeds, te)
	}

	condEmbed, err := loadLinear(vb, "cond_embed", true)
	if err != nil {
		return nil, err
	}

	inputProj, err := loadLinear(vb, "input_proj", true)
	if err != nil {
		return nil, err
	}

	resBlocks := make([]*flowResBlock, 0, 8)

	for i := 0; ; i++ {
		rbPath := vb.Path("res_blocks", strconv.Itoa(i))
		if !rbPath.Has("in_ln.weight") {
			break
		}

		rb, err := loadFlowResBlock(rbPath)
		if err != nil {
			return nil, fmt.Errorf("native: load flow res block %d: %w", i, err)
		}

		resBlocks = append(resBlocks, rb)
	}

	if len(resBlocks) == 0 {
		return nil, errors.New("native: no flow_net res blocks found")
	}

	channels := inputProj.Weight.Shape()[0]

	finalLayer, err := loadFlowFinalLayer(vb.Path("final_layer"), channels)
	if err != nil {
		return nil, err
	}

	return &flowNet{
		timeEmbeds: timeEmbeds,
		condEmbed:  condEmbed,
		inputProj:  inputProj,
		resBlocks:  resBlocks,
		finalLayer: finalLayer,
	}, nil
}

// Forward computes the flow direction for x with condition c and the time
// conditions ts (s and t for lsd, t for flow_matching, none for drifting).
// Shapes:
//
//	c:     [B, 1024]
//	ts[i]: [B, 1]
//	x:     [B, 32]
func (fn *flowNet) Forward(c *tensor.Tensor, ts []*tensor.Tensor, x *tensor.Tensor) (*tensor.Tensor, error) {
	y, err := fn.conditioning(c, ts)
	if err != nil {
		return nil, err
	}

	xProj, err := fn.inputProj.Forward(x)
	if err != nil {
		return nil, err
	}

	cur := xProj
	for i, rb := range fn.resBlocks {
		cur, err = rb.Forward(cur, y)
		if err != nil {
			return nil, fmt.Errorf("native: flow res block %d: %w", i, err)
		}
	}

	return fn.finalLayer.Forward(cur, y)
}

// conditioning returns cond_embed(c) plus the mean of the time embeddings of
// ts, one [B, 1] tensor per time condition (none for drifting heads).
func (fn *flowNet) conditioning(c *tensor.Tensor, ts []*tensor.Tensor) (*tensor.Tensor, error) {
	if len(ts) != len(fn.timeEmbeds) {
		return nil, fmt.Errorf("native: flow net takes %d time conditions, got %d", len(fn.timeEmbeds), len(ts))
	}

	cProj, err := fn.condEmbed.Forward(c)
	if err != nil {
		return nil, err
	}

	if len(ts) == 0 {
		return cProj, nil
	}

	tSum, err := fn.timeEmbeds[0].Forward(ts[0])
	if err != nil {
		return nil, err
	}

	for i := 1; i < len(ts); i++ {
		emb, err := fn.timeEmbeds[i].Forward(ts[i])
		if err != nil {
			return nil, err
		}

		tSum, err = addSameShapeInPlace(tSum, emb)
		if err != nil {
			return nil, err
		}
	}

	return addSameShape(scaleTensor(tSum, 1/float32(len(ts))), cProj)
}
