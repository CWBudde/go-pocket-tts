package native

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

type FlowLMConfig struct {
	DModel    int64
	NumHeads  int64
	MaxPeriod float64
	LDim      int64
}

func DefaultFlowLMConfig() FlowLMConfig {
	return FlowLMConfig{
		DModel:    1024,
		NumHeads:  16,
		MaxPeriod: 10000.0,
		LDim:      32,
	}
}

// FlowLM rebuilds the ONNX flow_lm_main + flow_lm_flow modules from safetensors weights.
type FlowLM struct {
	conditioner *LUTConditioner
	transformer *flowTransformer
	flowNet     *flowNet

	embStd    *tensor.Tensor // [32]
	embMean   *tensor.Tensor // [32]
	bosEmb    *tensor.Tensor // [32]
	inputProj *Linear        // flow_lm.input_linear
	outNorm   *LayerNorm     // flow_lm.out_norm
	outEOS    *Linear        // flow_lm.out_eos

	cfg FlowLMConfig
}

// FlowLMState stores per-request transformer cache state for incremental AR
// generation, matching xn-style prompt+step execution.
type FlowLMState struct {
	transformer *flowTransformerState
}

func LoadFlowLM(vb *VarBuilder, cfg FlowLMConfig) (*FlowLM, error) {
	flow := vb.Path("flow_lm")

	if cfg.DModel == 0 {
		cfg = DefaultFlowLMConfig()
	}

	if cfg.NumHeads == 0 {
		cfg.NumHeads = detectNumHeads(flow, 16)
	}

	conditioner, err := loadLUTConditioner(flow)
	if err != nil {
		return nil, fmt.Errorf("native: load conditioner: %w", err)
	}

	transformer, err := loadFlowTransformer(flow, cfg.NumHeads, cfg.MaxPeriod)
	if err != nil {
		return nil, fmt.Errorf("native: load flow transformer: %w", err)
	}

	flowNet, err := loadFlowNet(flow.Path("flow_net"))
	if err != nil {
		return nil, fmt.Errorf("native: load flow_net: %w", err)
	}

	embStd, err := flow.Tensor("emb_std", cfg.LDim)
	if err != nil {
		return nil, err
	}

	embMean, err := flow.Tensor("emb_mean", cfg.LDim)
	if err != nil {
		return nil, err
	}

	bosEmb, err := flow.Tensor("bos_emb", cfg.LDim)
	if err != nil {
		return nil, err
	}

	inputProj, err := loadLinear(flow, "input_linear", true)
	if err != nil {
		return nil, err
	}

	outNorm, err := loadLayerNorm(flow, "out_norm", 1e-5)
	if err != nil {
		return nil, err
	}

	outEOS, err := loadLinear(flow, "out_eos", true)
	if err != nil {
		return nil, err
	}

	return &FlowLM{
		conditioner: conditioner,
		transformer: transformer,
		flowNet:     flowNet,
		embStd:      embStd,
		embMean:     embMean,
		bosEmb:      bosEmb,
		inputProj:   inputProj,
		outNorm:     outNorm,
		outEOS:      outEOS,
		cfg:         cfg,
	}, nil
}

func (f *FlowLM) InitState() (*FlowLMState, error) {
	if f == nil || f.transformer == nil {
		return nil, errors.New("native: flow_lm transformer unavailable")
	}

	tfState, err := f.transformer.initState()
	if err != nil {
		return nil, err
	}

	return &FlowLMState{transformer: tfState}, nil
}

func (f *FlowLM) TextEmbeddings(tokenIDs []int64) (*tensor.Tensor, error) {
	if f == nil || f.conditioner == nil {
		return nil, errors.New("native: flow_lm not initialized")
	}

	return f.conditioner.EmbedTokens(tokenIDs)
}

func (f *FlowLM) PromptText(state *FlowLMState, textEmbeddings *tensor.Tensor) error {
	if f == nil || f.transformer == nil {
		return errors.New("native: flow_lm transformer unavailable")
	}

	if state == nil || state.transformer == nil {
		return errors.New("native: flow_lm state unavailable")
	}

	if textEmbeddings == nil {
		return errors.New("native: prompt text embeddings are nil")
	}

	shape := textEmbeddings.Shape()
	if len(shape) != 3 {
		return fmt.Errorf("native: prompt text embeddings must be [B,T,D], got %v", shape)
	}

	if shape[2] != f.cfg.DModel {
		return fmt.Errorf("native: prompt text embedding width must be %d, got %d", f.cfg.DModel, shape[2])
	}

	if shape[1] == 0 {
		return nil
	}

	if _, err := f.transformer.prefill(textEmbeddings, state.transformer); err != nil {
		return err
	}

	return nil
}

// FlowMain runs the flow_lm_main equivalent and returns:
// - last_hidden [B, DModel]
// - eos_logits [B, 1].
func (f *FlowLM) FlowMain(sequence, textEmbeddings *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	if f == nil {
		return nil, nil, errors.New("native: flow_lm is nil")
	}

	seq, err := replaceNaNWithVector(sequence, f.bosEmb)
	if err != nil {
		return nil, nil, err
	}

	in, err := f.inputProj.Forward(seq)
	if err != nil {
		return nil, nil, err
	}

	x, err := tensor.Concat([]*tensor.Tensor{textEmbeddings, in}, 1)
	if err != nil {
		return nil, nil, err
	}

	x, err = f.transformer.forward(x)
	if err != nil {
		return nil, nil, err
	}

	x, err = f.outNorm.Forward(x)
	if err != nil {
		return nil, nil, err
	}

	last, err := lastToken(x)
	if err != nil {
		return nil, nil, err
	}

	eos, err := f.outEOS.Forward(last)
	if err != nil {
		return nil, nil, err
	}

	return last, eos, nil
}

// SampleNextLatentStateful runs a single incremental generation step with
// cached transformer state. The caller should initialize and prompt state once
// via InitState + PromptText, then call this per frame.
func (f *FlowLM) SampleNextLatentStateful(state *FlowLMState, sequenceFrame *tensor.Tensor, decodeSteps int, eosThreshold, temperature float32, rng *rand.Rand) (*tensor.Tensor, bool, error) {
	if f == nil {
		return nil, false, errors.New("native: flow_lm is nil")
	}

	if state == nil || state.transformer == nil {
		return nil, false, errors.New("native: flow_lm state unavailable")
	}

	seq, err := replaceNaNWithVector(sequenceFrame, f.bosEmb)
	if err != nil {
		return nil, false, err
	}

	in, err := f.inputProj.Forward(seq)
	if err != nil {
		return nil, false, err
	}

	x, err := f.transformer.step(in, state.transformer)
	if err != nil {
		return nil, false, err
	}

	x, err = f.outNorm.Forward(x)
	if err != nil {
		return nil, false, err
	}

	last, err := lastToken(x)
	if err != nil {
		return nil, false, err
	}

	eos, err := f.outEOS.Forward(last)
	if err != nil {
		return nil, false, err
	}

	if len(eos.RawData()) < 1 {
		return nil, false, errors.New("native: eos logits tensor is empty")
	}

	isEOS := eos.RawData()[0] > eosThreshold

	noise, err := makeGaussianNoise(last.Shape()[0], f.cfg.LDim, temperature, rng)
	if err != nil {
		return nil, false, err
	}

	decoded, err := f.LSDDecode(last, noise, decodeSteps)
	if err != nil {
		return nil, false, err
	}

	next, err := decoded.Reshape([]int64{decoded.Shape()[0], 1, decoded.Shape()[1]})
	if err != nil {
		return nil, false, err
	}

	return next, isEOS, nil
}

// FlowDirection runs flow_lm_flow equivalent.
func (f *FlowLM) FlowDirection(condition, s, t, x *tensor.Tensor) (*tensor.Tensor, error) {
	if f == nil || f.flowNet == nil {
		return nil, errors.New("native: flow_lm flow net unavailable")
	}

	return f.flowNet.Forward(condition, s, t, x)
}

// LSDDecode runs Euler integration in flow space.
func (f *FlowLM) LSDDecode(condition, x0 *tensor.Tensor, steps int) (*tensor.Tensor, error) {
	if steps <= 0 {
		return nil, errors.New("native: lsd decode steps must be >0")
	}

	shape := x0.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("native: lsd decode input x0 must be [B, D], got %v", shape)
	}

	b := shape[0]
	current := x0.Clone()
	curData := current.RawData()

	inv := 1.0 / float32(steps)
	for i := range steps {
		sVal := float32(i) / float32(steps)
		tVal := float32(i+1) / float32(steps)

		s, err := tensor.Full([]int64{b, 1}, sVal)
		if err != nil {
			return nil, err
		}

		t, err := tensor.Full([]int64{b, 1}, tVal)
		if err != nil {
			return nil, err
		}

		flow, err := f.FlowDirection(condition, s, t, current)
		if err != nil {
			return nil, err
		}
		// Update current in-place: current += flow * (1/steps).
		// Avoids two extra Clone allocations (scaleTensor + addSameShape).
		flowData := flow.RawData()
		for j := range curData {
			curData[j] += flowData[j] * inv
		}
	}

	return current, nil
}

// SampleNextLatent mirrors xn sample_next_latent behavior.
func (f *FlowLM) SampleNextLatent(sequence, textEmbeddings *tensor.Tensor, decodeSteps int, eosThreshold, temperature float32, rng *rand.Rand) (*tensor.Tensor, bool, error) {
	lastHidden, eos, err := f.FlowMain(sequence, textEmbeddings)
	if err != nil {
		return nil, false, err
	}

	if len(eos.RawData()) < 1 {
		return nil, false, errors.New("native: eos logits tensor is empty")
	}

	isEOS := eos.RawData()[0] > eosThreshold

	noise, err := makeGaussianNoise(lastHidden.Shape()[0], f.cfg.LDim, temperature, rng)
	if err != nil {
		return nil, false, err
	}

	decoded, err := f.LSDDecode(lastHidden, noise, decodeSteps)
	if err != nil {
		return nil, false, err
	}

	next, err := decoded.Reshape([]int64{decoded.Shape()[0], 1, decoded.Shape()[1]})
	if err != nil {
		return nil, false, err
	}

	return next, isEOS, nil
}

func makeGaussianNoise(batch, dim int64, temperature float32, rng *rand.Rand) (*tensor.Tensor, error) {
	if batch <= 0 || dim <= 0 {
		return nil, fmt.Errorf("native: invalid gaussian noise shape [%d,%d]", batch, dim)
	}

	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}

	sigma := float64(temperature)
	if sigma < 0 {
		sigma = 0
	}

	sigma = math.Sqrt(sigma)

	data := make([]float32, int(batch*dim))
	for i := range data {
		data[i] = float32(rng.NormFloat64() * sigma)
	}

	return tensor.New(data, []int64{batch, dim})
}
