package native

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

type FlowLMConfig struct {
	DModel    int64
	NumHeads  int64
	MaxPeriod float64
	LDim      int64

	// InsertBOSBeforeVoice prepends flow_lm.bos_before_voice to audio-prompt
	// voice embeddings (model config flow_lm.insert_bos_before_voice).
	InsertBOSBeforeVoice bool

	// FlowType selects the sampler head (model config flow_lm.flow.type):
	// modelcfg.FlowTypeLSD (also ""), FlowTypeFlowMatching or FlowTypeDrifting.
	FlowType string
}

func DefaultFlowLMConfig() FlowLMConfig {
	return FlowLMConfig{
		DModel:    1024,
		NumHeads:  16,
		MaxPeriod: 10000.0,
		LDim:      32,
		FlowType:  modelcfg.FlowTypeLSD,
	}
}

// checkFlowTimeConds checks that a flow_net with got time embeddings fits
// flowType (upstream SimpleMLPAdaLN.from_pydantic_config).
func checkFlowTimeConds(flowType string, got int) error {
	var want int

	switch flowType {
	case "", modelcfg.FlowTypeLSD:
		want = 2
	case modelcfg.FlowTypeFlowMatching:
		want = 1
	case modelcfg.FlowTypeDrifting:
		want = 0
	default:
		return fmt.Errorf("native: unknown flow type %q", flowType)
	}

	if got != want {
		return fmt.Errorf("native: flow type %q needs %d flow_net.time_embed entries, checkpoint has %d", flowType, want, got)
	}

	return nil
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

	bosBeforeVoice *tensor.Tensor // [1, 1, DModel]; nil unless cfg.InsertBOSBeforeVoice

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

	err = checkFlowTimeConds(cfg.FlowType, len(flowNet.timeEmbeds))
	if err != nil {
		return nil, err
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

	bosBeforeVoice, err := loadBOSBeforeVoice(flow, cfg)
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

		bosBeforeVoice: bosBeforeVoice,
	}, nil
}

// VoicePrompt returns the audio-prompt voice embeddings [1, T, DModel] as the
// FlowLM consumes them: with InsertBOSBeforeVoice, flow_lm.bos_before_voice is
// prepended (upstream tts_model.py, before prompting the audio conditioning).
// Precomputed voice model states already contain it.
func (f *FlowLM) VoicePrompt(voiceEmb *tensor.Tensor) (*tensor.Tensor, error) {
	if f == nil || !f.cfg.InsertBOSBeforeVoice {
		return voiceEmb, nil
	}

	if f.bosBeforeVoice == nil {
		return nil, errors.New("native: flow_lm.bos_before_voice not loaded")
	}

	return tensor.Concat([]*tensor.Tensor{f.bosBeforeVoice, voiceEmb}, 1)
}

// Offset returns the number of positions prompted into the state so far.
func (s *FlowLMState) Offset() int64 {
	if s == nil || s.transformer == nil || len(s.transformer.layers) == 0 {
		return 0
	}

	return s.transformer.layers[0].offset
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

func (f *FlowLM) InitStateFromVoiceModelState(voiceState *safetensors.VoiceModelState) (*FlowLMState, error) {
	if f == nil || f.transformer == nil {
		return nil, errors.New("native: flow_lm transformer unavailable")
	}

	tfState, err := f.transformer.initStateFromVoiceModelState(voiceState)
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

	err := f.transformer.prefill(textEmbeddings, state.transformer)
	if err != nil {
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

	decoded, err := f.decode(last, noise, decodeSteps)
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

	return f.flowNet.Forward(condition, []*tensor.Tensor{s, t}, x)
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

// OTDecode runs Euler integration of an optimal-transport flow (flow_matching
// heads, one time condition): current += v(i/steps, current)/steps.
func (f *FlowLM) OTDecode(condition, x0 *tensor.Tensor, steps int) (*tensor.Tensor, error) {
	if steps <= 0 {
		return nil, errors.New("native: flow_matching decode steps must be >0")
	}

	shape := x0.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("native: flow_matching decode input x0 must be [B, D], got %v", shape)
	}

	if f == nil || f.flowNet == nil {
		return nil, errors.New("native: flow_lm flow net unavailable")
	}

	current := x0.Clone()
	curData := current.RawData()

	inv := 1.0 / float32(steps)
	for i := range steps {
		t, err := tensor.Full([]int64{shape[0], 1}, float32(i)/float32(steps))
		if err != nil {
			return nil, err
		}

		flow, err := f.flowNet.Forward(condition, []*tensor.Tensor{t}, current)
		if err != nil {
			return nil, err
		}

		for j, v := range flow.RawData() {
			curData[j] += v * inv
		}
	}

	return current, nil
}

// DriftingDecode runs a drifting head: one forward pass on x0 without time
// conditions.
func (f *FlowLM) DriftingDecode(condition, x0 *tensor.Tensor) (*tensor.Tensor, error) {
	if shape := x0.Shape(); len(shape) != 2 {
		return nil, fmt.Errorf("native: drifting decode input x0 must be [B, D], got %v", shape)
	}

	if f == nil || f.flowNet == nil {
		return nil, errors.New("native: flow_lm flow net unavailable")
	}

	return f.flowNet.Forward(condition, nil, x0)
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

	decoded, err := f.decode(lastHidden, noise, decodeSteps)
	if err != nil {
		return nil, false, err
	}

	next, err := decoded.Reshape([]int64{decoded.Shape()[0], 1, decoded.Shape()[1]})
	if err != nil {
		return nil, false, err
	}

	return next, isEOS, nil
}

// decode samples a latent from noise x0 with the sampler head of
// cfg.FlowType. Drifting heads ignore steps.
func (f *FlowLM) decode(condition, x0 *tensor.Tensor, steps int) (*tensor.Tensor, error) {
	switch f.cfg.FlowType {
	case "", modelcfg.FlowTypeLSD:
		return f.LSDDecode(condition, x0, steps)
	case modelcfg.FlowTypeFlowMatching:
		return f.OTDecode(condition, x0, steps)
	case modelcfg.FlowTypeDrifting:
		return f.DriftingDecode(condition, x0)
	default:
		return nil, fmt.Errorf("native: unknown flow type %q", f.cfg.FlowType)
	}
}

// loadBOSBeforeVoice loads flow_lm.bos_before_voice [1, 1, DModel] when
// cfg.InsertBOSBeforeVoice is set, and nil otherwise.
func loadBOSBeforeVoice(flow *VarBuilder, cfg FlowLMConfig) (*tensor.Tensor, error) {
	if !cfg.InsertBOSBeforeVoice {
		return nil, nil //nolint:nilnil // no tensor is the valid result without the flag
	}

	t, err := flow.Tensor("bos_before_voice", 1, 1, cfg.DModel)
	if err != nil {
		return nil, fmt.Errorf("native: insert_bos_before_voice needs flow_lm.bos_before_voice: %w", err)
	}

	return t, nil
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
