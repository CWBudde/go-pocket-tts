package native

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
	"github.com/example/go-pocket-tts/internal/safetensors"
)

type Config struct {
	FlowLM FlowLMConfig
	Mimi   MimiConfig
}

func DefaultConfig() Config {
	return Config{
		FlowLM: DefaultFlowLMConfig(),
		Mimi:   DefaultMimiConfig(),
	}
}

type Model struct {
	store *safetensors.Store
	flow  *FlowLM
	mimi  *MimiModel

	latentToMimiProj *latentToMimiProjector
}

func LoadModelFromSafetensors(path string, cfg Config) (*Model, error) {
	store, err := safetensors.OpenStore(path, safetensors.StoreOptions{})
	if err != nil {
		return nil, err
	}

	return LoadModelFromStore(store, cfg)
}

func LoadModelFromStore(store *safetensors.Store, cfg Config) (*Model, error) {
	if cfg.FlowLM.DModel == 0 {
		cfg = DefaultConfig()
	}

	vb := NewVarBuilder(store)

	flow, err := LoadFlowLM(vb, cfg.FlowLM)
	if err != nil {
		return nil, err
	}

	mimi, err := LoadMimiModel(vb, cfg.Mimi)
	if err != nil {
		return nil, err
	}

	projector, err := newLatentToMimiProjector(flow, mimi)
	if err != nil {
		return nil, err
	}

	return &Model{store: store, flow: flow, mimi: mimi, latentToMimiProj: projector}, nil
}

func (m *Model) Close() {
	if m != nil && m.store != nil {
		m.store.Close()
	}
}

func (m *Model) FlowLM() *FlowLM  { return m.flow }
func (m *Model) Mimi() *MimiModel { return m.mimi }

func (m *Model) TextEmbeddings(tokenIDs []int64) (*tensor.Tensor, error) {
	if m == nil || m.flow == nil {
		return nil, errors.New("native: model flow_lm unavailable")
	}

	return m.flow.TextEmbeddings(tokenIDs)
}

func (m *Model) FlowMain(sequence, textEmbeddings *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	if m == nil || m.flow == nil {
		return nil, nil, errors.New("native: model flow_lm unavailable")
	}

	return m.flow.FlowMain(sequence, textEmbeddings)
}

func (m *Model) FlowDirection(condition, s, t, x *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil || m.flow == nil {
		return nil, errors.New("native: model flow_lm unavailable")
	}

	return m.flow.FlowDirection(condition, s, t, x)
}

func (m *Model) SampleNextLatent(sequence, textEmbeddings *tensor.Tensor, decodeSteps int, eosThreshold, temperature float32, rng *rand.Rand) (*tensor.Tensor, bool, error) {
	if m == nil || m.flow == nil {
		return nil, false, errors.New("native: model flow_lm unavailable")
	}

	return m.flow.SampleNextLatent(sequence, textEmbeddings, decodeSteps, eosThreshold, temperature, rng)
}

func (m *Model) NewFlowState() (*FlowLMState, error) {
	if m == nil || m.flow == nil {
		return nil, errors.New("native: model flow_lm unavailable")
	}

	return m.flow.InitState()
}

func (m *Model) PromptFlow(state *FlowLMState, textEmbeddings *tensor.Tensor) error {
	if m == nil || m.flow == nil {
		return errors.New("native: model flow_lm unavailable")
	}

	return m.flow.PromptText(state, textEmbeddings)
}

func (m *Model) SampleNextLatentStateful(state *FlowLMState, sequenceFrame *tensor.Tensor, decodeSteps int, eosThreshold, temperature float32, rng *rand.Rand) (*tensor.Tensor, bool, error) {
	if m == nil || m.flow == nil {
		return nil, false, errors.New("native: model flow_lm unavailable")
	}

	return m.flow.SampleNextLatentStateful(state, sequenceFrame, decodeSteps, eosThreshold, temperature, rng)
}

// LatentToMimi maps FlowLM latents [B, T, 32] to Mimi latents [B, 512, T].
func (m *Model) LatentToMimi(latent *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil || m.flow == nil || m.mimi == nil {
		return nil, errors.New("native: model is not fully initialized")
	}

	if latent == nil {
		return nil, errors.New("native: latent tensor is nil")
	}

	shape := latent.Shape()
	if len(shape) != 3 || shape[2] != m.flow.cfg.LDim {
		return nil, fmt.Errorf("native: latent shape must be [B,T,%d], got %v", m.flow.cfg.LDim, shape)
	}

	if m.latentToMimiProj != nil {
		return m.latentToMimiProj.Project(latent)
	}

	// Fuse denormalization + layout transform in one pass:
	// [B,T,D] -> [B,D,T], value = latent*emb_std + emb_mean.
	denorm, err := denormLatentToBCT(latent, m.flow.embStd, m.flow.embMean, m.flow.cfg.LDim)
	if err != nil {
		return nil, err
	}

	return m.mimi.QuantizerProject(denorm)
}

type latentToMimiProjector struct {
	inChannels  int
	outChannels int
	weight      []float32 // [outChannels, inChannels]
	bias        []float32 // [outChannels]
}

func newLatentToMimiProjector(flow *FlowLM, mimi *MimiModel) (*latentToMimiProjector, error) {
	if flow == nil || mimi == nil || mimi.quantizerOutProj == nil || mimi.quantizerOutProj.weight == nil {
		return nil, nil
	}

	proj := mimi.quantizerOutProj
	if proj.stride != 1 || proj.dilation != 1 || proj.groups != 1 {
		return nil, nil
	}

	wShape := proj.weight.Shape()
	if len(wShape) != 3 {
		return nil, fmt.Errorf("native: quantizer projection weight rank must be 3, got %v", wShape)
	}

	outCh := int(wShape[0])
	inCh := int(wShape[1])

	kSize := int(wShape[2])
	if kSize != 1 {
		return nil, nil
	}

	if inCh <= 0 || outCh <= 0 || inCh != int(flow.cfg.LDim) {
		return nil, nil
	}

	std := flow.embStd.RawData()

	mean := flow.embMean.RawData()
	if len(std) < inCh || len(mean) < inCh {
		return nil, fmt.Errorf("native: flow emb stats shape mismatch: std=%d mean=%d in_ch=%d", len(std), len(mean), inCh)
	}

	wRaw := proj.weight.RawData()
	if len(wRaw) < outCh*inCh*kSize {
		return nil, fmt.Errorf("native: quantizer projection weight length mismatch: got %d want >= %d", len(wRaw), outCh*inCh*kSize)
	}

	scaledW := make([]float32, outCh*inCh)
	bias := make([]float32, outCh)

	var bRaw []float32
	if proj.bias != nil {
		bRaw = proj.bias.RawData()
		if len(bRaw) < outCh {
			return nil, fmt.Errorf("native: quantizer projection bias length mismatch: got %d want >= %d", len(bRaw), outCh)
		}
	}

	for oc := range outCh {
		bv := float32(0)
		if bRaw != nil {
			bv = bRaw[oc]
		}

		row := scaledW[oc*inCh : (oc+1)*inCh]

		wBase := oc * inCh * kSize
		for ic := range inCh {
			w := wRaw[wBase+ic*kSize]
			row[ic] = w * std[ic]
			bv += w * mean[ic]
		}

		bias[oc] = bv
	}

	return &latentToMimiProjector{
		inChannels:  inCh,
		outChannels: outCh,
		weight:      scaledW,
		bias:        bias,
	}, nil
}

func (p *latentToMimiProjector) Project(latent *tensor.Tensor) (*tensor.Tensor, error) {
	if p == nil {
		return nil, errors.New("native: latent-to-mimi projector is nil")
	}

	if latent == nil {
		return nil, errors.New("native: latent tensor is nil")
	}

	shape := latent.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("native: latent shape must be rank-3 [B,T,D], got %v", shape)
	}

	batch := int(shape[0])
	steps := int(shape[1])

	depth := int(shape[2])
	if batch <= 0 || steps <= 0 || depth <= 0 {
		return nil, fmt.Errorf("native: latent shape must be positive, got %v", shape)
	}

	if depth != p.inChannels {
		return nil, fmt.Errorf("native: latent depth mismatch, got %d want %d", depth, p.inChannels)
	}

	out, err := tensor.Zeros([]int64{shape[0], int64(p.outChannels), shape[1]})
	if err != nil {
		return nil, err
	}

	latentData := latent.RawData()
	outData := out.RawData()
	latentBatchStride := steps * depth
	outBatchStride := p.outChannels * steps

	workers := tensor.Workers()
	runParallel := workers > 1 && p.outChannels > 1
	runRows := func(batchIdx int, ocLo, ocHi int) {
		latentB := latentData[batchIdx*latentBatchStride : (batchIdx+1)*latentBatchStride]
		outB := outData[batchIdx*outBatchStride : (batchIdx+1)*outBatchStride]

		for oc := ocLo; oc < ocHi; oc++ {
			wRow := p.weight[oc*p.inChannels : (oc+1)*p.inChannels]
			outRow := outB[oc*steps : (oc+1)*steps]
			bv := p.bias[oc]

			for ti := range steps {
				off := ti * p.inChannels
				outRow[ti] = tensor.DotProduct(latentB[off:off+p.inChannels], wRow) + bv
			}
		}
	}

	for bi := range batch {
		if runParallel {
			parallelForByWorkers(p.outChannels, workers, func(lo, hi int) {
				runRows(bi, lo, hi)
			})

			continue
		}

		runRows(bi, 0, p.outChannels)
	}

	return out, nil
}

func parallelForByWorkers(n, workers int, fn func(lo, hi int)) {
	if n <= 1 || workers <= 1 {
		fn(0, n)
		return
	}

	if workers > n {
		workers = n
	}

	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup

	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)

		wg.Add(1)

		go func(lo, hi int) {
			defer wg.Done()

			fn(lo, hi)
		}(lo, hi)
	}

	wg.Wait()
}

func denormLatentToBCT(latent, embStd, embMean *tensor.Tensor, lDim int64) (*tensor.Tensor, error) {
	if latent == nil {
		return nil, errors.New("native: latent tensor is nil")
	}

	if embStd == nil || embMean == nil {
		return nil, errors.New("native: flow embedding stats are not initialized")
	}

	shape := latent.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("native: latent shape must be rank-3 [B,T,D], got %v", shape)
	}

	if shape[2] != lDim {
		return nil, fmt.Errorf("native: latent depth mismatch, got %d want %d", shape[2], lDim)
	}

	b := int(shape[0])
	t := int(shape[1])

	d := int(shape[2])
	if b <= 0 || t <= 0 || d <= 0 {
		return nil, fmt.Errorf("native: latent shape must be positive, got %v", shape)
	}

	std := embStd.RawData()

	mean := embMean.RawData()
	if len(std) < d || len(mean) < d {
		return nil, fmt.Errorf("native: embedding stats shape mismatch: std=%d mean=%d latent_d=%d", len(std), len(mean), d)
	}

	out, err := tensor.Zeros([]int64{shape[0], shape[2], shape[1]})
	if err != nil {
		return nil, err
	}

	latentData := latent.RawData()
	outData := out.RawData()
	latentBStride := t * d
	outBStride := d * t

	for bi := range b {
		latentBBase := bi * latentBStride
		outBBase := bi * outBStride

		for ti := range t {
			latentOff := latentBBase + ti*d

			for di := range d {
				outOff := outBBase + di*t + ti
				outData[outOff] = latentData[latentOff+di]*std[di] + mean[di]
			}
		}
	}

	return out, nil
}

// MimiDecode maps [B, 512, T] to [B, 1, N] PCM-like output.
func (m *Model) MimiDecode(mimiLatent *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil || m.mimi == nil {
		return nil, errors.New("native: model mimi unavailable")
	}

	return m.mimi.DecodeFromLatent(mimiLatent)
}

// EncodeVoiceHook exposes the future voice encoder path (Phase 20 target).
func (m *Model) EncodeVoiceHook(audio *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil || m.mimi == nil {
		return nil, errors.New("native: model mimi unavailable")
	}

	return m.mimi.EncodeToLatent(audio)
}
