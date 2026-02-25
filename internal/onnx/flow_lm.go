package onnx

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
)

// latentDim is the latent dimension used by the flow LM (32 for PocketTTS).
const latentDim = 32

// NewBOSSequence creates the initial BOS (beginning-of-sequence) tensor.
// It returns a [1, 1, 32] float32 tensor filled with NaN values.
// The flow_lm_main graph internally replaces NaN with a learned bos_emb vector.
func NewBOSSequence() *Tensor {
	data := make([]float32, latentDim)
	for i := range data {
		data[i] = float32(math.NaN())
	}
	t, _ := NewTensor(data, []int64{1, 1, int64(latentDim)})
	return t
}

// AppendLatentFrame concatenates a new latent frame [1, 1, 32] to the
// sequence tensor [1, S, 32], producing [1, S+1, 32].
func AppendLatentFrame(sequence, frame *Tensor) (*Tensor, error) {
	seqShape := sequence.Shape()
	frameShape := frame.Shape()

	if len(seqShape) != 3 || seqShape[0] != 1 || seqShape[2] != int64(latentDim) {
		return nil, fmt.Errorf("sequence shape %v invalid, want [1, S, %d]", seqShape, latentDim)
	}
	if len(frameShape) != 3 || frameShape[0] != 1 || frameShape[1] != 1 || frameShape[2] != int64(latentDim) {
		return nil, fmt.Errorf("frame shape %v invalid, want [1, 1, %d]", frameShape, latentDim)
	}

	seqData, err := ExtractFloat32(sequence)
	if err != nil {
		return nil, fmt.Errorf("extract sequence data: %w", err)
	}
	frameData, err := ExtractFloat32(frame)
	if err != nil {
		return nil, fmt.Errorf("extract frame data: %w", err)
	}

	combined := append(seqData, frameData...)
	newS := seqShape[1] + 1
	return NewTensor(combined, []int64{1, newS, int64(latentDim)})
}

// FlowLMStep runs a single autoregressive step of the flow_lm_main graph.
//
// Inputs:
//   - sequence: [1, S, 32] — growing latent sequence (starts as BOS NaN tensor)
//   - textEmbeddings: [1, T, 1024] — text conditioning from TextConditioner
//
// Outputs:
//   - lastHidden: [1, 1024] — transformer hidden state for flow decoding
//   - eosLogits: [1, 1] — raw EOS logit (compare against threshold)
func (e *Engine) FlowLMStep(ctx context.Context, sequence, textEmbeddings *Tensor) (lastHidden, eosLogits *Tensor, err error) {
	runner, ok := e.runners["flow_lm_main"]
	if !ok {
		return nil, nil, fmt.Errorf("flow_lm_main graph not found in manifest")
	}

	outputs, err := runner.Run(ctx, map[string]*Tensor{
		"sequence":        sequence,
		"text_embeddings": textEmbeddings,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("flow_lm_main: run: %w", err)
	}

	lastHidden, ok = outputs["last_hidden"]
	if !ok {
		return nil, nil, fmt.Errorf("flow_lm_main: missing 'last_hidden' in output")
	}
	eosLogits, ok = outputs["eos_logits"]
	if !ok {
		return nil, nil, fmt.Errorf("flow_lm_main: missing 'eos_logits' in output")
	}

	return lastHidden, eosLogits, nil
}

// FlowLMFlow runs the Euler flow integration (LSD decode) to convert
// last_hidden [1, 1024] into a latent frame [1, 1, 32].
//
// Algorithm:
//  1. Sample noise x ~ N(0, sqrt(temperature)), shape [1, 32]
//  2. For i in 0..steps: run flow_lm_flow(condition=lastHidden, s, t, x) → flow_dir
//     then x += flow_dir / steps
//  3. Return x reshaped to [1, 1, 32]
func (e *Engine) FlowLMFlow(ctx context.Context, lastHidden *Tensor, temperature float64, steps int) (*Tensor, error) {
	runner, ok := e.runners["flow_lm_flow"]
	if !ok {
		return nil, fmt.Errorf("flow_lm_flow graph not found in manifest")
	}

	// Sample initial noise x [1, 32].
	x := make([]float32, latentDim)
	if temperature > 0 {
		stddev := math.Sqrt(temperature)
		for i := range x {
			x[i] = float32(randNormal() * stddev)
		}
	}

	fSteps := float32(steps)
	for i := range steps {
		s := float32(i) / fSteps
		tVal := float32(i+1) / fSteps

		sTensor, _ := NewTensor([]float32{s}, []int64{1, 1})
		tTensor, _ := NewTensor([]float32{tVal}, []int64{1, 1})
		xTensor, _ := NewTensor(x, []int64{1, int64(latentDim)})

		outputs, err := runner.Run(ctx, map[string]*Tensor{
			"condition": lastHidden,
			"s":         sTensor,
			"t":         tTensor,
			"x":         xTensor,
		})
		if err != nil {
			return nil, fmt.Errorf("flow_lm_flow step %d: run: %w", i, err)
		}

		flowDir, ok := outputs["flow_direction"]
		if !ok {
			return nil, fmt.Errorf("flow_lm_flow step %d: missing 'flow_direction' in output", i)
		}

		dirData, err := ExtractFloat32(flowDir)
		if err != nil {
			return nil, fmt.Errorf("flow_lm_flow step %d: extract flow_direction: %w", i, err)
		}

		for j := range x {
			x[j] += dirData[j] / fSteps
		}
	}

	return NewTensor(x, []int64{1, 1, int64(latentDim)})
}

// randNormal returns a standard normal random value.
// Package-level var to allow deterministic testing.
var randNormal = func() float64 {
	return rand.NormFloat64()
}

// EOSDetected returns true if the raw EOS logit exceeds the threshold.
// The eos_logits tensor has shape [1, 1]; threshold is compared against
// the raw logit value (not sigmoid-transformed).
func EOSDetected(eosLogits *Tensor, threshold float64) bool {
	data, err := ExtractFloat32(eosLogits)
	if err != nil || len(data) == 0 {
		return false
	}
	return float64(data[0]) > threshold
}

// FlowLMKVState holds the KV-cache accumulated during prefill and updated each
// AR step. KV[i] has shape [2, 1, S, H, Dh] where dim-0 is K(0)/V(1).
// Offset is the current write position (equals the number of tokens processed).
type FlowLMKVState struct {
	KV     []*Tensor
	Offset int64
}

// FlowLMPrefill runs the flow_lm_prefill graph on text_embeddings (which may
// include prepended voice embeddings) and returns the resulting KV-cache state.
// The returned state is ready to be passed to FlowLMStepStateful for AR generation.
func (e *Engine) FlowLMPrefill(ctx context.Context, textEmbeddings *Tensor) (*FlowLMKVState, error) {
	runner, ok := e.runners["flow_lm_prefill"]
	if !ok {
		return nil, fmt.Errorf("flow_lm_prefill graph not found in manifest")
	}

	outputs, err := runner.Run(ctx, map[string]*Tensor{
		"text_embeddings": textEmbeddings,
	})
	if err != nil {
		return nil, fmt.Errorf("flow_lm_prefill: run: %w", err)
	}

	// Unpack kv_0, kv_1, ... until a key is missing.
	var kvTensors []*Tensor
	for i := 0; ; i++ {
		key := fmt.Sprintf("kv_%d", i)
		kv, ok := outputs[key]
		if !ok {
			break
		}
		kvTensors = append(kvTensors, kv)
	}
	if len(kvTensors) == 0 {
		return nil, fmt.Errorf("flow_lm_prefill: no kv_N outputs in result")
	}

	offsetTensor, ok := outputs["offset"]
	if !ok {
		return nil, fmt.Errorf("flow_lm_prefill: missing 'offset' in output")
	}
	offsetData, err := ExtractInt64(offsetTensor)
	if err != nil {
		return nil, fmt.Errorf("flow_lm_prefill: extract offset: %w", err)
	}
	if len(offsetData) == 0 {
		return nil, fmt.Errorf("flow_lm_prefill: offset tensor is empty")
	}

	return &FlowLMKVState{KV: kvTensors, Offset: offsetData[0]}, nil
}
