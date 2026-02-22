package onnx

import (
	"context"
	"fmt"
	"testing"
)

// ---------------------------------------------------------------------------
// Unit tests for Engine.FlowLMStep (no ORT dependency)
// ---------------------------------------------------------------------------

func TestFlowLMStep_MissingGraph(t *testing.T) {
	e := engineWithFakeRunners(map[string]runnerIface{})
	seq, _ := NewTensor(make([]float32, 32), []int64{1, 1, 32})
	emb, _ := NewTensor(make([]float32, 1024), []int64{1, 1, 1024})

	_, _, err := e.FlowLMStep(context.Background(), seq, emb)
	if err == nil {
		t.Fatal("expected error when flow_lm_main graph is absent")
	}
}

func TestFlowLMStep_PropagatesRunnerError(t *testing.T) {
	fake := &fakeRunner{
		name: "flow_lm_main",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			return nil, fmt.Errorf("ort failure")
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"flow_lm_main": fake})

	seq, _ := NewTensor(make([]float32, 32), []int64{1, 1, 32})
	emb, _ := NewTensor(make([]float32, 1024), []int64{1, 1, 1024})

	_, _, err := e.FlowLMStep(context.Background(), seq, emb)
	if err == nil {
		t.Fatal("expected error propagated from runner")
	}
}

func TestFlowLMStep_ReturnsOutputs(t *testing.T) {
	// Fake runner returns last_hidden [1, 1024] and eos_logits [1, 1].
	hiddenData := make([]float32, 1024)
	for i := range hiddenData {
		hiddenData[i] = float32(i) * 0.01
	}
	fakeHidden, _ := NewTensor(hiddenData, []int64{1, 1024})

	eosData := []float32{-5.0}
	fakeEOS, _ := NewTensor(eosData, []int64{1, 1})

	fake := &fakeRunner{
		name: "flow_lm_main",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			// Verify expected input keys.
			if _, ok := inputs["sequence"]; !ok {
				t.Error("expected 'sequence' input key")
			}
			if _, ok := inputs["text_embeddings"]; !ok {
				t.Error("expected 'text_embeddings' input key")
			}
			return map[string]*Tensor{
				"last_hidden": fakeHidden,
				"eos_logits":  fakeEOS,
			}, nil
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"flow_lm_main": fake})

	seq, _ := NewTensor(make([]float32, 32), []int64{1, 1, 32})
	emb, _ := NewTensor(make([]float32, 1024), []int64{1, 1, 1024})

	lastHidden, eosLogits, err := e.FlowLMStep(context.Background(), seq, emb)
	if err != nil {
		t.Fatalf("FlowLMStep: %v", err)
	}

	// Verify last_hidden shape [1, 1024].
	hShape := lastHidden.Shape()
	if len(hShape) != 2 || hShape[0] != 1 || hShape[1] != 1024 {
		t.Errorf("last_hidden shape = %v, want [1 1024]", hShape)
	}

	// Verify eos_logits shape [1, 1].
	eShape := eosLogits.Shape()
	if len(eShape) != 2 || eShape[0] != 1 || eShape[1] != 1 {
		t.Errorf("eos_logits shape = %v, want [1 1]", eShape)
	}
}

func TestFlowLMStep_MissingLastHiddenOutput(t *testing.T) {
	eosData := []float32{-5.0}
	fakeEOS, _ := NewTensor(eosData, []int64{1, 1})

	fake := &fakeRunner{
		name: "flow_lm_main",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			return map[string]*Tensor{"eos_logits": fakeEOS}, nil
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"flow_lm_main": fake})

	seq, _ := NewTensor(make([]float32, 32), []int64{1, 1, 32})
	emb, _ := NewTensor(make([]float32, 1024), []int64{1, 1, 1024})

	_, _, err := e.FlowLMStep(context.Background(), seq, emb)
	if err == nil {
		t.Fatal("expected error for missing last_hidden output")
	}
}

func TestFlowLMStep_MissingEOSLogitsOutput(t *testing.T) {
	hiddenData := make([]float32, 1024)
	fakeHidden, _ := NewTensor(hiddenData, []int64{1, 1024})

	fake := &fakeRunner{
		name: "flow_lm_main",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			return map[string]*Tensor{"last_hidden": fakeHidden}, nil
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"flow_lm_main": fake})

	seq, _ := NewTensor(make([]float32, 32), []int64{1, 1, 32})
	emb, _ := NewTensor(make([]float32, 1024), []int64{1, 1, 1024})

	_, _, err := e.FlowLMStep(context.Background(), seq, emb)
	if err == nil {
		t.Fatal("expected error for missing eos_logits output")
	}
}

// ---------------------------------------------------------------------------
// Unit tests for NewBOSSequence
// ---------------------------------------------------------------------------

func TestNewBOSSequence(t *testing.T) {
	bos := NewBOSSequence()

	// Shape must be [1, 1, 32].
	shape := bos.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 32 {
		t.Fatalf("BOS shape = %v, want [1 1 32]", shape)
	}

	// All values must be NaN.
	data, err := ExtractFloat32(bos)
	if err != nil {
		t.Fatalf("ExtractFloat32: %v", err)
	}
	for i, v := range data {
		if !isNaN(v) {
			t.Errorf("BOS[%d] = %v, want NaN", i, v)
		}
	}
}

// ---------------------------------------------------------------------------
// Unit tests for AppendLatentFrame
// ---------------------------------------------------------------------------

func TestAppendLatentFrame(t *testing.T) {
	// Start with BOS [1, 1, 32].
	seq := NewBOSSequence()

	// Create a frame [1, 1, 32] with known values.
	frameData := make([]float32, 32)
	for i := range frameData {
		frameData[i] = float32(i)
	}
	frame, err := NewTensor(frameData, []int64{1, 1, 32})
	if err != nil {
		t.Fatalf("NewTensor frame: %v", err)
	}

	// Append → should give [1, 2, 32].
	result, err := AppendLatentFrame(seq, frame)
	if err != nil {
		t.Fatalf("AppendLatentFrame: %v", err)
	}

	shape := result.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != 32 {
		t.Fatalf("shape = %v, want [1 2 32]", shape)
	}

	data, err := ExtractFloat32(result)
	if err != nil {
		t.Fatalf("ExtractFloat32: %v", err)
	}
	// First 32 values are NaN (BOS), next 32 are 0..31.
	for i := 0; i < 32; i++ {
		if !isNaN(data[i]) {
			t.Errorf("data[%d] = %v, want NaN", i, data[i])
		}
	}
	for i := 0; i < 32; i++ {
		if data[32+i] != float32(i) {
			t.Errorf("data[%d] = %v, want %v", 32+i, data[32+i], float32(i))
		}
	}
}

func TestAppendLatentFrame_GrowsSequence(t *testing.T) {
	seq := NewBOSSequence()

	// Append 3 frames → [1, 4, 32].
	for step := 0; step < 3; step++ {
		frame, _ := NewTensor(make([]float32, 32), []int64{1, 1, 32})
		var err error
		seq, err = AppendLatentFrame(seq, frame)
		if err != nil {
			t.Fatalf("step %d: AppendLatentFrame: %v", step, err)
		}
	}

	shape := seq.Shape()
	if shape[1] != 4 {
		t.Errorf("after 3 appends, shape[1] = %d, want 4", shape[1])
	}
}

// ---------------------------------------------------------------------------
// Unit tests for EOSDetected
// ---------------------------------------------------------------------------

func TestEOSDetected(t *testing.T) {
	tests := []struct {
		name      string
		logit     float32
		threshold float64
		want      bool
	}{
		{"above threshold → EOS", -3.0, -4.0, true},
		{"at threshold → EOS", -4.0, -4.0, false}, // strict >
		{"below threshold → no EOS", -5.0, -4.0, false},
		{"positive logit → EOS", 1.0, -4.0, true},
		{"zero logit → EOS", 0.0, -4.0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, _ := NewTensor([]float32{tt.logit}, []int64{1, 1})
			got := EOSDetected(tensor, tt.threshold)
			if got != tt.want {
				t.Errorf("EOSDetected(logit=%v, threshold=%v) = %v, want %v",
					tt.logit, tt.threshold, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Unit tests for Engine.FlowLMFlow (Euler flow integration / LSD decode)
// ---------------------------------------------------------------------------

func TestFlowLMFlow_MissingGraph(t *testing.T) {
	e := engineWithFakeRunners(map[string]runnerIface{})
	hidden, _ := NewTensor(make([]float32, 1024), []int64{1, 1024})

	_, err := e.FlowLMFlow(context.Background(), hidden, 0.7, 1)
	if err == nil {
		t.Fatal("expected error when flow_lm_flow graph is absent")
	}
}

func TestFlowLMFlow_PropagatesRunnerError(t *testing.T) {
	fake := &fakeRunner{
		name: "flow_lm_flow",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			return nil, fmt.Errorf("ort failure")
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"flow_lm_flow": fake})
	hidden, _ := NewTensor(make([]float32, 1024), []int64{1, 1024})

	_, err := e.FlowLMFlow(context.Background(), hidden, 0.7, 1)
	if err == nil {
		t.Fatal("expected error propagated from runner")
	}
}

func TestFlowLMFlow_SingleStep_ReturnsLatentFrame(t *testing.T) {
	// With 1 step: s=0, t=1, x starts as noise, result = x + flow_dir/1.
	// Fake runner returns a constant flow_direction.
	flowDir := make([]float32, 32)
	for i := range flowDir {
		flowDir[i] = 1.0 // constant flow direction
	}
	fakeFlow, _ := NewTensor(flowDir, []int64{1, 32})

	var capturedInputs map[string]*Tensor
	fake := &fakeRunner{
		name: "flow_lm_flow",
		fn: func(_ context.Context, inputs map[string]*Tensor) (map[string]*Tensor, error) {
			capturedInputs = inputs
			return map[string]*Tensor{"flow_direction": fakeFlow}, nil
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"flow_lm_flow": fake})
	hidden, _ := NewTensor(make([]float32, 1024), []int64{1, 1024})

	result, err := e.FlowLMFlow(context.Background(), hidden, 0.0, 1) // temp=0 → noise is all zeros
	if err != nil {
		t.Fatalf("FlowLMFlow: %v", err)
	}

	// Verify output shape [1, 1, 32].
	shape := result.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 32 {
		t.Errorf("result shape = %v, want [1 1 32]", shape)
	}

	// With temp=0, noise is all zeros. Result = 0 + 1.0/1 = 1.0 for each element.
	data, _ := ExtractFloat32(result)
	for i, v := range data {
		if v != 1.0 {
			t.Errorf("data[%d] = %v, want 1.0", i, v)
		}
	}

	// Verify input tensor names.
	for _, key := range []string{"condition", "s", "t", "x"} {
		if _, ok := capturedInputs[key]; !ok {
			t.Errorf("missing input key %q", key)
		}
	}

	// Verify s=0, t=1 for single step.
	sData, _ := ExtractFloat32(capturedInputs["s"])
	tData, _ := ExtractFloat32(capturedInputs["t"])
	if sData[0] != 0.0 {
		t.Errorf("s = %v, want 0.0", sData[0])
	}
	if tData[0] != 1.0 {
		t.Errorf("t = %v, want 1.0", tData[0])
	}
}

func TestFlowLMFlow_MultiStep_Arithmetic(t *testing.T) {
	// With 2 steps and temp=0 (noise = zeros):
	// Step 0: s=0, t=0.5, x = 0 + flow_dir/2
	// Step 1: s=0.5, t=1.0, x = prev + flow_dir/2
	// If flow_dir is always [2.0, 2.0, ...], result = 0 + 2/2 + 2/2 = 2.0
	callCount := 0
	fake := &fakeRunner{
		name: "flow_lm_flow",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			callCount++
			flowDir := make([]float32, 32)
			for i := range flowDir {
				flowDir[i] = 2.0
			}
			out, _ := NewTensor(flowDir, []int64{1, 32})
			return map[string]*Tensor{"flow_direction": out}, nil
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"flow_lm_flow": fake})
	hidden, _ := NewTensor(make([]float32, 1024), []int64{1, 1024})

	result, err := e.FlowLMFlow(context.Background(), hidden, 0.0, 2) // temp=0, 2 steps
	if err != nil {
		t.Fatalf("FlowLMFlow: %v", err)
	}

	if callCount != 2 {
		t.Errorf("runner called %d times, want 2", callCount)
	}

	data, _ := ExtractFloat32(result)
	for i, v := range data {
		if v != 2.0 {
			t.Errorf("data[%d] = %v, want 2.0", i, v)
		}
	}
}

func TestFlowLMFlow_MissingFlowDirectionOutput(t *testing.T) {
	fake := &fakeRunner{
		name: "flow_lm_flow",
		fn: func(_ context.Context, _ map[string]*Tensor) (map[string]*Tensor, error) {
			return map[string]*Tensor{}, nil
		},
	}
	e := engineWithFakeRunners(map[string]runnerIface{"flow_lm_flow": fake})
	hidden, _ := NewTensor(make([]float32, 1024), []int64{1, 1024})

	_, err := e.FlowLMFlow(context.Background(), hidden, 0.7, 1)
	if err == nil {
		t.Fatal("expected error for missing flow_direction output")
	}
}

func isNaN(f float32) bool {
	return f != f
}
