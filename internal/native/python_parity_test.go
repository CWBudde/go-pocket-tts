package native

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/runtime/ops"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
)

const nativePythonParityFixtureEnv = "POCKETTTS_NATIVE_PY_FIXTURE"

type nativePythonParityFixture struct {
	FlowLM *flowLMPythonParityCase `json:"flow_lm_prefill_step,omitempty"`
	Mimi   []mimiPythonParityCase  `json:"mimi,omitempty"`
}

type flowLMPythonParityCase struct {
	Tokens             []int64     `json:"tokens"`
	StepLatent         tensorJSON  `json:"step_latent"`
	PromptLayerOffsets []int64     `json:"prompt_layer_offsets,omitempty"`
	StepLayerOffsets   []int64     `json:"step_layer_offsets,omitempty"`
	StepLastHidden     *tensorJSON `json:"step_last_hidden,omitempty"`
	StepEOSLogits      *tensorJSON `json:"step_eos_logits,omitempty"`
}

type mimiPythonParityCase struct {
	Name         string      `json:"name"`
	Latent       tensorJSON  `json:"latent"`
	LatentToMimi *tensorJSON `json:"latent_to_mimi,omitempty"`
	MimiDecode   *tensorJSON `json:"mimi_decode,omitempty"`
}

type tensorJSON struct {
	Shape []int64   `json:"shape"`
	Data  []float32 `json:"data"`
}

func TestPythonParity_FlowLMPrefillAndStep(t *testing.T) {
	fixture := loadNativePythonParityFixture(t)
	if fixture.FlowLM == nil {
		t.Skip("fixture does not contain flow_lm_prefill_step")
	}

	ckpt := requireCheckpoint(t)
	m, err := LoadModelFromSafetensors(ckpt, DefaultConfig())
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	defer m.Close()

	tc := fixture.FlowLM
	textEmb, err := m.TextEmbeddings(tc.Tokens)
	if err != nil {
		t.Fatalf("text embeddings: %v", err)
	}

	state, err := m.NewFlowState()
	if err != nil {
		t.Fatalf("new flow state: %v", err)
	}

	if err := m.PromptFlow(state, textEmb); err != nil {
		t.Fatalf("prompt flow: %v", err)
	}

	if len(tc.PromptLayerOffsets) > 0 {
		assertFlowLayerOffsets(t, "prompt", state, tc.PromptLayerOffsets)
	}

	stepLatent, err := tc.StepLatent.tensor()
	if err != nil {
		t.Fatalf("step latent: %v", err)
	}

	last, eos, err := runFlowStepForParity(m.flow, state, stepLatent)
	if err != nil {
		t.Fatalf("run flow step: %v", err)
	}

	if len(tc.StepLayerOffsets) > 0 {
		assertFlowLayerOffsets(t, "step", state, tc.StepLayerOffsets)
	}

	tol := ops.Tolerance{Abs: 2e-4, Rel: 5e-3}
	if tc.StepLastHidden != nil {
		want, err := tc.StepLastHidden.tensor()
		if err != nil {
			t.Fatalf("step last hidden fixture: %v", err)
		}

		assertTensorParity(t, "flow_lm_step_last_hidden", last, want, tol)
	}

	if tc.StepEOSLogits != nil {
		want, err := tc.StepEOSLogits.tensor()
		if err != nil {
			t.Fatalf("step eos logits fixture: %v", err)
		}

		assertTensorParity(t, "flow_lm_step_eos_logits", eos, want, tol)
	}
}

func TestPythonParity_LatentToMimiAndDecode(t *testing.T) {
	fixture := loadNativePythonParityFixture(t)
	if len(fixture.Mimi) == 0 {
		t.Skip("fixture does not contain mimi cases")
	}

	ckpt := requireCheckpoint(t)
	m, err := LoadModelFromSafetensors(ckpt, DefaultConfig())
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	defer m.Close()

	convTol := ops.Tolerance{Abs: 2e-4, Rel: 1e-3}
	deconvTol := ops.Tolerance{Abs: 2e-4, Rel: 5e-2}

	for _, tc := range fixture.Mimi {
		t.Run(tc.Name, func(t *testing.T) {
			latent, err := tc.Latent.tensor()
			if err != nil {
				t.Fatalf("latent fixture: %v", err)
			}

			mimiLatent, err := m.LatentToMimi(latent)
			if err != nil {
				t.Fatalf("latent_to_mimi: %v", err)
			}

			if tc.LatentToMimi != nil {
				want, err := tc.LatentToMimi.tensor()
				if err != nil {
					t.Fatalf("latent_to_mimi fixture: %v", err)
				}

				assertTensorParity(t, "latent_to_mimi", mimiLatent, want, convTol)
			}

			if tc.MimiDecode != nil {
				audio, err := m.MimiDecode(mimiLatent)
				if err != nil {
					t.Fatalf("mimi_decode: %v", err)
				}

				want, err := tc.MimiDecode.tensor()
				if err != nil {
					t.Fatalf("mimi_decode fixture: %v", err)
				}

				assertTensorParity(t, "mimi_decode", audio, want, deconvTol)
			}
		})
	}
}

func loadNativePythonParityFixture(t *testing.T) nativePythonParityFixture {
	t.Helper()

	path := os.Getenv(nativePythonParityFixtureEnv)
	if path == "" {
		t.Skipf("set %s to a Python-generated native runtime parity fixture", nativePythonParityFixtureEnv)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}

	var fixture nativePythonParityFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}

	return fixture
}

func (tj tensorJSON) tensor() (*tensor.Tensor, error) {
	return tensor.New(tj.Data, tj.Shape)
}

func runFlowStepForParity(f *FlowLM, state *FlowLMState, stepLatent *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	seq, err := replaceNaNWithVector(stepLatent, f.bosEmb)
	if err != nil {
		return nil, nil, err
	}

	in, err := f.inputProj.Forward(seq)
	if err != nil {
		return nil, nil, err
	}

	x, err := f.transformer.step(in, state.transformer)
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

func assertFlowLayerOffsets(t *testing.T, phase string, state *FlowLMState, want []int64) {
	t.Helper()

	if state == nil || state.transformer == nil {
		t.Fatalf("%s state unavailable", phase)
	}

	if len(state.transformer.layers) != len(want) {
		t.Fatalf("%s layer count = %d, want %d", phase, len(state.transformer.layers), len(want))
	}

	for i, layer := range state.transformer.layers {
		if layer.offset != want[i] {
			t.Fatalf("%s layer %d offset = %d, want %d", phase, i, layer.offset, want[i])
		}
	}
}

func assertTensorParity(t *testing.T, name string, got, want *tensor.Tensor, tol ops.Tolerance) {
	t.Helper()

	rep, err := CompareTensor(name, got, want, tol)
	if err != nil {
		t.Fatalf("compare %s: %v", name, err)
	}

	if !rep.ShapeMatch {
		t.Fatalf("%s shape mismatch: got %v want %v", name, got.Shape(), want.Shape())
	}

	if !rep.Pass {
		t.Fatalf("%s parity failed: %+v", name, rep)
	}
}
