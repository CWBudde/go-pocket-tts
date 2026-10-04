package native

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/ops"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
	"github.com/cwbudde/go-pocket-tts/internal/text"
	"github.com/cwbudde/go-pocket-tts/internal/tokenizer"
)

// nativePythonParityFixtureEnv names one more fixture to run next to the
// committed ones in testdata/python_parity/ (scripts/dump_python_parity.py).
const nativePythonParityFixtureEnv = "POCKETTTS_NATIVE_PY_FIXTURE"

type nativePythonParityFixture struct {
	Source pythonParitySource        `json:"source"`
	Text   *textPythonParityCase     `json:"text,omitempty"`
	FlowLM *flowLMPythonParityCase   `json:"flow_lm_prefill_step,omitempty"`
	Voice  *voicePrefillPythonParity `json:"voice_prefill_step,omitempty"`
	Mimi   []mimiPythonParityCase    `json:"mimi,omitempty"`
}

type pythonParitySource struct {
	Upstream string `json:"upstream"`
	Config   string `json:"config"`
	Voice    string `json:"voice,omitempty"`
	// Files names the local files of a custom (--config) fixture, which
	// cannot be resolved by language name.
	Files *pythonParityFiles `json:"files,omitempty"`
}

type pythonParityFiles struct {
	Weights   string `json:"weights"`
	Tokenizer string `json:"tokenizer"`
	Voices    string `json:"voices,omitempty"`
}

type textPythonParityCase struct {
	Raw            string     `json:"raw"`
	Prepared       string     `json:"prepared"`
	Tokens         []int64    `json:"tokens"`
	EmbeddingsHead tensorJSON `json:"embeddings_head"`
}

type flowLMPythonParityCase struct {
	Tokens             []int64     `json:"tokens"`
	StepLatent         tensorJSON  `json:"step_latent"`
	PromptLayerOffsets []int64     `json:"prompt_layer_offsets,omitempty"`
	StepLayerOffsets   []int64     `json:"step_layer_offsets,omitempty"`
	StepLastHidden     *tensorJSON `json:"step_last_hidden,omitempty"`
	StepEOSLogits      *tensorJSON `json:"step_eos_logits,omitempty"`
}

type voicePrefillPythonParity struct {
	VoiceFormat        string      `json:"voice_format"`
	VoiceLayerOffsets  []int64     `json:"voice_layer_offsets"`
	PromptLayerOffsets []int64     `json:"prompt_layer_offsets"`
	StepLayerOffsets   []int64     `json:"step_layer_offsets"`
	StepLatent         tensorJSON  `json:"step_latent"`
	StepLastHidden     *tensorJSON `json:"step_last_hidden"`
	StepEOSLogits      *tensorJSON `json:"step_eos_logits"`
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

// pythonParityLanguage is the model config of a fixture and the local files
// its checks run on.
type pythonParityLanguage struct {
	mc            *modelcfg.ModelConfig
	modelPath     string
	tokenizerPath string
	voiceDir      string
}

// TestPythonParity runs every fixture in testdata/python_parity/ (plus
// $POCKETTTS_NATIVE_PY_FIXTURE) against the local model of the fixture's
// config. A fixture whose model is not downloaded is skipped.
func TestPythonParity(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("testdata", "python_parity", "*.json"))
	if err != nil {
		t.Fatal(err)
	}

	if extra := os.Getenv(nativePythonParityFixtureEnv); extra != "" {
		paths = append(paths, extra)
	}

	if len(paths) == 0 {
		t.Fatal("no Python parity fixtures in testdata/python_parity")
	}

	for _, path := range paths {
		t.Run(strings.TrimSuffix(filepath.Base(path), ".json"), func(t *testing.T) {
			fixture := loadNativePythonParityFixture(t, path)
			lang := pythonParityLanguageFor(t, fixture.Source)

			m, err := LoadModelFromSafetensors(lang.modelPath, ConfigFor(lang.mc))
			if err != nil {
				t.Fatalf("load model: %v", err)
			}
			defer m.Close()

			if fixture.Text != nil {
				t.Run("text", func(t *testing.T) { checkTextParity(t, m, lang, fixture.Text) })
			}

			if fixture.FlowLM != nil {
				t.Run("flow_lm_prefill_step", func(t *testing.T) { checkFlowLMParity(t, m, fixture.FlowLM) })
			}

			if fixture.Voice != nil {
				if fixture.Text == nil {
					t.Fatal("voice_prefill_step needs the text case for its tokens")
				}

				t.Run("voice_prefill_step", func(t *testing.T) {
					checkVoicePrefillParity(t, m, lang, fixture.Source.Voice, fixture.Text.Tokens, fixture.Voice)
				})
			}

			for _, tc := range fixture.Mimi {
				t.Run("mimi/"+tc.Name, func(t *testing.T) { checkMimiParity(t, m, tc) })
			}
		})
	}
}

// pythonParityLanguageFor resolves a fixture's config to the model config and
// the local files to check, or skips when the model is not downloaded. An
// embedded config name uses the language's local layout; a custom config file
// uses the files the fixture names.
func pythonParityLanguageFor(t *testing.T, src pythonParitySource) pythonParityLanguage {
	t.Helper()

	mc, err := modelcfg.Lookup(src.Config)
	if err == nil {
		paths := config.PathsForLanguage(src.Config)

		return pythonParityLanguage{
			mc:            mc,
			modelPath:     requireRepoFile(t, paths.ModelPath),
			tokenizerPath: paths.TokenizerModel,
			voiceDir:      filepath.Dir(paths.VoiceManifest),
		}
	}

	if src.Files == nil {
		t.Fatalf("fixture config %q is not an embedded model config and the fixture names no local files "+
			"(dump it with --config --weights --tokenizer)", src.Config)
	}

	mc, err = modelcfg.LoadCustom(requireRepoFile(t, src.Config))
	if err != nil {
		t.Fatalf("load fixture config %q: %v", src.Config, err)
	}

	return pythonParityLanguage{
		mc:            mc,
		modelPath:     requireRepoFile(t, src.Files.Weights),
		tokenizerPath: src.Files.Tokenizer,
		voiceDir:      src.Files.Voices,
	}
}

// checkTextParity checks the Go text preparation and tokenizer on the
// fixture's text and the text embeddings of its first tokens.
func checkTextParity(t *testing.T, m *Model, lang pythonParityLanguage, tc *textPythonParityCase) {
	t.Helper()

	prepared, _, err := text.PrepareText(tc.Raw, text.OptionsFor(lang.mc))
	if err != nil {
		t.Fatalf("PrepareText: %v", err)
	}

	if prepared != tc.Prepared {
		t.Errorf("prepared text = %q, want %q", prepared, tc.Prepared)
	}

	tok, err := tokenizer.Load(requireRepoFile(t, lang.tokenizerPath), lang.mc.FlowLM.LookupTable.NBins)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	ids, err := tok.Encode(tc.Prepared)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	if !slices.Equal(ids, tc.Tokens) {
		t.Errorf("tokens = %v, want %v", ids, tc.Tokens)
	}

	want, err := tc.EmbeddingsHead.tensor()
	if err != nil {
		t.Fatalf("embeddings fixture: %v", err)
	}

	got, err := m.TextEmbeddings(tc.Tokens[:want.Shape()[1]])
	if err != nil {
		t.Fatalf("text embeddings: %v", err)
	}

	assertTensorParity(t, "text_embeddings_head", got, want, ops.Tolerance{Abs: 1e-6, Rel: 1e-6})
}

func checkFlowLMParity(t *testing.T, m *Model, tc *flowLMPythonParityCase) {
	t.Helper()

	textEmb, err := m.TextEmbeddings(tc.Tokens)
	if err != nil {
		t.Fatalf("text embeddings: %v", err)
	}

	state, err := m.NewFlowState()
	if err != nil {
		t.Fatalf("new flow state: %v", err)
	}

	err = m.PromptFlow(state, textEmb)
	if err != nil {
		t.Fatalf("prompt flow: %v", err)
	}

	if len(tc.PromptLayerOffsets) > 0 {
		assertFlowLayerOffsets(t, "prompt", state, tc.PromptLayerOffsets)
	}

	checkFlowStepParity(t, m, state, tc.StepLatent, tc.StepLayerOffsets, tc.StepLastHidden, tc.StepEOSLogits)
}

// checkVoicePrefillParity conditions the flow state on the language's voice
// the way the native runtime does (internal/tts prepareFlowState), prompts
// the text and runs one step.
func checkVoicePrefillParity(
	t *testing.T,
	m *Model,
	lang pythonParityLanguage,
	voice string,
	tokens []int64,
	tc *voicePrefillPythonParity,
) {
	t.Helper()

	voicePath := requireRepoFile(t, lang.voiceDir, voice+".safetensors")

	textEmb, err := m.TextEmbeddings(tokens)
	if err != nil {
		t.Fatalf("text embeddings: %v", err)
	}

	var state *FlowLMState

	switch tc.VoiceFormat {
	case "model_state":
		vs, err := safetensors.LoadVoiceModelState(voicePath)
		if err != nil {
			t.Fatalf("load voice: %v", err)
		}

		state, err = m.NewFlowStateFromVoiceModelState(vs)
		if err != nil {
			t.Fatalf("NewFlowStateFromVoiceModelState: %v", err)
		}

		assertFlowLayerOffsets(t, "voice", state, tc.VoiceLayerOffsets)
	case "audio_prompt":
		// The runtime prompts voice and text in one pass, so there is no
		// voice-only state to compare.
		data, shape, err := safetensors.LoadVoiceEmbedding(voicePath)
		if err != nil {
			t.Fatalf("load voice: %v", err)
		}

		voiceEmb, err := tensor.New(data, shape)
		if err != nil {
			t.Fatal(err)
		}

		voiceEmb, err = m.VoicePrompt(voiceEmb)
		if err != nil {
			t.Fatalf("voice prompt: %v", err)
		}

		textEmb, err = tensor.Concat([]*tensor.Tensor{voiceEmb, textEmb}, 1)
		if err != nil {
			t.Fatal(err)
		}

		state, err = m.NewFlowState()
		if err != nil {
			t.Fatalf("new flow state: %v", err)
		}
	default:
		t.Fatalf("unknown voice_format %q", tc.VoiceFormat)
	}

	err = m.PromptFlow(state, textEmb)
	if err != nil {
		t.Fatalf("prompt flow: %v", err)
	}

	assertFlowLayerOffsets(t, "prompt", state, tc.PromptLayerOffsets)
	checkFlowStepParity(t, m, state, tc.StepLatent, tc.StepLayerOffsets, tc.StepLastHidden, tc.StepEOSLogits)
}

func checkFlowStepParity(
	t *testing.T,
	m *Model,
	state *FlowLMState,
	stepLatentJSON tensorJSON,
	stepOffsets []int64,
	lastHidden, eosLogits *tensorJSON,
) {
	t.Helper()

	stepLatent, err := stepLatentJSON.tensor()
	if err != nil {
		t.Fatalf("step latent: %v", err)
	}

	last, eos, err := runFlowStepForParity(m.flow, state, stepLatent)
	if err != nil {
		t.Fatalf("run flow step: %v", err)
	}

	if len(stepOffsets) > 0 {
		assertFlowLayerOffsets(t, "step", state, stepOffsets)
	}

	tol := ops.Tolerance{Abs: 2e-4, Rel: 5e-3}

	if lastHidden != nil {
		want, err := lastHidden.tensor()
		if err != nil {
			t.Fatalf("step last hidden fixture: %v", err)
		}

		assertTensorParity(t, "flow_lm_step_last_hidden", last, want, tol)
	}

	if eosLogits != nil {
		want, err := eosLogits.tensor()
		if err != nil {
			t.Fatalf("step eos logits fixture: %v", err)
		}

		assertTensorParity(t, "flow_lm_step_eos_logits", eos, want, tol)
	}
}

func checkMimiParity(t *testing.T, m *Model, tc mimiPythonParityCase) {
	t.Helper()

	convTol := ops.Tolerance{Abs: 2e-4, Rel: 1e-3}
	deconvTol := ops.Tolerance{Abs: 2e-4, Rel: 5e-2}

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
}

func loadNativePythonParityFixture(t *testing.T, path string) nativePythonParityFixture {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}

	var fixture nativePythonParityFixture

	err = json.Unmarshal(data, &fixture)
	if err != nil {
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

// assertTensorParity checks every element like numpy.allclose:
// |got − want| ≤ tol.Abs + tol.Rel·|want|. CompareTensor's Pass needs the
// maximum absolute and relative errors to both stay within tol, which fails
// on float32 noise around an element close to zero.
func assertTensorParity(t *testing.T, name string, got, want *tensor.Tensor, tol ops.Tolerance) {
	t.Helper()

	rep, err := CompareTensor(name, got, want, tol)
	if err != nil {
		t.Fatalf("compare %s: %v", name, err)
	}

	if !rep.ShapeMatch {
		t.Fatalf("%s shape mismatch: got %v want %v", name, got.Shape(), want.Shape())
	}

	gd, wd := got.RawData(), want.RawData()
	for i := range wd {
		g, w := float64(gd[i]), float64(wd[i])
		if math.IsNaN(g) || math.Abs(g-w) > tol.Abs+tol.Rel*math.Abs(w) {
			t.Fatalf("%s parity failed at element %d: got %g want %g (max abs err %g, tolerance %+v)",
				name, i, g, w, rep.MaxAbsErr, tol)
		}
	}
}
