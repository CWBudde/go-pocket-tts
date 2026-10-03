package modelcfg

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func loadTestdata(t *testing.T, name string) *ModelConfig {
	t.Helper()

	cfg, err := Load(filepath.Join("testdata", name+".yaml"))
	if err != nil {
		t.Fatalf("Load(%s): %v", name, err)
	}

	return cfg
}

func TestLoad_English202601(t *testing.T) {
	cfg := loadTestdata(t, "english_2026-01")

	if !cfg.PadWithSpacesForShortInputs {
		t.Error("PadWithSpacesForShortInputs = false; want true")
	}

	if cfg.FlowLM.InsertBOSBeforeVoice {
		t.Error("FlowLM.InsertBOSBeforeVoice = true; want false")
	}

	if got := cfg.MimiInnerDim(); got != 512 {
		t.Errorf("MimiInnerDim() = %d; want 512", got)
	}

	if cfg.FlowLM.Flow.Type != FlowTypeLSD {
		t.Errorf("FlowLM.Flow.Type = %q; want default %q", cfg.FlowLM.Flow.Type, FlowTypeLSD)
	}

	if got := cfg.NumTimeConds(); got != 2 {
		t.Errorf("NumTimeConds() = %d; want 2", got)
	}

	if cfg.FlowLM.LookupTable.Tokenizer != TokenizerTokenizers {
		t.Errorf("Tokenizer = %q; want %q", cfg.FlowLM.LookupTable.Tokenizer, TokenizerTokenizers)
	}

	if cfg.FlowLM.Transformer.NumLayers != 6 || cfg.FlowLM.Transformer.DModel != 1024 {
		t.Errorf("FlowLM.Transformer = %+v; want 6 layers, d_model 1024", cfg.FlowLM.Transformer)
	}

	if cfg.Mimi.Transformer.MaxPeriod != 10000 {
		t.Errorf("Mimi.Transformer.MaxPeriod = %v; want default 10000", cfg.Mimi.Transformer.MaxPeriod)
	}

	if want := "hf://kyutai/pocket-tts-without-voice-cloning/languages/english_2026-01/model.safetensors@d29db7978e464fb90cb3359ee0c69a273b9142cc"; cfg.WeightsPathWithoutVoiceCloning != want {
		t.Errorf("WeightsPathWithoutVoiceCloning = %q; want %q", cfg.WeightsPathWithoutVoiceCloning, want)
	}

	// Upstream defaults for fields the YAML leaves out.
	if cfg.DefaultTemperature != 0.3 {
		t.Errorf("DefaultTemperature = %v; want 0.3", cfg.DefaultTemperature)
	}

	if !cfg.AppendTerminalPunctuation || !cfg.CapitalizeFirstLetter {
		t.Error("AppendTerminalPunctuation and CapitalizeFirstLetter must default to true")
	}

	if cfg.RemoveSemicolons {
		t.Error("RemoveSemicolons = true; want false")
	}

	if cfg.ModelRecommendedFramesAfterEOS != nil {
		t.Errorf("ModelRecommendedFramesAfterEOS = %v; want nil", *cfg.ModelRecommendedFramesAfterEOS)
	}
}

func TestLoad_German(t *testing.T) {
	cfg := loadTestdata(t, "german")

	if cfg.PadWithSpacesForShortInputs {
		t.Error("PadWithSpacesForShortInputs = true; want false")
	}

	if !cfg.FlowLM.InsertBOSBeforeVoice {
		t.Error("FlowLM.InsertBOSBeforeVoice = false; want true")
	}

	if got := cfg.MimiInnerDim(); got != 32 {
		t.Errorf("MimiInnerDim() = %d; want 32", got)
	}

	if !cfg.RemoveSemicolons {
		t.Error("RemoveSemicolons = false; want true")
	}

	wantReplace := map[string]string{
		"\"": "", "“": "", "”": "", "„": "", "«": "", "»": "",
		"’": "'", "‘": "'", "(": "", ")": "", "[": "", "]": "",
	}
	if len(cfg.ReplaceCharacters) != len(wantReplace) {
		t.Errorf("ReplaceCharacters has %d entries; want %d", len(cfg.ReplaceCharacters), len(wantReplace))
	}

	for from, to := range wantReplace {
		got, ok := cfg.ReplaceCharacters[from]
		if !ok || got != to {
			t.Errorf("ReplaceCharacters[%q] = %q (present %v); want %q", from, got, ok, to)
		}
	}

	if !strings.Contains(cfg.FlowLM.LookupTable.TokenizerPath, "languages/german/tokenizer.json") {
		t.Errorf("TokenizerPath = %q; want the german tokenizer.json", cfg.FlowLM.LookupTable.TokenizerPath)
	}
}

func TestLoad_EnglishDrifting(t *testing.T) {
	cfg := loadTestdata(t, "english_drifting_26-09")

	if cfg.FlowLM.Flow.Type != FlowTypeDrifting {
		t.Errorf("FlowLM.Flow.Type = %q; want %q", cfg.FlowLM.Flow.Type, FlowTypeDrifting)
	}

	if got := cfg.NumTimeConds(); got != 0 {
		t.Errorf("NumTimeConds() = %d; want 0", got)
	}

	if cfg.DefaultTemperature != 0.3 {
		t.Errorf("DefaultTemperature = %v; want 0.3", cfg.DefaultTemperature)
	}
}

func TestNumTimeConds(t *testing.T) {
	for flowType, want := range map[string]int{
		FlowTypeLSD:          2,
		FlowTypeFlowMatching: 1,
		FlowTypeDrifting:     0,
	} {
		cfg := &ModelConfig{FlowLM: FlowLMConfig{Flow: FlowConfig{Type: flowType}}}
		if got := cfg.NumTimeConds(); got != want {
			t.Errorf("NumTimeConds() for %q = %d; want %d", flowType, got, want)
		}
	}
}

// Upstream uses `config.mimi.inner_dim or config.mimi.seanet.dimension`.
func TestMimiInnerDim_FallsBackToSEANetDimension(t *testing.T) {
	cfg := &ModelConfig{Mimi: MimiConfig{
		SEANet:    SEANetConfig{Dimension: 512},
		Quantizer: QuantizerConfig{Dimension: 32},
	}}
	if got := cfg.MimiInnerDim(); got != 512 {
		t.Errorf("MimiInnerDim() = %d; want 512", got)
	}
}

// Upstream: frames_after_eos or model_recommended_frames_after_eos or the
// per-chunk guess.
func TestFramesAfterEOS(t *testing.T) {
	two, zero := 2, 0

	for _, tc := range []struct {
		name string
		cfg  *ModelConfig
		want int
	}{
		{"nil config keeps the guess", nil, 5},
		{"no recommendation keeps the guess", &ModelConfig{}, 5},
		{"recommendation wins", &ModelConfig{ModelRecommendedFramesAfterEOS: &two}, 2},
		{"zero recommendation wins", &ModelConfig{ModelRecommendedFramesAfterEOS: &zero}, 0},
	} {
		if got := tc.cfg.FramesAfterEOS(5); got != tc.want {
			t.Errorf("%s: FramesAfterEOS(5) = %d; want %d", tc.name, got, tc.want)
		}
	}
}

func readTestdata(t *testing.T, name string) string {
	t.Helper()

	data, err := os.ReadFile(filepath.Join("testdata", name+".yaml"))
	if err != nil {
		t.Fatal(err)
	}

	return string(data)
}

func TestParse_RejectsUnknownField(t *testing.T) {
	data := readTestdata(t, "german") + "\nsome_new_option: true\n"

	_, err := Parse([]byte(data))
	if err == nil || !strings.Contains(err.Error(), "some_new_option") {
		t.Fatalf("Parse() error = %v; want an error naming the unknown field", err)
	}
}

// Upstream's yaml.safe_load raises on a second document instead of ignoring it.
func TestParse_RejectsMultipleDocuments(t *testing.T) {
	data := readTestdata(t, "german") + "\n---\nsome_new_option: true\n"

	_, err := Parse([]byte(data))
	if err == nil || !strings.Contains(err.Error(), "single YAML document") {
		t.Fatalf("Parse() error = %v; want an error rejecting the second document", err)
	}
}

func TestParse_RejectsMissingRequiredField(t *testing.T) {
	data := strings.Replace(readTestdata(t, "german"), "    num_heads: 16\n", "", 1)

	_, err := Parse([]byte(data))
	if err == nil || !strings.Contains(err.Error(), "flow_lm.transformer.num_heads") {
		t.Fatalf("Parse() error = %v; want an error naming flow_lm.transformer.num_heads", err)
	}
}

func TestParse_RejectsMissingSection(t *testing.T) {
	_, err := Parse([]byte("weights_path: x\n"))
	if err == nil {
		t.Fatal("Parse() without flow_lm/mimi succeeded; want an error")
	}
}

func TestParse_RejectsInvalidFlowType(t *testing.T) {
	data := strings.Replace(readTestdata(t, "english_drifting_26-09"), "type: drifting", "type: diffusion", 1)

	_, err := Parse([]byte(data))
	if err == nil || !strings.Contains(err.Error(), "diffusion") {
		t.Fatalf("Parse() error = %v; want an error naming the invalid flow type", err)
	}
}

func TestParse_RejectsInvalidTokenizer(t *testing.T) {
	data := strings.Replace(readTestdata(t, "german"), "tokenizer: tokenizers", "tokenizer: bpe", 1)

	_, err := Parse([]byte(data))
	if err == nil || !strings.Contains(err.Error(), "bpe") {
		t.Fatalf("Parse() error = %v; want an error naming the invalid tokenizer", err)
	}
}

func TestParse_ExplicitValuesOverrideDefaults(t *testing.T) {
	data := readTestdata(t, "german") + "\ndefault_temperature: 0.7\ncapitalize_first_letter: false\nmodel_recommended_frames_after_eos: 3\n"

	cfg, err := Parse([]byte(data))
	if err != nil {
		t.Fatalf("Parse(): %v", err)
	}

	if cfg.DefaultTemperature != 0.7 {
		t.Errorf("DefaultTemperature = %v; want 0.7", cfg.DefaultTemperature)
	}

	if cfg.CapitalizeFirstLetter {
		t.Error("CapitalizeFirstLetter = true; want false")
	}

	if cfg.ModelRecommendedFramesAfterEOS == nil || *cfg.ModelRecommendedFramesAfterEOS != 3 {
		t.Errorf("ModelRecommendedFramesAfterEOS = %v; want 3", cfg.ModelRecommendedFramesAfterEOS)
	}
}

func TestLoad_MissingFile(t *testing.T) {
	_, err := Load(filepath.Join(t.TempDir(), "nope.yaml"))
	if err == nil {
		t.Fatal("Load() on a missing file succeeded; want an error")
	}
}
