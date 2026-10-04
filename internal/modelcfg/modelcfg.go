// Package modelcfg mirrors the upstream PocketTTS model configuration
// (pocket_tts/config/<language>.yaml, schema in pocket_tts/utils/config.py).
//
// Parsing is strict like the upstream pydantic models (extra="forbid"):
// unknown keys and missing required keys are errors, and fields with an
// upstream default are pre-filled before decoding.
package modelcfg

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"slices"
	"strings"
	"unicode/utf8"

	"gopkg.in/yaml.v3"
)

// Sampler head types (flow_lm.flow.type).
const (
	FlowTypeLSD          = "lsd"
	FlowTypeFlowMatching = "flow_matching"
	FlowTypeDrifting     = "drifting"
)

// Tokenizer kinds (flow_lm.lookup_table.tokenizer).
const (
	TokenizerSentencePiece = "sentencepiece"
	TokenizerTokenizers    = "tokenizers"
)

// ModelConfig is the root of an upstream model config (upstream: Config).
type ModelConfig struct {
	FlowLM                         FlowLMConfig      `yaml:"flow_lm"`
	Mimi                           MimiConfig        `yaml:"mimi"`
	WeightsPath                    string            `yaml:"weights_path"`
	WeightsPathWithoutVoiceCloning string            `yaml:"weights_path_without_voice_cloning"`
	PadWithSpacesForShortInputs    bool              `yaml:"pad_with_spaces_for_short_inputs"`
	RemoveSemicolons               bool              `yaml:"remove_semicolons"`
	AppendTerminalPunctuation      bool              `yaml:"append_terminal_punctuation"`
	CapitalizeFirstLetter          bool              `yaml:"capitalize_first_letter"`
	ReplaceCharacters              map[string]string `yaml:"replace_characters"`
	ModelRecommendedFramesAfterEOS *int              `yaml:"model_recommended_frames_after_eos"`
	DefaultTemperature             float64           `yaml:"default_temperature"`

	// DefaultVoice, DefaultText and VoicesRevision are not part of the
	// upstream YAML (upstream keeps them in DEFAULT_VOICE_FOR_LANGUAGE,
	// DEFAULT_TEXT_FOR_LANGUAGE and the voice download code); the language
	// registry fills them in.
	DefaultVoice   string `yaml:"-"`
	DefaultText    string `yaml:"-"`
	VoicesRevision string `yaml:"-"`
}

// FlowLMConfig is the flow_lm section (upstream: FlowLMConfig).
type FlowLMConfig struct {
	DType                string                  `yaml:"dtype"`
	Flow                 FlowConfig              `yaml:"flow"`
	Transformer          FlowLMTransformerConfig `yaml:"transformer"`
	LookupTable          LookupTableConfig       `yaml:"lookup_table"`
	WeightsPath          string                  `yaml:"weights_path"`
	InsertBOSBeforeVoice bool                    `yaml:"insert_bos_before_voice"`
}

// FlowConfig describes the sampler head (upstream: FlowConfig).
type FlowConfig struct {
	Dim   int    `yaml:"dim"`
	Depth int    `yaml:"depth"`
	Type  string `yaml:"type"`
}

// FlowLMTransformerConfig is the FlowLM backbone (upstream: FlowLMTransformerConfig).
type FlowLMTransformerConfig struct {
	HiddenScale int `yaml:"hidden_scale"`
	MaxPeriod   int `yaml:"max_period"`
	DModel      int `yaml:"d_model"`
	NumHeads    int `yaml:"num_heads"`
	NumLayers   int `yaml:"num_layers"`
}

// LookupTableConfig is the text conditioner (upstream: LookupTable).
type LookupTableConfig struct {
	Dim           int    `yaml:"dim"`
	NBins         int    `yaml:"n_bins"`
	Tokenizer     string `yaml:"tokenizer"`
	TokenizerPath string `yaml:"tokenizer_path"`
}

// MimiConfig is the mimi section (upstream: MimiConfig).
type MimiConfig struct {
	DType       string                `yaml:"dtype"`
	SampleRate  int                   `yaml:"sample_rate"`
	Channels    int                   `yaml:"channels"`
	FrameRate   float64               `yaml:"frame_rate"`
	SEANet      SEANetConfig          `yaml:"seanet"`
	Transformer MimiTransformerConfig `yaml:"transformer"`
	Quantizer   QuantizerConfig       `yaml:"quantizer"`
	WeightsPath string                `yaml:"weights_path"`
	InnerDim    *int                  `yaml:"inner_dim"`
	OuterDim    *int                  `yaml:"outer_dim"`
}

// SEANetConfig is the Mimi convolutional encoder/decoder (upstream: SEANetConfig).
type SEANetConfig struct {
	Dimension          int    `yaml:"dimension"`
	Channels           int    `yaml:"channels"`
	NFilters           int    `yaml:"n_filters"`
	NResidualLayers    int    `yaml:"n_residual_layers"`
	Ratios             []int  `yaml:"ratios"`
	KernelSize         int    `yaml:"kernel_size"`
	ResidualKernelSize int    `yaml:"residual_kernel_size"`
	LastKernelSize     int    `yaml:"last_kernel_size"`
	DilationBase       int    `yaml:"dilation_base"`
	PadMode            string `yaml:"pad_mode"`
	Compress           int    `yaml:"compress"`
}

// MimiTransformerConfig is the Mimi transformer (upstream: MimiTransformerConfig).
type MimiTransformerConfig struct {
	DModel           int     `yaml:"d_model"`
	InputDimension   int     `yaml:"input_dimension"`
	OutputDimensions []int   `yaml:"output_dimensions"`
	NumHeads         int     `yaml:"num_heads"`
	NumLayers        int     `yaml:"num_layers"`
	LayerScale       float64 `yaml:"layer_scale"`
	Context          int     `yaml:"context"`
	MaxPeriod        float64 `yaml:"max_period"`
	DimFeedforward   int     `yaml:"dim_feedforward"`
}

// QuantizerConfig is the Mimi quantizer projection (upstream: QuantizerConfig).
type QuantizerConfig struct {
	Dimension       int `yaml:"dimension"`
	OutputDimension int `yaml:"output_dimension"`
}

// requiredKeys lists the dotted paths of every upstream field without a
// default value. Presence is checked on the raw YAML so that a legitimate
// zero value is not mistaken for a missing key.
var requiredKeys = []string{
	"flow_lm.dtype",
	"flow_lm.flow.dim",
	"flow_lm.flow.depth",
	"flow_lm.transformer.hidden_scale",
	"flow_lm.transformer.max_period",
	"flow_lm.transformer.d_model",
	"flow_lm.transformer.num_heads",
	"flow_lm.transformer.num_layers",
	"flow_lm.lookup_table.dim",
	"flow_lm.lookup_table.n_bins",
	"flow_lm.lookup_table.tokenizer",
	"flow_lm.lookup_table.tokenizer_path",
	"mimi.dtype",
	"mimi.sample_rate",
	"mimi.channels",
	"mimi.frame_rate",
	"mimi.seanet.dimension",
	"mimi.seanet.channels",
	"mimi.seanet.n_filters",
	"mimi.seanet.n_residual_layers",
	"mimi.seanet.ratios",
	"mimi.seanet.kernel_size",
	"mimi.seanet.residual_kernel_size",
	"mimi.seanet.last_kernel_size",
	"mimi.seanet.dilation_base",
	"mimi.seanet.pad_mode",
	"mimi.seanet.compress",
	"mimi.transformer.d_model",
	"mimi.transformer.input_dimension",
	"mimi.transformer.output_dimensions",
	"mimi.transformer.num_heads",
	"mimi.transformer.num_layers",
	"mimi.transformer.layer_scale",
	"mimi.transformer.context",
	"mimi.transformer.dim_feedforward",
	"mimi.quantizer.dimension",
	"mimi.quantizer.output_dimension",
}

// defaults returns a config pre-filled with the upstream pydantic defaults.
func defaults() ModelConfig {
	return ModelConfig{
		FlowLM: FlowLMConfig{
			Flow: FlowConfig{Type: FlowTypeLSD},
		},
		Mimi: MimiConfig{
			Transformer: MimiTransformerConfig{MaxPeriod: 10000},
		},
		AppendTerminalPunctuation: true,
		CapitalizeFirstLetter:     true,
		DefaultTemperature:        0.3,
	}
}

// Load reads and parses a model config YAML file.
func Load(path string) (*ModelConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("modelcfg: read %s: %w", path, err)
	}

	cfg, err := Parse(data)
	if err != nil {
		return nil, fmt.Errorf("modelcfg: %s: %w", path, err)
	}

	return cfg, nil
}

// Parse decodes a model config YAML document strictly and validates it.
func Parse(data []byte) (*ModelConfig, error) {
	cfg := defaults()

	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)

	err := dec.Decode(&cfg)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}

	// Like upstream's yaml.safe_load, reject trailing documents instead of
	// silently ignoring them.
	var trailing yaml.Node

	err = dec.Decode(&trailing)
	if !errors.Is(err, io.EOF) {
		if err != nil {
			return nil, fmt.Errorf("decode: %w", err)
		}

		return nil, errors.New("decode: expected a single YAML document")
	}

	var raw map[string]any

	err = yaml.Unmarshal(data, &raw)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}

	var missing []string

	for _, key := range requiredKeys {
		if !hasKey(raw, key) {
			missing = append(missing, key)
		}
	}

	if len(missing) > 0 {
		return nil, fmt.Errorf("missing required fields: %s", strings.Join(missing, ", "))
	}

	err = cfg.Validate()
	if err != nil {
		return nil, err
	}

	return &cfg, nil
}

// Validate checks the enumerated fields and n_bins.
func (c *ModelConfig) Validate() error {
	var errs []error

	// n_bins is the tokenizer vocab size (checked at tokenizer load), so it
	// must be positive.
	if c.FlowLM.LookupTable.NBins < 1 {
		errs = append(errs, fmt.Errorf("flow_lm.lookup_table.n_bins %d: want a positive vocab size",
			c.FlowLM.LookupTable.NBins))
	}

	if !slices.Contains([]string{FlowTypeLSD, FlowTypeFlowMatching, FlowTypeDrifting}, c.FlowLM.Flow.Type) {
		errs = append(errs, fmt.Errorf("flow_lm.flow.type %q: want %q, %q or %q",
			c.FlowLM.Flow.Type, FlowTypeLSD, FlowTypeFlowMatching, FlowTypeDrifting))
	}

	if !slices.Contains([]string{TokenizerSentencePiece, TokenizerTokenizers}, c.FlowLM.LookupTable.Tokenizer) {
		errs = append(errs, fmt.Errorf("flow_lm.lookup_table.tokenizer %q: want %q or %q",
			c.FlowLM.LookupTable.Tokenizer, TokenizerSentencePiece, TokenizerTokenizers))
	}

	// Upstream str.maketrans only accepts single-character keys.
	for _, from := range slices.Sorted(maps.Keys(c.ReplaceCharacters)) {
		if utf8.RuneCountInString(from) != 1 {
			errs = append(errs, fmt.Errorf("replace_characters key %q: want exactly one character", from))
		}
	}

	return errors.Join(errs...)
}

// NumTimeConds returns the number of time conditions the sampler head takes
// (upstream: SimpleMLPAdaLN.from_pydantic_config). Unknown types return 0;
// Validate rejects them.
func (c *ModelConfig) NumTimeConds() int {
	switch c.FlowLM.Flow.Type {
	case FlowTypeLSD:
		return 2
	case FlowTypeFlowMatching:
		return 1
	default:
		return 0
	}
}

// MimiInnerDim returns the latent dimension between the Mimi quantizer and
// the FlowLM (upstream: config.mimi.inner_dim or config.mimi.seanet.dimension).
func (c *ModelConfig) MimiInnerDim() int {
	if c.Mimi.InnerDim != nil && *c.Mimi.InnerDim != 0 {
		return *c.Mimi.InnerDim
	}

	return c.Mimi.SEANet.Dimension
}

// FramesAfterEOS returns model_recommended_frames_after_eos, or chunkGuess
// (text.ChunkMetadata.FramesAfterEOS) when the config has none or c is nil
// (upstream generate_audio_stream).
func (c *ModelConfig) FramesAfterEOS(chunkGuess int) int {
	if c == nil || c.ModelRecommendedFramesAfterEOS == nil {
		return chunkGuess
	}

	return *c.ModelRecommendedFramesAfterEOS
}

func hasKey(node map[string]any, dotted string) bool {
	head, rest, nested := strings.Cut(dotted, ".")

	value, ok := node[head]
	if !ok || value == nil {
		return false
	}

	if !nested {
		return true
	}

	child, ok := value.(map[string]any)
	if !ok {
		return false
	}

	return hasKey(child, rest)
}
