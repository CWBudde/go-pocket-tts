package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/example/go-pocket-tts/internal/audio"
	"github.com/example/go-pocket-tts/internal/config"
	textpkg "github.com/example/go-pocket-tts/internal/text"
	"github.com/example/go-pocket-tts/internal/tts"
	"github.com/spf13/cobra"
)

func newSynthCmd() *cobra.Command {
	var text string
	var out string
	var voice string
	var ttsArgs []string
	var backend string
	var chunk bool
	var maxChunkChars int
	var normalize bool
	var dcBlock bool
	var fadeInMS float64
	var fadeOutMS float64

	cmd := &cobra.Command{
		Use:   "synth",
		Short: "Synthesize text to WAV",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			selectedBackend, err := resolveSynthBackend(backend, cfg.TTS.Backend)
			if err != nil {
				return err
			}
			inputText, err := readSynthText(text, os.Stdin)
			if err != nil {
				return err
			}

			selectedVoice := cfg.TTS.Voice
			if voice != "" {
				selectedVoice = voice
			}

			chunks, err := buildSynthesisChunks(inputText, chunk, maxChunkChars)
			if err != nil {
				return err
			}

			var result []byte
			switch selectedBackend {
			case config.BackendNative, config.BackendNativeSafetensors:
				if len(ttsArgs) > 0 {
					return fmt.Errorf("--tts-arg is only supported with --backend cli")
				}
				var resolvedVoice string
				resolvedVoice, err = resolveVoiceForNative(selectedVoice)
				if err != nil {
					return err
				}
				nativeCfg := cfg
				nativeCfg.TTS.Backend = selectedBackend
				result, err = synthesizeNative(cmd.Context(), nativeCfg, chunks, resolvedVoice)
			case config.BackendCLI:
				var resolvedVoice string
				resolvedVoice, err = resolveVoiceOrPath(selectedVoice)
				if err != nil {
					return err
				}
				result, err = synthesizeChunks(cmd.Context(), synthChunksOptions{
					CLI: synthCLIOptions{
						ExecutablePath: cfg.TTS.CLIPath,
						ConfigPath:     cfg.TTS.CLIConfigPath,
						Voice:          resolvedVoice,
						Quiet:          cfg.TTS.Quiet,
						ExtraArgs:      ttsArgs,
						Stderr:         os.Stderr,
					},
					Chunks:    chunks,
					ChunkMode: chunk,
				})
			default:
				return fmt.Errorf("unsupported backend %q", selectedBackend)
			}
			if err != nil {
				return mapSynthError(err)
			}

			if normalize || dcBlock || fadeInMS > 0 || fadeOutMS > 0 {
				processed, err := applyDSPToWAV(result, synthDSPOptions{
					Normalize: normalize,
					DCBlock:   dcBlock,
					FadeInMS:  fadeInMS,
					FadeOutMS: fadeOutMS,
				})
				if err != nil {
					return err
				}
				result = processed
			}

			return writeSynthOutput(out, result, os.Stdout)
		},
	}

	cmd.Flags().StringVar(&text, "text", "", "Text to synthesize (if empty, read from stdin)")
	cmd.Flags().StringVar(&out, "out", "out.wav", "Output WAV path ('-' for stdout)")
	cmd.Flags().StringVar(
		&backend,
		"backend",
		"",
		"Synthesis backend override (native-onnx|native-safetensors|cli; native is alias for native-onnx)",
	)
	cmd.Flags().StringVar(&voice, "voice", "", "Voice ID from voices/manifest.json (overrides config)")
	cmd.Flags().BoolVar(&chunk, "chunk", false, "Split text into sentence chunks and synthesize sequentially")
	cmd.Flags().IntVar(&maxChunkChars, "max-chunk-chars", 220, "Maximum characters per chunk when --chunk is enabled")
	cmd.Flags().BoolVar(&normalize, "normalize", false, "Peak-normalize output audio")
	cmd.Flags().BoolVar(&dcBlock, "dc-block", false, "Apply DC-block high-pass filter")
	cmd.Flags().Float64Var(&fadeInMS, "fade-in-ms", 0, "Apply linear fade-in duration in milliseconds")
	cmd.Flags().Float64Var(&fadeOutMS, "fade-out-ms", 0, "Apply linear fade-out duration in milliseconds")
	cmd.Flags().StringArrayVar(&ttsArgs, "tts-arg", nil, "Pass-through pocket-tts flag in key=value form (repeatable)")

	return cmd
}

type synthCLIOptions struct {
	ExecutablePath string
	ConfigPath     string
	Voice          string
	Quiet          bool
	Text           string
	ExtraArgs      []string
	Stderr         io.Writer
}

type synthChunksOptions struct {
	CLI       synthCLIOptions
	Chunks    []string
	ChunkMode bool
}

type synthDSPOptions struct {
	Normalize bool
	DCBlock   bool
	FadeInMS  float64
	FadeOutMS float64
}

var runChunkSynthesis = synthesizeViaCLI

func synthesizeViaCLI(ctx context.Context, opts synthCLIOptions) ([]byte, error) {
	exe := opts.ExecutablePath
	if exe == "" {
		exe = "pocket-tts"
	}
	if strings.TrimSpace(opts.Text) == "" {
		return nil, fmt.Errorf("synth failed: empty input text")
	}

	args := []string{"generate", "--text", "-", "--output-path", "-"}
	if opts.Voice != "" {
		args = append(args, "--voice", opts.Voice)
	}
	if opts.ConfigPath != "" {
		args = append(args, "--config", opts.ConfigPath)
	}
	if opts.Quiet {
		args = append(args, "--quiet")
	}

	extra, err := buildPassthroughArgs(opts.ExtraArgs)
	if err != nil {
		return nil, err
	}
	args = append(args, extra...)

	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Stdin = strings.NewReader(opts.Text)
	if opts.Stderr != nil {
		cmd.Stderr = opts.Stderr
	}

	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

func buildSynthesisChunks(input string, chunk bool, maxChunkChars int) ([]string, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return nil, fmt.Errorf("empty input text")
	}
	if !chunk {
		return []string{input}, nil
	}

	chunks := textpkg.ChunkBySentence(input, maxChunkChars)
	out := make([]string, 0, len(chunks))
	for _, c := range chunks {
		c = strings.TrimSpace(c)
		if c != "" {
			out = append(out, c)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no non-empty chunks produced from input")
	}
	return out, nil
}

func synthesizeChunks(ctx context.Context, opts synthChunksOptions) ([]byte, error) {
	results := make([][]byte, 0, len(opts.Chunks))
	for i, chunkText := range opts.Chunks {
		chunkOpts := opts.CLI
		chunkOpts.Text = chunkText
		wavBytes, err := runChunkSynthesis(ctx, chunkOpts)
		if err != nil {
			return nil, fmt.Errorf("chunk %d synthesis failed: %w", i+1, err)
		}
		results = append(results, wavBytes)
	}

	if !opts.ChunkMode || len(results) == 1 {
		return results[0], nil
	}
	return concatenateWAVChunks(results)
}

func synthesizeNative(ctx context.Context, cfg config.Config, chunks []string, voicePath string) ([]byte, error) {
	svc, err := tts.NewService(cfg)
	if err != nil {
		return nil, fmt.Errorf("initialize native synth service: %w", err)
	}
	defer svc.Close()

	merged := make([]float32, 0, 24000)
	for i, chunkText := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		samples, err := svc.Synthesize(chunkText, voicePath)
		if err != nil {
			return nil, fmt.Errorf("native chunk %d synthesis failed: %w", i+1, err)
		}
		merged = append(merged, samples...)
	}
	if len(merged) == 0 {
		return nil, fmt.Errorf("native synthesis produced no samples")
	}

	wavData, err := audio.EncodeWAV(merged)
	if err != nil {
		return nil, fmt.Errorf("encode native synthesis WAV: %w", err)
	}
	return wavData, nil
}

func concatenateWAVChunks(chunkWAVs [][]byte) ([]byte, error) {
	merged := make([]float32, 0, 24000)
	for i, data := range chunkWAVs {
		samples, err := audio.DecodeWAV(data)
		if err != nil {
			return nil, fmt.Errorf("decode chunk %d WAV: %w", i+1, err)
		}
		merged = append(merged, samples...)
	}
	out, err := audio.EncodeWAV(merged)
	if err != nil {
		return nil, fmt.Errorf("encode merged WAV: %w", err)
	}
	return out, nil
}

func applyDSPToWAV(wavData []byte, opts synthDSPOptions) ([]byte, error) {
	samples, err := audio.DecodeWAV(wavData)
	if err != nil {
		return nil, fmt.Errorf("decode WAV for DSP: %w", err)
	}

	processed := samples
	if opts.Normalize {
		processed = audio.PeakNormalize(processed)
	}
	if opts.DCBlock {
		processed = audio.DCBlock(processed, audio.ExpectedSampleRate)
	}
	if opts.FadeInMS > 0 {
		processed = audio.FadeIn(processed, audio.ExpectedSampleRate, opts.FadeInMS)
	}
	if opts.FadeOutMS > 0 {
		processed = audio.FadeOut(processed, audio.ExpectedSampleRate, opts.FadeOutMS)
	}

	out, err := audio.EncodeWAV(processed)
	if err != nil {
		return nil, fmt.Errorf("encode WAV after DSP: %w", err)
	}
	return out, nil
}

func writeSynthOutput(outPath string, wavData []byte, stdout io.Writer) error {
	if outPath == "-" {
		if stdout == nil {
			return fmt.Errorf("stdout writer is nil")
		}
		_, err := stdout.Write(wavData)
		return err
	}
	return os.WriteFile(outPath, wavData, 0o644)
}

func readSynthText(text string, stdin io.Reader) (string, error) {
	if strings.TrimSpace(text) != "" {
		return text, nil
	}

	b, err := io.ReadAll(stdin)
	if err != nil {
		return "", fmt.Errorf("read stdin: %w", err)
	}
	input := strings.TrimSpace(string(b))
	if input == "" {
		return "", fmt.Errorf("either provide --text or pipe text on stdin")
	}
	return input, nil
}

func resolveSynthBackend(flagBackend, cfgBackend string) (string, error) {
	backend := strings.TrimSpace(flagBackend)
	if backend == "" {
		backend = strings.TrimSpace(cfgBackend)
	}
	return config.NormalizeBackend(backend)
}

// resolveVoiceForNative resolves a voice identifier to an absolute .safetensors
// path for the native backend. Unlike resolveVoiceOrPath (which falls back to
// returning the raw voice string for the CLI), an unresolved ID here means no
// voice file — we return an empty string so Synthesize skips voice conditioning.
func resolveVoiceForNative(voice string) (string, error) {
	if strings.TrimSpace(voice) == "" {
		return "", nil
	}

	// If it looks like a file path (contains a slash or ends in .safetensors),
	// treat it as a direct path.
	if strings.Contains(voice, string(filepath.Separator)) || strings.HasSuffix(voice, ".safetensors") {
		return voice, nil
	}

	// Resolve voice ID via the manifest.
	vm, err := tts.NewVoiceManager(filepath.Join("voices", "manifest.json"))
	if err != nil {
		// Manifest missing or unreadable — skip voice conditioning.
		return "", nil
	}
	path, err := vm.ResolvePath(voice)
	if err != nil {
		if strings.Contains(err.Error(), "unknown voice id") {
			// Not in manifest — skip voice conditioning rather than error.
			return "", nil
		}
		return "", fmt.Errorf("resolve --voice %q: %w", voice, err)
	}
	return path, nil
}

func resolveVoiceOrPath(voice string) (string, error) {
	if strings.TrimSpace(voice) == "" {
		return "", nil
	}

	vm, err := tts.NewVoiceManager(filepath.Join("voices", "manifest.json"))
	if err != nil {
		// Manifest is optional for integration and built-in voices; fall back.
		return voice, nil
	}
	path, err := vm.ResolvePath(voice)
	if err != nil {
		// If voice is not declared in manifest, treat it as a raw CLI voice value.
		if strings.Contains(err.Error(), "unknown voice id") {
			return voice, nil
		}
		return "", fmt.Errorf("resolve --voice %q: %w", voice, err)
	}
	return path, nil
}

func buildPassthroughArgs(items []string) ([]string, error) {
	args := make([]string, 0, len(items))
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		parts := strings.SplitN(item, "=", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid --tts-arg %q: expected key=value", item)
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		if key == "" {
			return nil, fmt.Errorf("invalid --tts-arg %q: empty key", item)
		}
		if strings.HasPrefix(key, "--") {
			args = append(args, key+"="+val)
		} else if strings.HasPrefix(key, "-") {
			args = append(args, "-"+strings.TrimPrefix(key, "-")+"="+val)
		} else {
			args = append(args, "--"+key+"="+val)
		}
	}
	return args, nil
}

func mapSynthError(err error) error {
	if errors.Is(err, exec.ErrNotFound) {
		return fmt.Errorf("synth failed: pocket-tts executable not found; set --tts-cli-path or POCKETTTS_TTS_CLI_PATH: %w", err)
	}

	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return fmt.Errorf("synth failed: pocket-tts returned non-zero exit; check stderr details above: %w", err)
	}

	return err
}
