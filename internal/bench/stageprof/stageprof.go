package stageprof

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/audio"
	"github.com/cwbudde/go-pocket-tts/internal/config"
	nativemodel "github.com/cwbudde/go-pocket-tts/internal/native"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/ops"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	textpkg "github.com/cwbudde/go-pocket-tts/internal/text"
	"github.com/cwbudde/go-pocket-tts/internal/tokenizer"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
)

const maxTokensPerChunk = 50

type timings struct {
	prepare  time.Duration
	generate time.Duration
	encode   time.Duration
	total    time.Duration
	samples  int
	chunks   int
}

type options struct {
	input          string
	runs           int
	warmup         int
	cpuprofile     string
	runtimeWorkers int
	convWorkers    int
	debugLogs      bool
}

func parseFlags() options {
	var opts options

	flag.StringVar(&opts.input, "text", "Hello from PocketTTS in the browser.", "input text")
	flag.IntVar(&opts.runs, "runs", 5, "number of profiled runs")
	flag.IntVar(&opts.warmup, "warmup", 1, "number of warmup runs")
	flag.StringVar(&opts.cpuprofile, "cpuprofile", "", "write cpu profile")
	flag.IntVar(&opts.runtimeWorkers, "runtime-workers", 0, "tensor workers (0 = fallback to conv-workers)")
	flag.IntVar(&opts.convWorkers, "conv-workers", 2, "conv workers")
	flag.BoolVar(&opts.debugLogs, "debug-logs", false, "enable debug logs from generation stages")
	flag.Parse()

	return opts
}

// effectiveWorkers resolves the tensor worker count, falling back to the conv
// worker count and finally to 1.
func effectiveWorkers(cfg config.Config) int {
	tw := cfg.Runtime.Workers
	if tw <= 0 {
		tw = cfg.Runtime.ConvWorkers
	}

	if tw <= 0 {
		tw = 1
	}

	return tw
}

func Main() {
	opts := parseFlags()

	if opts.debugLogs {
		slog.SetDefault(
			slog.New(
				slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}),
			),
		)
	}

	if opts.runs < 1 {
		fatalf("--runs must be >= 1")
	}

	cfg := config.DefaultConfig()
	cfg.TTS.Backend = config.BackendNative
	cfg.Runtime.Workers = opts.runtimeWorkers
	cfg.Runtime.ConvWorkers = opts.convWorkers

	ops.SetConvWorkers(cfg.Runtime.ConvWorkers)

	tw := effectiveWorkers(cfg)
	tensor.SetWorkers(tw)

	tok, err := tokenizer.NewSentencePieceTokenizer(cfg.Paths.TokenizerModel)
	if err != nil {
		fatalf("init tokenizer: %v", err)
	}

	model, err := nativemodel.LoadModelFromSafetensors(cfg.Paths.ModelPath, nativemodel.DefaultConfig())
	if err != nil {
		fatalf("load model: %v", err)
	}
	defer model.Close()

	rt := tts.NewNativeSafetensorsRuntime(model)
	defer rt.Close()

	ctx := context.Background()

	for i := range opts.warmup {
		_, err := runOnce(ctx, rt, tok, cfg, opts.input)
		if err != nil {
			fatalf("warmup run %d failed: %v", i+1, err)
		}
	}

	if opts.cpuprofile != "" {
		f, err := os.Create(opts.cpuprofile)
		if err != nil {
			fatalf("create cpuprofile: %v", err)
		}
		defer f.Close()

		err = pprof.StartCPUProfile(f)
		if err != nil {
			fatalf("start cpuprofile: %v", err)
		}

		defer pprof.StopCPUProfile()
	}

	agg := profileRuns(ctx, rt, tok, cfg, opts)

	_, err = os.Stdout.WriteString(formatReport(opts, cfg, tw, agg))
	if err != nil {
		fatalf("write report: %v", err)
	}
}

// profileRuns executes the profiled runs and aggregates their timings.
func profileRuns(ctx context.Context, rt tts.Runtime, tok textpkg.Tokenizer, cfg config.Config, opts options) timings {
	var agg timings

	for i := range opts.runs {
		t, err := runOnce(ctx, rt, tok, cfg, opts.input)
		if err != nil {
			fatalf("profiled run %d failed: %v", i+1, err)
		}

		agg.prepare += t.prepare
		agg.generate += t.generate
		agg.encode += t.encode
		agg.total += t.total
		agg.samples = t.samples
		agg.chunks = t.chunks
	}

	return agg
}

// formatReport renders the aggregated timings as key: value lines.
func formatReport(opts options, cfg config.Config, tw int, agg timings) string {
	div := float64(opts.runs)
	avgPrepare := agg.prepare.Seconds() * 1000 / div
	avgGenerate := agg.generate.Seconds() * 1000 / div
	avgEncode := agg.encode.Seconds() * 1000 / div
	avgTotal := agg.total.Seconds() * 1000 / div

	audioMS := float64(agg.samples) * 1000.0 / float64(audio.ExpectedSampleRate)
	rtf := avgTotal / audioMS

	var sb strings.Builder

	fmt.Fprintf(&sb, "text: %q\n", opts.input)
	fmt.Fprintf(&sb, "runs: %d (warmup %d)\n", opts.runs, opts.warmup)
	fmt.Fprintf(&sb, "runtime_workers_effective: %d\n", tw)
	fmt.Fprintf(&sb, "conv_workers: %d\n", cfg.Runtime.ConvWorkers)
	fmt.Fprintf(&sb, "chunks: %d\n", agg.chunks)
	fmt.Fprintf(&sb, "audio_ms: %.2f\n", audioMS)
	fmt.Fprintf(&sb, "avg_prepare_ms: %.2f\n", avgPrepare)
	fmt.Fprintf(&sb, "avg_generate_ms: %.2f\n", avgGenerate)
	fmt.Fprintf(&sb, "avg_encode_ms: %.2f\n", avgEncode)
	fmt.Fprintf(&sb, "avg_total_ms: %.2f\n", avgTotal)
	fmt.Fprintf(&sb, "rtf: %.3f\n", rtf)

	if avgTotal > 0 {
		fmt.Fprintf(&sb, "share_prepare_pct: %.2f\n", 100*avgPrepare/avgTotal)
		fmt.Fprintf(&sb, "share_generate_pct: %.2f\n", 100*avgGenerate/avgTotal)
		fmt.Fprintf(&sb, "share_encode_pct: %.2f\n", 100*avgEncode/avgTotal)
	}

	return sb.String()
}

func runOnce(ctx context.Context, rt tts.Runtime, tok textpkg.Tokenizer, cfg config.Config, input string) (timings, error) {
	var out timings
	startTotal := time.Now()

	var chunks []textpkg.ChunkMetadata
	var prepErr error

	pprof.Do(ctx, pprof.Labels("stage", "prepare"), func(context.Context) {
		start := time.Now()
		chunks, prepErr = textpkg.PrepareChunks(input, tok, maxTokensPerChunk)
		out.prepare = time.Since(start)
	})

	if prepErr != nil {
		return out, fmt.Errorf("prepare chunks: %w", prepErr)
	}

	if len(chunks) == 0 {
		return out, errors.New("no chunks produced")
	}

	var allAudio []float32
	var genErr error

	pprof.Do(ctx, pprof.Labels("stage", "generate"), func(ctx context.Context) {
		start := time.Now()

		for _, chunk := range chunks {
			estimatedMaxSteps := textpkg.EstimateMaxFrames(chunk.NumTokens, textpkg.DefaultMimiFrameRate)
			mimiStepsPerLatent := 16
			gcfg := tts.RuntimeGenerateConfig{
				Temperature:        cfg.TTS.Temperature,
				EOSThreshold:       cfg.TTS.EOSThreshold,
				MaxSteps:           cfg.TTS.MaxSteps,
				EstimatedMaxSteps:  estimatedMaxSteps,
				LSDDecodeSteps:     cfg.TTS.LSDDecodeSteps,
				FramesAfterEOS:     chunk.FramesAfterEOS(),
				MimiStepsPerLatent: mimiStepsPerLatent,
				MimiSequenceLength: estimatedMaxSteps * mimiStepsPerLatent,
			}

			pcm, err := rt.GenerateAudio(ctx, chunk.TokenIDs, gcfg)
			if err != nil {
				genErr = err
				return
			}

			allAudio = append(allAudio, pcm...)
		}

		out.generate = time.Since(start)
	})

	if genErr != nil {
		return out, fmt.Errorf("generate audio: %w", genErr)
	}

	var encErr error

	pprof.Do(ctx, pprof.Labels("stage", "encode"), func(context.Context) {
		start := time.Now()
		_, encErr = audio.EncodeWAV(allAudio)
		out.encode = time.Since(start)
	})

	if encErr != nil {
		return out, fmt.Errorf("encode wav: %w", encErr)
	}

	out.total = time.Since(startTotal)
	out.samples = len(allAudio)
	out.chunks = len(chunks)

	return out, nil
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
