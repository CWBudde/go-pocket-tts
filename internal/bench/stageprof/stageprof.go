package stageprof

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"runtime/pprof"
	"time"

	"github.com/example/go-pocket-tts/internal/audio"
	"github.com/example/go-pocket-tts/internal/config"
	nativemodel "github.com/example/go-pocket-tts/internal/native"
	"github.com/example/go-pocket-tts/internal/runtime/ops"
	"github.com/example/go-pocket-tts/internal/runtime/tensor"
	textpkg "github.com/example/go-pocket-tts/internal/text"
	"github.com/example/go-pocket-tts/internal/tokenizer"
	"github.com/example/go-pocket-tts/internal/tts"
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

func Main() {
	var (
		input          string
		runs           int
		warmup         int
		cpuprofile     string
		runtimeWorkers int
		convWorkers    int
		debugLogs      bool
	)
	flag.StringVar(&input, "text", "Hello from PocketTTS in the browser.", "input text")
	flag.IntVar(&runs, "runs", 5, "number of profiled runs")
	flag.IntVar(&warmup, "warmup", 1, "number of warmup runs")
	flag.StringVar(&cpuprofile, "cpuprofile", "", "write cpu profile")
	flag.IntVar(&runtimeWorkers, "runtime-workers", 0, "tensor workers (0 = fallback to conv-workers)")
	flag.IntVar(&convWorkers, "conv-workers", 2, "conv workers")
	flag.BoolVar(&debugLogs, "debug-logs", false, "enable debug logs from generation stages")
	flag.Parse()

	if debugLogs {
		slog.SetDefault(
			slog.New(
				slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}),
			),
		)
	}

	if runs < 1 {
		fatalf("--runs must be >= 1")
	}

	cfg := config.DefaultConfig()
	cfg.TTS.Backend = config.BackendNative
	cfg.Runtime.Workers = runtimeWorkers
	cfg.Runtime.ConvWorkers = convWorkers

	ops.SetConvWorkers(cfg.Runtime.ConvWorkers)

	tw := cfg.Runtime.Workers
	if tw <= 0 {
		tw = cfg.Runtime.ConvWorkers
	}

	if tw <= 0 {
		tw = 1
	}

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

	for i := range warmup {
		_, err := runOnce(ctx, rt, tok, cfg, input)
		if err != nil {
			fatalf("warmup run %d failed: %v", i+1, err)
		}
	}

	if cpuprofile != "" {
		f, err := os.Create(cpuprofile)
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

	var agg timings

	for i := range runs {
		t, err := runOnce(ctx, rt, tok, cfg, input)
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

	div := float64(runs)
	avgPrepare := agg.prepare.Seconds() * 1000 / div
	avgGenerate := agg.generate.Seconds() * 1000 / div
	avgEncode := agg.encode.Seconds() * 1000 / div
	avgTotal := agg.total.Seconds() * 1000 / div

	audioMS := float64(agg.samples) * 1000.0 / float64(audio.ExpectedSampleRate)
	rtf := avgTotal / audioMS

	fmt.Printf("text: %q\n", input)
	fmt.Printf("runs: %d (warmup %d)\n", runs, warmup)
	fmt.Printf("runtime_workers_effective: %d\n", tw)
	fmt.Printf("conv_workers: %d\n", cfg.Runtime.ConvWorkers)
	fmt.Printf("chunks: %d\n", agg.chunks)
	fmt.Printf("audio_ms: %.2f\n", audioMS)
	fmt.Printf("avg_prepare_ms: %.2f\n", avgPrepare)
	fmt.Printf("avg_generate_ms: %.2f\n", avgGenerate)
	fmt.Printf("avg_encode_ms: %.2f\n", avgEncode)
	fmt.Printf("avg_total_ms: %.2f\n", avgTotal)
	fmt.Printf("rtf: %.3f\n", rtf)

	if avgTotal > 0 {
		fmt.Printf("share_prepare_pct: %.2f\n", 100*avgPrepare/avgTotal)
		fmt.Printf("share_generate_pct: %.2f\n", 100*avgGenerate/avgTotal)
		fmt.Printf("share_encode_pct: %.2f\n", 100*avgEncode/avgTotal)
	}
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
			gcfg := tts.RuntimeGenerateConfig{
				Temperature:    cfg.TTS.Temperature,
				EOSThreshold:   cfg.TTS.EOSThreshold,
				MaxSteps:       cfg.TTS.MaxSteps,
				LSDDecodeSteps: cfg.TTS.LSDDecodeSteps,
				FramesAfterEOS: chunk.FramesAfterEOS(),
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
