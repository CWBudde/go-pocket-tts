package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/example/go-pocket-tts/internal/bench"
	"github.com/spf13/cobra"
)

func newBenchCmd() *cobra.Command {
	var (
		text         string
		voice        string
		runs         int
		format       string
		rtfThreshold float64
	)

	cmd := &cobra.Command{
		Use:   "bench",
		Short: "Benchmark synthesis latency and realtime factor",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, err := requireConfig()
			if err != nil {
				return err
			}

			if strings.TrimSpace(text) == "" {
				return fmt.Errorf("--text is required for bench")
			}
			if runs < 1 {
				return fmt.Errorf("--runs must be at least 1")
			}
			if format != "table" && format != "json" {
				return fmt.Errorf("--format must be 'table' or 'json'")
			}

			selectedVoice := cfg.TTS.Voice
			if voice != "" {
				selectedVoice = voice
			}
			resolvedVoice, err := resolveVoiceOrPath(selectedVoice)
			if err != nil {
				return err
			}

			results, err := runBench(cmd.Context(), benchOptions{
				ExecutablePath: cfg.TTS.CLIPath,
				ConfigPath:     cfg.TTS.CLIConfigPath,
				Voice:          resolvedVoice,
				Quiet:          cfg.TTS.Quiet,
				Text:           text,
				Runs:           runs,
			})
			if err != nil {
				return err
			}

			durations := make([]time.Duration, len(results))
			for i, r := range results {
				durations[i] = r.Duration
			}
			stats := bench.ComputeStats(durations)

			switch format {
			case "json":
				bench.FormatJSON(results, stats, os.Stdout)
			default:
				bench.FormatTable(results, stats, os.Stdout)
			}

			// Compute mean RTF across all runs.
			var totalRTF float64
			for _, r := range results {
				totalRTF += r.RTF
			}
			meanRTF := totalRTF / float64(len(results))

			if err := bench.CheckRTFThreshold(meanRTF, rtfThreshold); err != nil {
				return err
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&text, "text", "", "Text to synthesize for each run (required)")
	cmd.Flags().StringVar(&voice, "voice", "", "Voice ID (overrides config)")
	cmd.Flags().IntVar(&runs, "runs", 5, "Number of synthesis runs")
	cmd.Flags().StringVar(&format, "format", "table", "Output format: table|json")
	cmd.Flags().Float64Var(&rtfThreshold, "rtf-threshold", 0, "Exit non-zero if mean RTF exceeds this value (0 = disabled)")

	return cmd
}

type benchOptions struct {
	ExecutablePath string
	ConfigPath     string
	Voice          string
	Quiet          bool
	Text           string
	Runs           int
}

func runBench(ctx context.Context, opts benchOptions) ([]bench.RunResult, error) {
	results := make([]bench.RunResult, 0, opts.Runs)

	for i := range opts.Runs {
		start := time.Now()
		wavBytes, err := synthesizeViaCLI(ctx, synthCLIOptions{
			ExecutablePath: opts.ExecutablePath,
			ConfigPath:     opts.ConfigPath,
			Voice:          opts.Voice,
			Quiet:          opts.Quiet,
			Text:           opts.Text,
		})
		if err != nil {
			return nil, fmt.Errorf("run %d failed: %w", i+1, err)
		}
		dur := time.Since(start)

		audioDur, err := bench.WAVDuration(wavBytes)
		if err != nil {
			// Non-fatal: log and continue with zero audio duration.
			fmt.Fprintf(os.Stderr, "warn: run %d: could not parse WAV duration: %v\n", i+1, err)
		}

		results = append(results, bench.RunResult{
			Index:       i,
			Cold:        i == 0,
			Duration:    dur,
			WAVDuration: audioDur,
			RTF:         bench.CalcRTF(dur, audioDur),
		})
	}

	return results, nil
}
