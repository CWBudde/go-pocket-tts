//go:build js && wasm

package native

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/runtime/ops"
	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

// BenchmarkWASMDecode isolates the native decode path on js/wasm:
// - mimi_only: MimiDecode with precomputed [B,512,T] input
// - decode_stage: LatentToMimi + MimiDecode (matches generation "decode" stage)
func BenchmarkWASMDecode(b *testing.B) {
	path := requireCheckpointForBench(b)

	m, err := LoadModelFromSafetensors(path, DefaultConfig())
	if err != nil {
		b.Fatalf("load model: %v", err)
	}
	defer m.Close()

	frameCounts := decodeBenchFrameCounts(b)
	workerCounts := []int{1, 2}

	for _, workers := range workerCounts {
		workers := workers

		b.Run(fmt.Sprintf("workers=%d", workers), func(b *testing.B) {
			prevTensorWorkers := tensor.Workers()
			tensor.SetWorkers(workers)
			ops.SetConvWorkers(workers)

			b.Cleanup(func() {
				tensor.SetWorkers(prevTensorWorkers)
				ops.SetConvWorkers(1)
			})

			for _, frames := range frameCounts {
				frames := frames
				latent := decodeBenchLatent(b, frames)

				mimiIn, err := m.LatentToMimi(latent)
				if err != nil {
					b.Fatalf("latent to mimi (frames=%d): %v", frames, err)
				}

				baseOut, err := m.MimiDecode(mimiIn)
				if err != nil {
					b.Fatalf("mimi decode base (frames=%d): %v", frames, err)
				}

				outBytes := int64(len(baseOut.RawData()) * 4)
				if outBytes <= 0 {
					b.Fatalf("unexpected empty decode output for frames=%d", frames)
				}

				b.Run(fmt.Sprintf("frames=%d/mimi_only", frames), func(b *testing.B) {
					for range 3 {
						_, err := m.MimiDecode(mimiIn)
						if err != nil {
							b.Fatalf("warmup mimi decode: %v", err)
						}
					}

					runtime.GC()

					b.SetBytes(outBytes)
					b.ReportAllocs()
					b.ResetTimer()

					for range b.N {
						_, err := m.MimiDecode(mimiIn)
						if err != nil {
							b.Fatalf("mimi decode: %v", err)
						}
					}
				})

				b.Run(fmt.Sprintf("frames=%d/decode_stage", frames), func(b *testing.B) {
					for range 3 {
						warmMimiIn, err := m.LatentToMimi(latent)
						if err != nil {
							b.Fatalf("warmup latent to mimi: %v", err)
						}

						_, err = m.MimiDecode(warmMimiIn)
						if err != nil {
							b.Fatalf("warmup mimi decode: %v", err)
						}
					}

					runtime.GC()

					b.SetBytes(outBytes)
					b.ReportAllocs()
					b.ResetTimer()

					for range b.N {
						mimiLatent, err := m.LatentToMimi(latent)
						if err != nil {
							b.Fatalf("latent to mimi: %v", err)
						}

						_, err = m.MimiDecode(mimiLatent)
						if err != nil {
							b.Fatalf("mimi decode: %v", err)
						}
					}
				})
			}
		})
	}
}

func decodeBenchLatent(tb testing.TB, frames int) *tensor.Tensor {
	tb.Helper()

	if frames <= 0 {
		tb.Fatalf("frames must be > 0")
	}

	data := make([]float32, frames*32)
	for i := range data {
		s := float32(math.Sin(float64(i) * 0.13))
		r := float32((i%11)-5) * 0.05
		data[i] = 0.25*s + r
	}

	latent, err := tensor.New(data, []int64{1, int64(frames), 32})
	if err != nil {
		tb.Fatalf("latent tensor: %v", err)
	}

	return latent
}

func decodeBenchFrameCounts(tb testing.TB) []int {
	tb.Helper()

	raw := strings.TrimSpace(os.Getenv("POCKETTTS_BENCH_DECODE_FRAMES"))
	if raw == "" {
		// Default matches short prompts (e.g. "Hello from PocketTTS in the browser.").
		return []int{4}
	}

	parts := strings.Split(raw, ",")
	out := make([]int, 0, len(parts))
	seen := map[int]struct{}{}

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		n, err := strconv.Atoi(part)
		if err != nil {
			tb.Fatalf("invalid POCKETTTS_BENCH_DECODE_FRAMES entry %q: %v", part, err)
		}

		if n <= 0 {
			tb.Fatalf("invalid POCKETTTS_BENCH_DECODE_FRAMES entry %q: must be > 0", part)
		}

		if _, ok := seen[n]; ok {
			continue
		}

		seen[n] = struct{}{}
		out = append(out, n)
	}

	if len(out) == 0 {
		tb.Fatalf("POCKETTTS_BENCH_DECODE_FRAMES parsed to empty set")
	}

	slices.Sort(out)

	return out
}
