package native

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

func requireCheckpointForBench(b *testing.B) string {
	b.Helper()
	candidates := []string{
		filepath.Join("models", "tts_b6369a24.safetensors"),
		filepath.Join("..", "..", "models", "tts_b6369a24.safetensors"),
	}
	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	b.Skipf("checkpoint not available in any expected location: %v", candidates)
	return ""
}

func nanFrame() (*tensor.Tensor, error) {
	data := make([]float32, 32)
	for i := range data {
		data[i] = float32(math.NaN())
	}
	return tensor.New(data, []int64{1, 1, 32})
}

// BenchmarkSynthesisFullPipeline runs a representative synthesis loop using the
// real checkpoint: text embedding → prompt → N stateful steps → Mimi decode.
// Run with -cpuprofile or -memprofile to locate hot paths.
func BenchmarkSynthesisFullPipeline(b *testing.B) {
	path := requireCheckpointForBench(b)
	m, err := LoadModelFromSafetensors(path, DefaultConfig())
	if err != nil {
		b.Fatalf("load model: %v", err)
	}
	defer m.Close()

	tokenIDs := []int64{10, 20, 30, 40, 50}
	textEmb, err := m.TextEmbeddings(tokenIDs)
	if err != nil {
		b.Fatalf("text embeddings: %v", err)
	}

	const maxSteps = 20
	const decodeSteps = 10
	const eosThreshold = float32(0.5)
	const temperature = float32(1.0)
	rng := rand.New(rand.NewSource(42))

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		state, err := m.NewFlowState()
		if err != nil {
			b.Fatalf("new flow state: %v", err)
		}
		if err := m.PromptFlow(state, textEmb); err != nil {
			b.Fatalf("prompt flow: %v", err)
		}

		frame, err := nanFrame()
		if err != nil {
			b.Fatalf("nan frame: %v", err)
		}

		var allLatents []*tensor.Tensor
		for step := 0; step < maxSteps; step++ {
			next, isEOS, err := m.SampleNextLatentStateful(state, frame, decodeSteps, eosThreshold, temperature, rng)
			if err != nil {
				b.Fatalf("step %d: %v", step, err)
			}
			allLatents = append(allLatents, next)
			frame = next
			if isEOS {
				break
			}
		}

		if len(allLatents) > 0 {
			combined, err := tensor.Concat(allLatents, 1)
			if err != nil {
				b.Fatalf("concat latents: %v", err)
			}
			mimiIn, err := m.LatentToMimi(combined)
			if err != nil {
				b.Fatalf("latent to mimi: %v", err)
			}
			_, err = m.MimiDecode(mimiIn)
			if err != nil {
				b.Fatalf("mimi decode: %v", err)
			}
		}
	}
}

// BenchmarkFlowStep benchmarks a single stateful transformer step — the AR
// inner loop. Run with -cpuprofile to see per-layer costs.
func BenchmarkFlowStep(b *testing.B) {
	path := requireCheckpointForBench(b)
	m, err := LoadModelFromSafetensors(path, DefaultConfig())
	if err != nil {
		b.Fatalf("load model: %v", err)
	}
	defer m.Close()

	tokenIDs := []int64{10, 20, 30, 40, 50}
	textEmb, err := m.TextEmbeddings(tokenIDs)
	if err != nil {
		b.Fatalf("text embeddings: %v", err)
	}

	state, err := m.NewFlowState()
	if err != nil {
		b.Fatalf("new flow state: %v", err)
	}
	if err := m.PromptFlow(state, textEmb); err != nil {
		b.Fatalf("prompt flow: %v", err)
	}

	frame, err := nanFrame()
	if err != nil {
		b.Fatalf("nan frame: %v", err)
	}

	rng := rand.New(rand.NewSource(42))

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := m.SampleNextLatentStateful(state, frame, 10, 0.5, 1.0, rng)
		if err != nil {
			b.Fatalf("step: %v", err)
		}
	}
}

// BenchmarkMimiDecode benchmarks Mimi decoding of accumulated latents.
// This is the Conv1D/ConvTranspose1D hot path.
func BenchmarkMimiDecode(b *testing.B) {
	path := requireCheckpointForBench(b)
	m, err := LoadModelFromSafetensors(path, DefaultConfig())
	if err != nil {
		b.Fatalf("load model: %v", err)
	}
	defer m.Close()

	// 20 latent frames (typical short utterance).
	latent, err := tensor.New(make([]float32, 1*20*32), []int64{1, 20, 32})
	if err != nil {
		b.Fatalf("latent: %v", err)
	}

	mimiIn, err := m.LatentToMimi(latent)
	if err != nil {
		b.Fatalf("latent to mimi: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := m.MimiDecode(mimiIn)
		if err != nil {
			b.Fatalf("mimi decode: %v", err)
		}
	}
}
