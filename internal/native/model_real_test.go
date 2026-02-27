package native

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/example/go-pocket-tts/internal/runtime/ops"
	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

func requireCheckpoint(t *testing.T) string {
	t.Helper()

	candidates := []string{
		filepath.Join("models", "tts_b6369a24.safetensors"),
		filepath.Join("..", "..", "models", "tts_b6369a24.safetensors"),
	}
	for _, path := range candidates {
		_, err := os.Stat(path)
		if err == nil {
			return path
		}
	}

	t.Skipf("checkpoint not available in any expected location: %v", candidates)

	return ""
}

func TestLoadModelFromSafetensors_RealCheckpoint(t *testing.T) {
	path := requireCheckpoint(t)

	m, err := LoadModelFromSafetensors(path, DefaultConfig())
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	defer m.Close()

	if m.FlowLM() == nil || m.Mimi() == nil {
		t.Fatalf("loaded model missing flow or mimi modules")
	}
}

func TestFlowMainAndFlowDirection_RealCheckpoint(t *testing.T) {
	path := requireCheckpoint(t)

	m, err := LoadModelFromSafetensors(path, DefaultConfig())
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	defer m.Close()

	textEmb, err := m.TextEmbeddings([]int64{10, 20, 30})
	if err != nil {
		t.Fatalf("text embeddings: %v", err)
	}

	seq, err := tensor.New([]float32{
		float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN()),
		float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN()),
		float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN()),
		float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN()),
		float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN()),
		float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN()),
		float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN()),
		float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN()),
	}, []int64{1, 1, 32})
	if err != nil {
		t.Fatalf("seq: %v", err)
	}

	last, eos, err := m.FlowMain(seq, textEmb)
	if err != nil {
		t.Fatalf("flow main: %v", err)
	}

	if got := last.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 1024 {
		t.Fatalf("last_hidden shape = %v", got)
	}

	if got := eos.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 1 {
		t.Fatalf("eos shape = %v", got)
	}

	s, _ := tensor.New([]float32{0}, []int64{1, 1})
	tv, _ := tensor.New([]float32{1}, []int64{1, 1})
	x, _ := tensor.New(make([]float32, 32), []int64{1, 32})

	dir, err := m.FlowDirection(last, s, tv, x)
	if err != nil {
		t.Fatalf("flow direction: %v", err)
	}

	if got := dir.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 32 {
		t.Fatalf("flow direction shape = %v", got)
	}
}

func TestLatentToMimiAndDecode_RealCheckpoint(t *testing.T) {
	path := requireCheckpoint(t)

	m, err := LoadModelFromSafetensors(path, DefaultConfig())
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	defer m.Close()

	latent, err := tensor.New(make([]float32, 1*2*32), []int64{1, 2, 32})
	if err != nil {
		t.Fatalf("latent: %v", err)
	}

	mimiLatent, err := m.LatentToMimi(latent)
	if err != nil {
		t.Fatalf("latent_to_mimi: %v", err)
	}

	if got := mimiLatent.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 512 || got[2] != 2 {
		t.Fatalf("mimi latent shape = %v", got)
	}

	audio, err := m.MimiDecode(mimiLatent)
	if err != nil {
		t.Fatalf("mimi decode: %v", err)
	}

	if got := audio.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 1 || got[2] <= 0 {
		t.Fatalf("audio shape = %v", got)
	}
}

func TestEncodeVoiceHook_NotImplemented(t *testing.T) {
	path := requireCheckpoint(t)

	m, err := LoadModelFromSafetensors(path, DefaultConfig())
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	defer m.Close()

	audio, _ := tensor.New(make([]float32, 24000), []int64{1, 1, 24000})

	_, err = m.EncodeVoiceHook(audio)
	if err == nil {
		t.Fatal("expected not implemented error")
	}

	if !errors.Is(err, ErrMimiEncoderNotImplemented) {
		t.Fatalf("EncodeVoiceHook err = %v, want %v", err, ErrMimiEncoderNotImplemented)
	}
}

func TestCompareTensor(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3}, []int64{3})
	b, _ := tensor.New([]float32{1, 2.00001, 2.99999}, []int64{3})
	tol, _ := ops.KernelTolerance("linear")

	r, err := CompareTensor("linear", a, b, tol)
	if err != nil {
		t.Fatalf("compare: %v", err)
	}

	if !r.ShapeMatch || !r.Pass {
		t.Fatalf("report = %+v", r)
	}
}
