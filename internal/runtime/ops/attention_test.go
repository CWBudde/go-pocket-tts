package ops

import (
	"math"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

func TestCausalMask(t *testing.T) {
	s := mustTensorT(t, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, []int64{1, 3, 3})

	out, err := CausalMask(s, 0)
	if err != nil {
		t.Fatalf("causal mask: %v", err)
	}

	got := out.Data()
	if !math.IsInf(float64(got[1]), -1) || !math.IsInf(float64(got[2]), -1) || !math.IsInf(float64(got[5]), -1) {
		t.Fatalf("causal mask did not set upper triangle to -Inf: %v", got)
	}

	if got[0] != 1 || got[3] != 4 || got[4] != 5 || got[6] != 7 || got[7] != 8 || got[8] != 9 {
		t.Fatalf("causal mask changed non-masked values: %v", got)
	}
}

func TestAttentionCausal(t *testing.T) {
	q := mustTensorT(t, []float32{1, 1}, []int64{1, 1, 2, 1})
	k := mustTensorT(t, []float32{0, 10}, []int64{1, 1, 2, 1})
	v := mustTensorT(t, []float32{1, 5}, []int64{1, 1, 2, 1})

	out, err := Attention(q, k, v, true, 0)
	if err != nil {
		t.Fatalf("attention: %v", err)
	}

	got := out.Data()
	if math.Abs(float64(got[0]-1)) > 1e-4 {
		t.Fatalf("query0 output = %f, want near 1.0 (future token masked)", got[0])
	}

	if got[1] < 4.0 {
		t.Fatalf("query1 output = %f, want > 4.0", got[1])
	}
}

func TestCausalMaskErrors(t *testing.T) {
	_, err := CausalMask(nil, 0)
	if err == nil || !strings.Contains(err.Error(), "is nil") {
		t.Fatalf("CausalMask(nil) err = %v, want nil input error", err)
	}

	sRank1 := mustTensorT(t, []float32{1, 2, 3}, []int64{3})
	_, err = CausalMask(sRank1, 0)

	if err == nil || !strings.Contains(err.Error(), "rank >= 2") {
		t.Fatalf("CausalMask(rank1) err = %v, want rank error", err)
	}

	sZeroQ := mustTensorT(t, []float32{}, []int64{1, 0, 2})
	_, err = CausalMask(sZeroQ, 0)

	if err == nil || !strings.Contains(err.Error(), "positive query/key") {
		t.Fatalf("CausalMask(zero q) err = %v, want positive dims error", err)
	}
}

func TestAttentionErrors(t *testing.T) {
	_, err := Attention(nil, nil, nil, false, 0)
	if err == nil || !strings.Contains(err.Error(), "non-nil") {
		t.Fatalf("Attention(nil) err = %v, want nil input error", err)
	}

	qRank1 := mustTensorT(t, []float32{1, 2}, []int64{2})
	kRank1 := mustTensorT(t, []float32{1, 2}, []int64{2})

	vRank1 := mustTensorT(t, []float32{1, 2}, []int64{2})

	_, err = Attention(qRank1, kRank1, vRank1, false, 0)
	if err == nil || !strings.Contains(err.Error(), "rank >= 2") {
		t.Fatalf("Attention(rank1) err = %v, want rank error", err)
	}

	q := mustTensorT(t, make([]float32, 6), []int64{1, 2, 3})
	kBadD := mustTensorT(t, make([]float32, 8), []int64{1, 2, 4})

	v := mustTensorT(t, make([]float32, 4), []int64{1, 2, 2})

	_, err = Attention(q, kBadD, v, false, 0)
	if err == nil || !strings.Contains(err.Error(), "depth mismatch") {
		t.Fatalf("Attention(depth mismatch) err = %v, want depth mismatch error", err)
	}

	k := mustTensorT(t, make([]float32, 6), []int64{1, 2, 3})

	vBadSeq := mustTensorT(t, make([]float32, 3), []int64{1, 1, 3})

	_, err = Attention(q, k, vBadSeq, false, 0)
	if err == nil || !strings.Contains(err.Error(), "sequence mismatch") {
		t.Fatalf("Attention(sequence mismatch) err = %v, want sequence mismatch error", err)
	}
}

func TestAttention4DMatchesGeneric(t *testing.T) {
	q := mustTensorT(t, seqDataT(2*3*5*4), []int64{2, 3, 5, 4})
	k := mustTensorT(t, seqDataT(2*3*7*4), []int64{2, 3, 7, 4})
	v := mustTensorT(t, seqDataT(2*3*7*6), []int64{2, 3, 7, 6})

	got, err := Attention(q, k, v, true, 1)
	if err != nil {
		t.Fatalf("Attention failed: %v", err)
	}

	want, err := attentionGeneric(q, k, v, true, 1)
	if err != nil {
		t.Fatalf("attentionGeneric failed: %v", err)
	}

	if !equalApprox(got.Data(), want.Data(), 1e-4) {
		t.Fatalf("4D fast-path output mismatch with generic implementation")
	}
}

func TestAttention4DMatchesGenericNonCausal(t *testing.T) {
	q := mustTensorT(t, seqDataT(1*2*4*8), []int64{1, 2, 4, 8})
	k := mustTensorT(t, seqDataT(1*2*6*8), []int64{1, 2, 6, 8})
	v := mustTensorT(t, seqDataT(1*2*6*5), []int64{1, 2, 6, 5})

	got, err := Attention(q, k, v, false, 0)
	if err != nil {
		t.Fatalf("Attention failed: %v", err)
	}

	want, err := attentionGeneric(q, k, v, false, 0)
	if err != nil {
		t.Fatalf("attentionGeneric failed: %v", err)
	}

	if !equalApprox(got.Data(), want.Data(), 1e-4) {
		t.Fatalf("4D fast-path output mismatch with generic implementation")
	}
}

func BenchmarkAttention4DFused(b *testing.B) {
	q := mustTensorB(b, seqDataT(1*16*32*64), []int64{1, 16, 32, 64})
	k := mustTensorB(b, seqDataT(1*16*32*64), []int64{1, 16, 32, 64})
	v := mustTensorB(b, seqDataT(1*16*32*64), []int64{1, 16, 32, 64})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := Attention(q, k, v, true, 0)
		if err != nil {
			b.Fatalf("Attention failed: %v", err)
		}
	}
}

func BenchmarkAttention4DGeneric(b *testing.B) {
	q := mustTensorB(b, seqDataT(1*16*32*64), []int64{1, 16, 32, 64})
	k := mustTensorB(b, seqDataT(1*16*32*64), []int64{1, 16, 32, 64})
	v := mustTensorB(b, seqDataT(1*16*32*64), []int64{1, 16, 32, 64})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := attentionGeneric(q, k, v, true, 0)
		if err != nil {
			b.Fatalf("attentionGeneric failed: %v", err)
		}
	}
}

func BenchmarkAttention4DFusedParallel(b *testing.B) {
	prev := tensor.Workers()

	tensor.SetWorkers(8)
	defer tensor.SetWorkers(prev)

	q := mustTensorB(b, seqDataT(1*16*32*64), []int64{1, 16, 32, 64})
	k := mustTensorB(b, seqDataT(1*16*32*64), []int64{1, 16, 32, 64})
	v := mustTensorB(b, seqDataT(1*16*32*64), []int64{1, 16, 32, 64})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := Attention(q, k, v, true, 0)
		if err != nil {
			b.Fatalf("Attention failed: %v", err)
		}
	}
}

func mustTensorB(b *testing.B, data []float32, shape []int64) *tensor.Tensor {
	b.Helper()

	t, err := tensor.New(data, shape)
	if err != nil {
		b.Fatalf("tensor.New(%v, %v): %v", data, shape, err)
	}

	return t
}
