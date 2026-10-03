package native

import (
	"math"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
)

func TestMulLastDimInPlaceMatchesBroadcastMul(t *testing.T) {
	t.Parallel()

	x, err := tensor.New([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}, []int64{1, 2, 4})
	if err != nil {
		t.Fatalf("x: %v", err)
	}

	scale, err := tensor.New([]float32{0.5, -1, 2, 0.25}, []int64{4})
	if err != nil {
		t.Fatalf("scale: %v", err)
	}

	got := x.Clone()

	got, err = mulLastDimInPlace(got, scale)
	if err != nil {
		t.Fatalf("mulLastDimInPlace: %v", err)
	}

	want, err := tensor.BroadcastMul(x, scale)
	if err != nil {
		t.Fatalf("BroadcastMul: %v", err)
	}

	const eps = 1e-6
	gd := got.RawData()

	wd := want.RawData()
	if len(gd) != len(wd) {
		t.Fatalf("len mismatch: got %d want %d", len(gd), len(wd))
	}

	for i := range gd {
		if math.Abs(float64(gd[i]-wd[i])) > eps {
			t.Fatalf("value mismatch at %d: got %.8f want %.8f", i, gd[i], wd[i])
		}
	}
}

func TestGELUTanhTensor_ReferenceValues(t *testing.T) {
	t.Parallel()

	// Reference values of upstream F.gelu(x, approximate="tanh"):
	// 0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³))), evaluated in float64.
	// erf is the exact GELU; it differs by up to ~4e-4 at |x| = 3.
	cases := []struct{ x, tanh, erf float64 }{
		{-3, -0.003637392, -0.004049694},
		{-1.5, -0.100428423, -0.100210802},
		{-1, -0.158808009, -0.158655254},
		{-0.5, -0.154285990, -0.154268769},
		{0, 0, 0},
		{0.5, 0.345714010, 0.345731231},
		{1, 0.841191991, 0.841344746},
		{3, 2.996362608, 2.995950306},
	}

	in := make([]float32, len(cases))
	for i, c := range cases {
		in[i] = float32(c.x)
	}

	x, err := tensor.New(in, []int64{int64(len(in))})
	if err != nil {
		t.Fatalf("x: %v", err)
	}

	got := geluTanhTensor(x).RawData()
	inPlace := geluTanhTensorInPlace(x.Clone()).RawData()

	for i, c := range cases {
		if math.Abs(float64(got[i])-c.tanh) > 2e-6 {
			t.Errorf("gelu_tanh(%v) = %.9f, want %.9f", c.x, got[i], c.tanh)
		}

		if inPlace[i] != got[i] {
			t.Errorf("in-place gelu_tanh(%v) = %.9f, copy = %.9f", c.x, inPlace[i], got[i])
		}

		if math.Abs(c.x) >= 1 && math.Abs(float64(got[i])-c.erf) < 1e-4 {
			t.Errorf("gelu_tanh(%v) = %.9f matches erf GELU %.9f", c.x, got[i], c.erf)
		}
	}

	if in[0] != -3 {
		t.Errorf("geluTanhTensor modified its input: %v", in)
	}
}

func BenchmarkGELU(b *testing.B) {
	data := make([]float32, 1<<20)
	for i := range data {
		data[i] = float32(i%2001-1000) / 250
	}

	x, err := tensor.New(data, []int64{int64(len(data))})
	if err != nil {
		b.Fatal(err)
	}

	buf := x.Clone()

	b.Run("Tanh", func(b *testing.B) {
		for range b.N {
			copy(buf.RawData(), data)
			geluTanhTensorInPlace(buf)
		}
	})

	// Erf is the exact GELU the transformers used before upstream #278.
	b.Run("Erf", func(b *testing.B) {
		for range b.N {
			d := buf.RawData()
			copy(d, data)

			for i, v := range d {
				fv := float64(v)
				d[i] = float32(0.5 * fv * (1 + math.Erf(fv/math.Sqrt2)))
			}
		}
	})
}

func TestAddSameShapeInPlaceRejectsShapeMismatch(t *testing.T) {
	t.Parallel()

	a, err := tensor.New([]float32{1, 2, 3, 4}, []int64{2, 2})
	if err != nil {
		t.Fatalf("a: %v", err)
	}

	b, err := tensor.New([]float32{1, 2, 3}, []int64{3})
	if err != nil {
		t.Fatalf("b: %v", err)
	}

	_, err = addSameShapeInPlace(a, b)
	if err == nil {
		t.Fatal("expected shape mismatch error")
	}
}
