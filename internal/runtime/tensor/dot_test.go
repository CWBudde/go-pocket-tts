package tensor

import (
	"fmt"
	"math"
	"testing"
)

func TestDotProduct(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"empty", nil, nil, 0},
		{"single", []float32{3}, []float32{4}, 12},
		{"basic", []float32{1, 2, 3}, []float32{4, 5, 6}, 32},
		{"negative", []float32{-1, 2}, []float32{3, -4}, -11},
		{"zeros", []float32{0, 0, 0}, []float32{1, 2, 3}, 0},
		{"long enough for AVX2", make16(1), make16(2), 32}, // 16 elems of 1*2 = 32
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DotProduct(tt.a, tt.b)
			if math.Abs(float64(got-tt.want)) > 1e-5 {
				t.Fatalf("DotProduct = %v; want %v", got, tt.want)
			}
		})
	}
}

func BenchmarkDotProduct(b *testing.B) {
	for _, n := range []int{8, 64, 512, 4096} {
		a := make([]float32, n)
		v := make([]float32, n)

		for i := range a {
			a[i] = float32(i) * 0.01
			v[i] = float32(n-i) * 0.01
		}

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for range b.N {
				DotProduct(a, v)
			}
		})
	}
}

func BenchmarkDotProductGeneric(b *testing.B) {
	for _, n := range []int{8, 64, 512, 4096} {
		a := make([]float32, n)
		v := make([]float32, n)

		for i := range a {
			a[i] = float32(i) * 0.01
			v[i] = float32(n-i) * 0.01
		}

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for range b.N {
				dotF32Generic(a, v)
			}
		})
	}
}
