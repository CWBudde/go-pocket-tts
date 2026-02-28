package tensor

import (
	"fmt"
	"math"
	"testing"
)

func TestAxpy(t *testing.T) {
	tests := []struct {
		name  string
		dst   []float32
		alpha float32
		src   []float32
		want  []float32
	}{
		{
			name:  "basic",
			dst:   []float32{1, 2, 3},
			alpha: 0.5,
			src:   []float32{4, 5, 6},
			want:  []float32{3, 4.5, 6},
		},
		{
			name:  "empty",
			dst:   nil,
			alpha: 1,
			src:   nil,
			want:  nil,
		},
		{
			name:  "length mismatch uses shorter input",
			dst:   []float32{1, 2, 3, 4},
			alpha: 2,
			src:   []float32{10, 20},
			want:  []float32{21, 42, 3, 4},
		},
		{
			name:  "zero alpha no change",
			dst:   []float32{1, 2, 3},
			alpha: 0,
			src:   []float32{9, 9, 9},
			want:  []float32{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := append([]float32(nil), tt.dst...)
			Axpy(got, tt.alpha, tt.src)
			if len(got) != len(tt.want) {
				t.Fatalf("len(got)=%d want=%d", len(got), len(tt.want))
			}

			for i := range got {
				if math.Abs(float64(got[i]-tt.want[i])) > 1e-5 {
					t.Fatalf("got[%d]=%v want=%v (all got=%v)", i, got[i], tt.want[i], got)
				}
			}
		})
	}
}

func BenchmarkAxpy(b *testing.B) {
	for _, n := range []int{8, 64, 512, 4096} {
		dst := make([]float32, n)
		src := make([]float32, n)
		for i := range n {
			dst[i] = float32(i) * 0.1
			src[i] = float32(n-i) * 0.05
		}

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for range b.N {
				Axpy(dst, 0.7, src)
			}
		})
	}
}
