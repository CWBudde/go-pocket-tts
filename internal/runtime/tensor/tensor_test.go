package tensor

import (
	"math"
	"testing"
)

func TestReshapePreservesValues(t *testing.T) {
	x, err := New([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	y, err := x.Reshape([]int64{3, 2})
	if err != nil {
		t.Fatalf("reshape: %v", err)
	}
	if got := y.Shape(); !equalI64(got, []int64{3, 2}) {
		t.Fatalf("shape = %v, want [3 2]", got)
	}
	if got := y.Data(); !equalF32(got, []float32{1, 2, 3, 4, 5, 6}, 0) {
		t.Fatalf("data = %v", got)
	}
}

func TestTranspose2D(t *testing.T) {
	x, _ := New([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	y, err := x.Transpose(0, 1)
	if err != nil {
		t.Fatalf("transpose: %v", err)
	}
	if got := y.Shape(); !equalI64(got, []int64{3, 2}) {
		t.Fatalf("shape = %v, want [3 2]", got)
	}
	want := []float32{1, 4, 2, 5, 3, 6}
	if got := y.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func TestConcatDim1(t *testing.T) {
	a, _ := New([]float32{1, 2, 3, 4}, []int64{1, 2, 2})
	b, _ := New([]float32{5, 6, 7, 8}, []int64{1, 2, 2})
	out, err := Concat([]*Tensor{a, b}, 1)
	if err != nil {
		t.Fatalf("concat: %v", err)
	}
	if got := out.Shape(); !equalI64(got, []int64{1, 4, 2}) {
		t.Fatalf("shape = %v, want [1 4 2]", got)
	}
	want := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	if got := out.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func TestNarrow(t *testing.T) {
	x, _ := New([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	out, err := x.Narrow(1, 1, 2)
	if err != nil {
		t.Fatalf("narrow: %v", err)
	}
	if got := out.Shape(); !equalI64(got, []int64{2, 2}) {
		t.Fatalf("shape = %v, want [2 2]", got)
	}
	want := []float32{2, 3, 5, 6}
	if got := out.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func TestGather(t *testing.T) {
	x, _ := New([]float32{10, 20, 30, 40, 50, 60}, []int64{2, 3})
	out, err := x.Gather(1, []int64{2, 0})
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	if got := out.Shape(); !equalI64(got, []int64{2, 2}) {
		t.Fatalf("shape = %v, want [2 2]", got)
	}
	want := []float32{30, 10, 60, 40}
	if got := out.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func TestBroadcastAddMul(t *testing.T) {
	a, _ := New([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	b, _ := New([]float32{10, 20, 30}, []int64{1, 3})
	add, err := BroadcastAdd(a, b)
	if err != nil {
		t.Fatalf("broadcast add: %v", err)
	}
	wantAdd := []float32{11, 22, 33, 14, 25, 36}
	if got := add.Data(); !equalF32(got, wantAdd, 0) {
		t.Fatalf("add = %v, want %v", got, wantAdd)
	}

	mul, err := BroadcastMul(a, b)
	if err != nil {
		t.Fatalf("broadcast mul: %v", err)
	}
	wantMul := []float32{10, 40, 90, 40, 100, 180}
	if got := mul.Data(); !equalF32(got, wantMul, 0) {
		t.Fatalf("mul = %v, want %v", got, wantMul)
	}
}

func TestSoftmax(t *testing.T) {
	x, _ := New([]float32{1, 2, 3}, []int64{3})
	out, err := Softmax(x, 0)
	if err != nil {
		t.Fatalf("softmax: %v", err)
	}
	want := []float32{0.09003057, 0.24472848, 0.66524094}
	if got := out.Data(); !equalF32(got, want, 1e-5) {
		t.Fatalf("softmax = %v, want ~%v", got, want)
	}
}

func TestLayerNorm(t *testing.T) {
	x, _ := New([]float32{1, 2, 3, 4}, []int64{1, 4})
	w, _ := New([]float32{1, 1, 1, 1}, []int64{4})
	b, _ := New([]float32{0, 0, 0, 0}, []int64{4})
	out, err := LayerNorm(x, w, b, 1e-5)
	if err != nil {
		t.Fatalf("layernorm: %v", err)
	}
	got := out.Data()
	want := []float32{-1.3416355, -0.44721183, 0.44721183, 1.3416355}
	if !equalF32(got, want, 1e-4) {
		t.Fatalf("layernorm = %v, want ~%v", got, want)
	}
}

func TestMatMul2D(t *testing.T) {
	a, _ := New([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	b, _ := New([]float32{7, 8, 9, 10, 11, 12}, []int64{3, 2})
	out, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("matmul: %v", err)
	}
	if got := out.Shape(); !equalI64(got, []int64{2, 2}) {
		t.Fatalf("shape = %v, want [2 2]", got)
	}
	want := []float32{58, 64, 139, 154}
	if got := out.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func TestMatMulBatchBroadcast(t *testing.T) {
	a, _ := New([]float32{
		1, 2, 3, 4, // batch 0, 2x2
		5, 6, 7, 8, // batch 1, 2x2
	}, []int64{2, 2, 2})
	b, _ := New([]float32{1, 0, 0, 1}, []int64{1, 2, 2})
	out, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("matmul: %v", err)
	}
	if got := out.Shape(); !equalI64(got, []int64{2, 2, 2}) {
		t.Fatalf("shape = %v, want [2 2 2]", got)
	}
	if got := out.Data(); !equalF32(got, a.Data(), 0) {
		t.Fatalf("data = %v, want %v", got, a.Data())
	}
}

func TestLinear(t *testing.T) {
	x, _ := New([]float32{1, 2, 3, 4}, []int64{2, 2})
	w, _ := New([]float32{1, 0, 0, 1, 1, 1}, []int64{3, 2})
	b, _ := New([]float32{0.5, -0.5, 1}, []int64{3})
	out, err := Linear(x, w, b)
	if err != nil {
		t.Fatalf("linear: %v", err)
	}
	if got := out.Shape(); !equalI64(got, []int64{2, 3}) {
		t.Fatalf("shape = %v, want [2 3]", got)
	}
	want := []float32{1.5, 1.5, 4, 3.5, 3.5, 8}
	if got := out.Data(); !equalF32(got, want, 0) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func equalI64(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalF32(a, b []float32, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if tol == 0 {
			if a[i] != b[i] {
				return false
			}
			continue
		}
		if math.Abs(float64(a[i]-b[i])) > tol {
			return false
		}
	}
	return true
}
