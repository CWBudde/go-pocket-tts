package tensor

import "testing"

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
