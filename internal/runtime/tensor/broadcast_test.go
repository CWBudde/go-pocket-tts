package tensor

import "testing"

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

func TestLeftPadShape(t *testing.T) {
	shape := []int64{2, 3}

	// Equal rank should return a copy.
	gotEqual := leftPadShape(shape, 2)
	if !equalI64(gotEqual, []int64{2, 3}) {
		t.Fatalf("leftPadShape equal rank = %v, want [2 3]", gotEqual)
	}

	gotEqual[0] = 99

	if shape[0] != 2 {
		t.Fatalf("leftPadShape should return a copy when rank matches, source mutated: %v", shape)
	}

	gotPadded := leftPadShape(shape, 4)
	if !equalI64(gotPadded, []int64{1, 1, 2, 3}) {
		t.Fatalf("leftPadShape padded = %v, want [1 1 2 3]", gotPadded)
	}
}
