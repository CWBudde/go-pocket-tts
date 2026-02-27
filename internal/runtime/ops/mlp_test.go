package ops

import (
	"math"
	"strings"
	"testing"
)

func TestMLP(t *testing.T) {
	x := mustTensorT(t, []float32{1, -1}, []int64{1, 2})
	w1 := mustTensorT(t, []float32{1, 0, 0, 1}, []int64{2, 2})
	w2 := mustTensorT(t, []float32{1, 1}, []int64{1, 2})

	out, err := MLP(x, w1, nil, w2, nil)
	if err != nil {
		t.Fatalf("mlp: %v", err)
	}

	if got := out.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 1 {
		t.Fatalf("shape = %v, want [1 1]", got)
	}

	if got := out.Data()[0]; math.Abs(float64(got-0.4621172)) > 1e-4 {
		t.Fatalf("mlp output = %f, want ~0.4621172", got)
	}
}

func TestMLPErrors(t *testing.T) {
	x := mustTensorT(t, []float32{1, 2}, []int64{1, 2})
	wGood := mustTensorT(t, []float32{1, 0, 0, 1}, []int64{2, 2})

	_, err := MLP(nil, wGood, nil, wGood, nil)
	if err == nil || !strings.Contains(err.Error(), "first linear") {
		t.Fatalf("MLP(nil x) err = %v, want first linear error", err)
	}

	wBadSecond := mustTensorT(t, []float32{1, 2, 3}, []int64{1, 3})

	_, err = MLP(x, wGood, nil, wBadSecond, nil)
	if err == nil || !strings.Contains(err.Error(), "second linear") {
		t.Fatalf("MLP(bad second linear) err = %v, want second linear error", err)
	}
}
