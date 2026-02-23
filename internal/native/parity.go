package native

import (
	"fmt"
	"math"

	"github.com/example/go-pocket-tts/internal/runtime/ops"
	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

type TensorParityReport struct {
	Name       string
	ShapeMatch bool
	MaxAbsErr  float64
	MaxRelErr  float64
	Tolerance  ops.Tolerance
	Pass       bool
}

func CompareTensor(name string, got, want *tensor.Tensor, tol ops.Tolerance) (TensorParityReport, error) {
	r := TensorParityReport{Name: name, Tolerance: tol}
	if got == nil || want == nil {
		return r, fmt.Errorf("native parity: %s got/want tensor must be non-nil", name)
	}
	gShape := got.Shape()
	wShape := want.Shape()
	if !equalShape(gShape, wShape) {
		r.ShapeMatch = false
		r.Pass = false
		return r, nil
	}
	r.ShapeMatch = true

	gd := got.RawData()
	wd := want.RawData()
	if len(gd) != len(wd) {
		r.Pass = false
		return r, fmt.Errorf("native parity: %s data length mismatch %d vs %d", name, len(gd), len(wd))
	}

	for i := range gd {
		a := float64(gd[i])
		b := float64(wd[i])
		absErr := math.Abs(a - b)
		if absErr > r.MaxAbsErr {
			r.MaxAbsErr = absErr
		}
		den := math.Abs(b)
		relErr := absErr
		if den > 0 {
			relErr = absErr / den
		}
		if relErr > r.MaxRelErr {
			r.MaxRelErr = relErr
		}
	}
	r.Pass = r.MaxAbsErr <= tol.Abs && r.MaxRelErr <= tol.Rel
	return r, nil
}
