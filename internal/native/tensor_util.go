package native

import (
	"errors"
	"fmt"
	"math"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

func addSameShape(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if a == nil || b == nil {
		return nil, errors.New("native: add requires non-nil tensors")
	}

	if !equalShape(a.Shape(), b.Shape()) {
		return nil, fmt.Errorf("native: add shape mismatch %v vs %v", a.Shape(), b.Shape())
	}

	out := a.Clone()
	od := out.RawData()

	bd := b.RawData()
	for i := range od {
		od[i] += bd[i]
	}

	return out, nil
}

func mulSameShape(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if a == nil || b == nil {
		return nil, errors.New("native: mul requires non-nil tensors")
	}

	if !equalShape(a.Shape(), b.Shape()) {
		return nil, fmt.Errorf("native: mul shape mismatch %v vs %v", a.Shape(), b.Shape())
	}

	out := a.Clone()
	od := out.RawData()

	bd := b.RawData()
	for i := range od {
		od[i] *= bd[i]
	}

	return out, nil
}

func scaleTensor(x *tensor.Tensor, s float32) *tensor.Tensor {
	out := x.Clone()

	d := out.RawData()
	for i := range d {
		d[i] *= s
	}

	return out
}

func addScalar(x *tensor.Tensor, s float32) *tensor.Tensor {
	out := x.Clone()

	d := out.RawData()
	for i := range d {
		d[i] += s
	}

	return out
}

func siluTensor(x *tensor.Tensor) *tensor.Tensor {
	out := x.Clone()

	d := out.RawData()
	for i, v := range d {
		d[i] = v / (1 + float32(math.Exp(float64(-v))))
	}

	return out
}

func geluErfTensor(x *tensor.Tensor) *tensor.Tensor {
	out := x.Clone()

	d := out.RawData()
	for i, v := range d {
		fv := float64(v)
		d[i] = float32(0.5 * fv * (1 + math.Erf(fv/math.Sqrt2)))
	}

	return out
}

func geluErfTensorInPlace(x *tensor.Tensor) *tensor.Tensor {
	d := x.RawData()
	for i, v := range d {
		fv := float64(v)
		d[i] = float32(0.5 * fv * (1 + math.Erf(fv/math.Sqrt2)))
	}

	return x
}

func eluTensor(x *tensor.Tensor) *tensor.Tensor {
	out := x.Clone()

	d := out.RawData()
	for i, v := range d {
		if v <= 0 {
			d[i] = float32(math.Exp(float64(v))) - 1
		}
	}

	return out
}

func eluTensorInPlace(x *tensor.Tensor) *tensor.Tensor {
	d := x.RawData()
	for i, v := range d {
		if v <= 0 {
			d[i] = float32(math.Exp(float64(v))) - 1
		}
	}

	return x
}

func lastToken(x *tensor.Tensor) (*tensor.Tensor, error) {
	shape := x.Shape()
	if len(shape) != 3 || shape[1] < 1 {
		return nil, fmt.Errorf("native: lastToken expects [B, T, D] with T>=1, got %v", shape)
	}

	last, err := x.Narrow(1, shape[1]-1, 1)
	if err != nil {
		return nil, err
	}

	return last.Reshape([]int64{shape[0], shape[2]})
}

func splitLastDim3(x *tensor.Tensor) (a, b, c *tensor.Tensor, err error) {
	shape := x.Shape()
	if len(shape) < 1 {
		return nil, nil, nil, errors.New("native: splitLastDim3 requires rank >= 1")
	}

	last := shape[len(shape)-1]
	if last%3 != 0 {
		return nil, nil, nil, fmt.Errorf("native: splitLastDim3 last dim %d is not divisible by 3", last)
	}

	chunk := last / 3

	a, err = x.Narrow(-1, 0, chunk)
	if err != nil {
		return nil, nil, nil, err
	}

	b, err = x.Narrow(-1, chunk, chunk)
	if err != nil {
		return nil, nil, nil, err
	}

	c, err = x.Narrow(-1, 2*chunk, chunk)
	if err != nil {
		return nil, nil, nil, err
	}

	return a, b, c, nil
}

func modulate(x, shift, scale *tensor.Tensor) (*tensor.Tensor, error) {
	if x == nil || shift == nil || scale == nil {
		return nil, errors.New("native: modulate requires non-nil tensors")
	}

	onePlusScale := addScalar(scale, 1.0)

	mul, err := tensor.BroadcastMul(x, onePlusScale)
	if err != nil {
		return nil, fmt.Errorf("native: modulate mul: %w", err)
	}

	out, err := tensor.BroadcastAdd(mul, shift)
	if err != nil {
		return nil, fmt.Errorf("native: modulate add: %w", err)
	}

	return out, nil
}

func addSameShapeInPlace(dst, src *tensor.Tensor) (*tensor.Tensor, error) {
	if dst == nil || src == nil {
		return nil, errors.New("native: add requires non-nil tensors")
	}

	if !equalShape(dst.Shape(), src.Shape()) {
		return nil, fmt.Errorf("native: add shape mismatch %v vs %v", dst.Shape(), src.Shape())
	}

	dd := dst.RawData()
	sd := src.RawData()
	for i := range dd {
		dd[i] += sd[i]
	}

	return dst, nil
}

func mulLastDimInPlace(x, scale *tensor.Tensor) (*tensor.Tensor, error) {
	if x == nil || scale == nil {
		return nil, errors.New("native: mulLastDimInPlace requires non-nil tensors")
	}

	xShape := x.Shape()
	if len(xShape) < 1 {
		return nil, errors.New("native: mulLastDimInPlace rank must be >= 1")
	}

	sShape := scale.Shape()
	if len(sShape) != 1 {
		return nil, fmt.Errorf("native: mulLastDimInPlace scale must be rank-1, got %v", sShape)
	}

	last := int(xShape[len(xShape)-1])
	if last <= 0 || int(sShape[0]) != last {
		return nil, fmt.Errorf("native: mulLastDimInPlace scale length mismatch: got %v want %d", sShape, last)
	}

	xd := x.RawData()
	sd := scale.RawData()
	for i := range xd {
		xd[i] *= sd[i%last]
	}

	return x, nil
}

func replaceNaNWithVector(x, vec *tensor.Tensor) (*tensor.Tensor, error) {
	if x == nil || vec == nil {
		return nil, errors.New("native: replaceNaNWithVector requires non-nil tensors")
	}

	xShape := x.Shape()
	if len(xShape) == 0 {
		return nil, errors.New("native: replaceNaNWithVector input rank must be >=1")
	}

	d := xShape[len(xShape)-1]

	vShape := vec.Shape()
	if len(vShape) != 1 || vShape[0] != d {
		return nil, fmt.Errorf("native: replaceNaNWithVector vector shape %v incompatible with last dim %d", vShape, d)
	}

	out := x.Clone()
	od := out.RawData()
	vd := vec.RawData()
	dd := int(d)

	for i := range od {
		if math.IsNaN(float64(od[i])) {
			od[i] = vd[i%dd]
		}
	}

	return out, nil
}

func rmsNormWithAlpha(x, alpha *tensor.Tensor, eps float32) (*tensor.Tensor, error) {
	if x == nil || alpha == nil {
		return nil, errors.New("native: rmsNormWithAlpha requires non-nil tensors")
	}

	shape := x.Shape()
	if len(shape) < 1 {
		return nil, errors.New("native: rmsNormWithAlpha rank must be >=1")
	}

	d := shape[len(shape)-1]
	if d <= 0 {
		return nil, errors.New("native: rmsNormWithAlpha last dim must be >0")
	}

	if aShape := alpha.Shape(); len(aShape) != 1 || aShape[0] != d {
		return nil, fmt.Errorf("native: rmsNormWithAlpha alpha shape %v incompatible with last dim %d", aShape, d)
	}

	out := x.Clone()
	xd := out.RawData()
	ad := alpha.RawData()
	dd := int(d)

	outer := len(xd) / dd
	for i := range outer {
		base := i * dd
		// Python _rms_norm uses x.var(dim=-1) which computes variance with
		// Bessel correction (N-1 denominator), NOT mean(x^2).
		var mean float64
		for j := range dd {
			mean += float64(xd[base+j])
		}

		mean /= float64(dd)
		var variance float64

		for j := range dd {
			diff := float64(xd[base+j]) - mean
			variance += diff * diff
		}

		if dd > 1 {
			variance /= float64(dd - 1) // Bessel correction (torch default)
		}

		inv := float32(1.0 / math.Sqrt(variance+float64(eps)))
		for j := range dd {
			xd[base+j] = xd[base+j] * inv * ad[j]
		}
	}

	return out, nil
}
