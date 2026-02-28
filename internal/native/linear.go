package native

import (
	"errors"
	"fmt"
	"math"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

type Linear struct {
	Weight *tensor.Tensor // [out, in]
	Bias   *tensor.Tensor // optional [out]
	inDim  int64
	outDim int64
}

func loadLinear(vb *VarBuilder, name string, withBias bool) (*Linear, error) {
	w, err := vb.Tensor(name + ".weight")
	if err != nil {
		return nil, err
	}

	if len(w.Shape()) != 2 {
		return nil, fmt.Errorf("native: linear %q weight must be rank-2, got %v", name, w.Shape())
	}
	var b *tensor.Tensor

	if withBias {
		t, ok, err := vb.TensorMaybe(name + ".bias")
		if err != nil {
			return nil, err
		}

		if ok {
			if len(t.Shape()) != 1 || t.Shape()[0] != w.Shape()[0] {
				return nil, fmt.Errorf("native: linear %q bias shape %v incompatible with weight %v", name, t.Shape(), w.Shape())
			}

			b = t
		}
	}

	return &Linear{Weight: w, Bias: b, inDim: w.Shape()[1], outDim: w.Shape()[0]}, nil
}

func (l *Linear) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if l == nil || l.Weight == nil {
		return nil, errors.New("native: linear is not initialized")
	}

	if x == nil {
		return nil, errors.New("native: linear input is nil")
	}

	xShape := x.Shape()
	if len(xShape) < 1 {
		return nil, errors.New("native: linear requires x rank >= 1")
	}

	if xShape[len(xShape)-1] != l.inDim {
		return nil, fmt.Errorf("native: linear mismatch: x last dim %d, weight in dim %d", xShape[len(xShape)-1], l.inDim)
	}

	outShape := append([]int64(nil), xShape...)
	outShape[len(outShape)-1] = l.outDim

	out, err := tensor.Zeros(outShape)
	if err != nil {
		return nil, err
	}

	err = l.ForwardInto(x, out)
	if err != nil {
		return nil, err
	}

	return out, nil
}

func (l *Linear) ForwardInto(x, out *tensor.Tensor) error {
	if l == nil || l.Weight == nil {
		return errors.New("native: linear is not initialized")
	}

	if x == nil || out == nil {
		return errors.New("native: linear requires non-nil x and out")
	}

	xShape := x.Shape()
	if len(xShape) < 1 {
		return errors.New("native: linear requires x rank >= 1")
	}

	if xShape[len(xShape)-1] != l.inDim {
		return fmt.Errorf("native: linear mismatch: x last dim %d, weight in dim %d", xShape[len(xShape)-1], l.inDim)
	}

	outShape := out.Shape()
	if len(outShape) != len(xShape) {
		return fmt.Errorf("native: linear out rank mismatch: x %v out %v", xShape, outShape)
	}

	for i := 0; i < len(xShape)-1; i++ {
		if xShape[i] != outShape[i] {
			return fmt.Errorf("native: linear out shape mismatch: x %v out %v", xShape, outShape)
		}
	}

	if outShape[len(outShape)-1] != l.outDim {
		return fmt.Errorf("native: linear out last dim mismatch: got %d want %d", outShape[len(outShape)-1], l.outDim)
	}

	return l.forwardIntoTrusted(x, out)
}

func (l *Linear) forwardIntoTrusted(x, out *tensor.Tensor) error {
	if l == nil || l.Weight == nil {
		return errors.New("native: linear is not initialized")
	}

	if x == nil || out == nil {
		return errors.New("native: linear requires non-nil x and out")
	}

	xData := x.RawData()
	outData := out.RawData()
	wData := l.Weight.RawData()

	inI := int(l.inDim)
	outI := int(l.outDim)
	batch := len(xData) / inI

	if len(outData) != batch*outI {
		return fmt.Errorf("native: linear out data len mismatch: got %d want %d", len(outData), batch*outI)
	}

	var biasData []float32
	if l.Bias != nil {
		biasData = l.Bias.RawData()
	}

	runBatchRange := func(lo, hi int) {
		for bIdx := lo; bIdx < hi; bIdx++ {
			xSlice := xData[bIdx*inI : bIdx*inI+inI]
			yBase := bIdx * outI
			for o := range outI {
				sum := tensor.DotProduct(xSlice, wData[o*inI:(o+1)*inI])
				if biasData != nil {
					sum += biasData[o]
				}
				outData[yBase+o] = sum
			}
		}
	}

	runOutputRangeSingleRow := func(lo, hi int) {
		xSlice := xData[:inI]
		for o := lo; o < hi; o++ {
			sum := tensor.DotProduct(xSlice, wData[o*inI:(o+1)*inI])
			if biasData != nil {
				sum += biasData[o]
			}
			outData[o] = sum
		}
	}

	const linearParallelMinFMAs = int64(1 << 18)
	totalFMAs := int64(batch) * int64(outI) * int64(inI)
	workers := tensor.Workers()

	switch {
	case workers > 1 && totalFMAs >= linearParallelMinFMAs && batch > 1:
		parallelForByWorkers(batch, workers, runBatchRange)
	case workers > 1 && totalFMAs >= linearParallelMinFMAs && batch == 1 && outI > 1:
		parallelForByWorkers(outI, workers, runOutputRangeSingleRow)
	default:
		runBatchRange(0, batch)
	}

	return nil
}

type LayerNorm struct {
	Weight *tensor.Tensor
	Bias   *tensor.Tensor
	Eps    float32
	dim    int64
}

func loadLayerNorm(vb *VarBuilder, name string, eps float32) (*LayerNorm, error) {
	w, err := vb.Tensor(name + ".weight")
	if err != nil {
		return nil, err
	}

	b, err := vb.Tensor(name + ".bias")
	if err != nil {
		return nil, err
	}

	if len(w.Shape()) != 1 || len(b.Shape()) != 1 || w.Shape()[0] != b.Shape()[0] {
		return nil, fmt.Errorf("native: layernorm %q invalid shapes weight=%v bias=%v", name, w.Shape(), b.Shape())
	}

	return &LayerNorm{Weight: w, Bias: b, Eps: eps, dim: w.Shape()[0]}, nil
}

func (ln *LayerNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if ln == nil || ln.Weight == nil || ln.Bias == nil {
		return nil, errors.New("native: layernorm is not initialized")
	}

	if x == nil {
		return nil, errors.New("native: layernorm input is nil")
	}

	out, err := tensor.Zeros(x.Shape())
	if err != nil {
		return nil, err
	}

	err = ln.ForwardInto(x, out)
	if err != nil {
		return nil, err
	}

	return out, nil
}

func (ln *LayerNorm) ForwardInto(x, out *tensor.Tensor) error {
	if ln == nil || ln.Weight == nil || ln.Bias == nil {
		return errors.New("native: layernorm is not initialized")
	}

	if x == nil || out == nil {
		return errors.New("native: layernorm requires non-nil x and out")
	}

	if ln.Eps <= 0 {
		return errors.New("native: layernorm eps must be > 0")
	}

	xShape := x.Shape()
	if len(xShape) < 1 {
		return errors.New("native: layernorm requires rank >= 1")
	}

	if !equalShape(xShape, out.Shape()) {
		return fmt.Errorf("native: layernorm out shape mismatch: x %v out %v", xShape, out.Shape())
	}

	d := xShape[len(xShape)-1]
	if d <= 0 {
		return errors.New("native: layernorm last dimension must be > 0")
	}

	if d != ln.dim {
		return fmt.Errorf("native: layernorm last dimension mismatch: got %d want %d", d, ln.dim)
	}

	return ln.forwardIntoTrusted(x, out)
}

func (ln *LayerNorm) forwardIntoTrusted(x, out *tensor.Tensor) error {
	if ln == nil || ln.Weight == nil || ln.Bias == nil {
		return errors.New("native: layernorm is not initialized")
	}

	if x == nil || out == nil {
		return errors.New("native: layernorm requires non-nil x and out")
	}

	xData := x.RawData()
	outData := out.RawData()
	weightData := ln.Weight.RawData()
	biasData := ln.Bias.RawData()

	dd := int(ln.dim)
	if dd <= 0 {
		return errors.New("native: layernorm invalid normalized dimension")
	}

	if len(xData) != len(outData) || len(xData)%dd != 0 {
		return fmt.Errorf("native: layernorm data shape mismatch: x=%d out=%d dim=%d", len(xData), len(outData), dd)
	}

	outer := len(xData) / dd
	runRows := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			start := o * dd
			src := xData[start : start+dd]
			dst := outData[start : start+dd]

			var mean float64
			for _, v := range src {
				mean += float64(v)
			}

			mean /= float64(dd)

			var variance float64
			for _, v := range src {
				delta := float64(v) - mean
				variance += delta * delta
			}

			variance /= float64(dd)
			invStd := float32(1.0 / math.Sqrt(variance+float64(ln.Eps)))

			for i := range dd {
				n := (src[i] - float32(mean)) * invStd
				n = n*weightData[i] + biasData[i]
				dst[i] = n
			}
		}
	}

	const layerNormParallelMinOps = int64(1 << 17)
	totalOps := int64(len(xData))
	workers := tensor.Workers()
	if workers > 1 && outer > 1 && totalOps >= layerNormParallelMinOps {
		parallelForByWorkers(outer, workers, runRows)
	} else {
		runRows(0, outer)
	}

	return nil
}
