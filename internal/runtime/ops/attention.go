package ops

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

// CausalMask sets positions where key index > query index + offset to -Inf.
// Expected input shape: [..., query, key].
func CausalMask(scores *tensor.Tensor, offset int64) (*tensor.Tensor, error) {
	if scores == nil {
		return nil, errors.New("ops: causal mask scores is nil")
	}

	shape := scores.Shape()
	if len(shape) < 2 {
		return nil, fmt.Errorf("ops: causal mask requires rank >= 2, got %d", len(shape))
	}

	q := int(shape[len(shape)-2])

	k := int(shape[len(shape)-1])
	if q <= 0 || k <= 0 {
		return nil, fmt.Errorf("ops: causal mask requires positive query/key dims, got %d and %d", q, k)
	}

	out := scores.Clone()
	data := out.RawData()
	blocks := len(data) / (q * k)
	applyCausalMaskInPlace(data, q, k, blocks, offset)

	return out, nil
}

// Attention computes scaled dot-product attention.
// q shape: [..., tq, d], k shape: [..., tk, d], v shape: [..., tk, dv]
// output: [..., tq, dv]
func Attention(q, k, v *tensor.Tensor, causal bool, offset int64) (*tensor.Tensor, error) {
	if q == nil || k == nil || v == nil {
		return nil, errors.New("ops: attention requires non-nil q/k/v")
	}

	if out, handled, err := attention4D(q, k, v, causal, offset); handled || err != nil {
		return out, err
	}

	return attentionGeneric(q, k, v, causal, offset)
}

func attentionGeneric(q, k, v *tensor.Tensor, causal bool, offset int64) (*tensor.Tensor, error) {
	qShape := q.Shape()
	kShape := k.Shape()

	vShape := v.Shape()
	if len(qShape) < 2 || len(kShape) < 2 || len(vShape) < 2 {
		return nil, errors.New("ops: attention requires rank >= 2 inputs")
	}

	d := qShape[len(qShape)-1]
	if d != kShape[len(kShape)-1] {
		return nil, fmt.Errorf("ops: attention q/k depth mismatch %d vs %d", d, kShape[len(kShape)-1])
	}

	if kShape[len(kShape)-2] != vShape[len(vShape)-2] {
		return nil, fmt.Errorf("ops: attention key/value sequence mismatch %d vs %d", kShape[len(kShape)-2], vShape[len(vShape)-2])
	}

	kT, err := k.Transpose(-1, -2)
	if err != nil {
		return nil, fmt.Errorf("ops: attention transpose k: %w", err)
	}

	scores, err := tensor.MatMul(q, kT)
	if err != nil {
		return nil, fmt.Errorf("ops: attention q*k^T: %w", err)
	}

	scale := float32(1.0 / math.Sqrt(float64(d)))

	err = scaleMaskSoftmaxInPlace(scores, scale, causal, offset)
	if err != nil {
		return nil, fmt.Errorf("ops: attention scale/mask/softmax: %w", err)
	}

	out, err := tensor.MatMul(scores, v)
	if err != nil {
		return nil, fmt.Errorf("ops: attention probs*v: %w", err)
	}

	return out, nil
}

func attention4D(q, k, v *tensor.Tensor, causal bool, offset int64) (*tensor.Tensor, bool, error) {
	qShape := q.Shape()
	kShape := k.Shape()

	vShape := v.Shape()
	if len(qShape) != 4 || len(kShape) != 4 || len(vShape) != 4 {
		return nil, false, nil
	}

	b, h, tq, d := qShape[0], qShape[1], qShape[2], qShape[3]
	if d != kShape[3] {
		return nil, true, fmt.Errorf("ops: attention q/k depth mismatch %d vs %d", d, kShape[3])
	}

	if kShape[0] != b || kShape[1] != h || vShape[0] != b || vShape[1] != h {
		return nil, true, fmt.Errorf("ops: attention batch/head mismatch q=%v k=%v v=%v", qShape, kShape, vShape)
	}

	tk := kShape[2]
	if vShape[2] != tk {
		return nil, true, fmt.Errorf("ops: attention key/value sequence mismatch %d vs %d", tk, vShape[2])
	}

	dv := vShape[3]
	if b <= 0 || h <= 0 || tq <= 0 || tk <= 0 || d <= 0 || dv <= 0 {
		return nil, true, fmt.Errorf("ops: attention expects positive dims, got q=%v k=%v v=%v", qShape, kShape, vShape)
	}

	out, err := tensor.Zeros([]int64{b, h, tq, dv})
	if err != nil {
		return nil, true, err
	}

	qData := q.RawData()
	kData := k.RawData()
	vData := v.RawData()
	outData := out.RawData()
	scale := float32(1.0 / math.Sqrt(float64(d)))

	bI := int(b)
	hI := int(h)
	tqI := int(tq)
	tkI := int(tk)
	dI := int(d)
	dvI := int(dv)
	qBStride := hI * tqI * dI
	qBHStride := tqI * dI
	kBStride := hI * tkI * dI
	kBHStride := tkI * dI
	vBStride := hI * tkI * dvI
	vBHStride := tkI * dvI
	outBStride := hI * tqI * dvI
	outBHStride := tqI * dvI
	jobsPerHead := tqI
	jobsPerBatch := hI * jobsPerHead
	jobCount := bI * jobsPerBatch

	const attentionParallelMinWork = int64(1 << 20)
	totalWork := int64(jobCount) * int64(tkI) * int64(dI+dvI)
	workers := tensor.Workers()
	runParallel := workers > 1 && jobCount > 1 && totalWork >= attentionParallelMinWork

	var failed atomic.Bool
	var firstErr error
	var errMu sync.Mutex
	setErr := func(err error) {
		if err == nil {
			return
		}

		if failed.CompareAndSwap(false, true) {
			errMu.Lock()
			firstErr = err
			errMu.Unlock()
		}
	}

	runJobs := func(lo, hi int) {
		scores := getScratch(tkI)
		defer putScratch(scores)

		for job := lo; job < hi; job++ {
			if failed.Load() {
				return
			}

			bi := job / jobsPerBatch
			rem := job % jobsPerBatch
			headIdx := rem / jobsPerHead
			qi := rem % jobsPerHead

			qBBase := bi * qBStride
			kBBase := bi * kBStride
			vBBase := bi * vBStride
			outBBase := bi * outBStride
			qBHBase := qBBase + headIdx*qBHStride
			kBHBase := kBBase + headIdx*kBHStride
			vBHBase := vBBase + headIdx*vBHStride
			outBHBase := outBBase + headIdx*outBHStride

			qOff := qBHBase + qi*dI
			qRow := qData[qOff : qOff+dI]

			maxV := float32(math.Inf(-1))

			maxKey := int64(qi) + offset
			for ki := range tkI {
				if causal && int64(ki) > maxKey {
					scores[ki] = float32(math.Inf(-1))
					continue
				}

				kOff := kBHBase + ki*dI
				kRow := kData[kOff : kOff+dI]
				s := tensor.DotProduct(qRow, kRow) * scale

				scores[ki] = s
				if s > maxV {
					maxV = s
				}
			}

			outRow := outData[outBHBase+qi*dvI : outBHBase+(qi+1)*dvI]
			for i := range outRow {
				outRow[i] = 0
			}

			if math.IsInf(float64(maxV), -1) {
				continue
			}

			var sum float64

			for ki := range tkI {
				s := scores[ki]
				if math.IsInf(float64(s), -1) {
					scores[ki] = 0
					continue
				}

				e := math.Exp(float64(s - maxV))
				scores[ki] = float32(e)
				sum += e
			}

			if sum == 0 || math.IsNaN(sum) {
				setErr(errors.New("ops: softmax encountered zero normalization sum"))
				return
			}

			inv := float32(1.0 / sum)
			for ki := range tkI {
				w := scores[ki] * inv
				if w == 0 {
					continue
				}

				vOff := vBHBase + ki*dvI
				vRow := vData[vOff : vOff+dvI]
				tensor.Axpy(outRow, w, vRow)
			}
		}
	}
	if runParallel {
		parallelFor(jobCount, workers, runJobs)
	} else {
		runJobs(0, jobCount)
	}

	if firstErr != nil {
		return nil, true, firstErr
	}

	return out, true, nil
}

func applyCausalMaskInPlace(data []float32, q, k, blocks int, offset int64) {
	negInf := float32(math.Inf(-1))

	for b := range blocks {
		base := b * q * k
		for qi := range q {
			maxKey := int64(qi) + offset

			row := base + qi*k
			for ki := range k {
				if int64(ki) > maxKey {
					data[row+ki] = negInf
				}
			}
		}
	}
}

func scaleMaskSoftmaxInPlace(scores *tensor.Tensor, scale float32, causal bool, offset int64) error {
	if scores == nil {
		return errors.New("ops: softmax input is nil")
	}

	shape := scores.Shape()
	if len(shape) < 2 {
		return fmt.Errorf("ops: softmax expects rank >= 2, got %d", len(shape))
	}

	q := int(shape[len(shape)-2])
	k := int(shape[len(shape)-1])

	if q <= 0 || k <= 0 {
		return fmt.Errorf("ops: softmax expects positive query/key dims, got %d and %d", q, k)
	}

	data := scores.RawData()

	blocks := len(data) / (q * k)
	for b := range blocks {
		base := b * q * k
		for qi := range q {
			row := base + qi*k
			maxKey := int64(qi) + offset
			maxV := float32(math.Inf(-1))

			for ki := range k {
				idx := row + ki

				v := data[idx] * scale
				if causal && int64(ki) > maxKey {
					v = float32(math.Inf(-1))
				}

				data[idx] = v
				if v > maxV {
					maxV = v
				}
			}

			if math.IsInf(float64(maxV), -1) {
				for ki := range k {
					data[row+ki] = 0
				}

				continue
			}

			var sum float64

			for ki := range k {
				idx := row + ki
				e := math.Exp(float64(data[idx] - maxV))
				data[idx] = float32(e)
				sum += e
			}

			if sum == 0 || math.IsNaN(sum) {
				return errors.New("ops: softmax encountered zero normalization sum")
			}

			inv := float32(1.0 / sum)

			for ki := range k {
				idx := row + ki
				data[idx] *= inv
			}
		}
	}

	return nil
}
