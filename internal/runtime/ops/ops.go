package ops

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

// convWorkers controls the number of goroutines used by the parallel Conv1D
// and ConvTranspose1D fast paths.  A value of 0 or 1 means sequential
// (default).  Values ≥ 2 enable parallel execution.
//
// Set via SetConvWorkers, typically wired to --conv-workers.
var convWorkers atomic.Int32

// SetConvWorkers sets the maximum number of goroutines used for parallel
// Conv1D / ConvTranspose1D execution.  n ≤ 1 disables parallelism.
func SetConvWorkers(n int) {
	if n < 0 {
		n = 0
	}

	convWorkers.Store(int32(n))
}

// getConvWorkers returns the current worker count (0 or 1 → sequential).
func getConvWorkers() int { return int(convWorkers.Load()) }

// parallelFor splits the range [0, n) into chunks and runs fn(lo, hi)
// concurrently.  When workers ≤ 1 the call is sequential (no goroutines).
func parallelFor(n, workers int, fn func(lo, hi int)) {
	if workers <= 1 || n <= 1 {
		fn(0, n)
		return
	}

	if workers > n {
		workers = n
	}
	var wg sync.WaitGroup

	chunk := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)

		wg.Add(1)

		go func(lo, hi int) {
			defer wg.Done()

			fn(lo, hi)
		}(lo, hi)
	}

	wg.Wait()
}

// scratchPool is a size-class pool for reusable []float32 scratch buffers.
// It avoids the multi-MB per-call allocations in the im2col and kernel-repack
// paths (conv1DFastGroups1, convTranspose1DGroups1).
//
// Size classes are powers of two from 2^10 (1 Ki) to 2^26 (64 Mi floats ≈ 256 MB).
// A request for n floats rounds up to the next power-of-two class.
var scratchPools [17]sync.Pool // indices 10..26 → pools[0..16]

// getScratch returns a zeroed []float32 of exactly n elements from the pool.
// The caller MUST call putScratch when done.
func getScratch(n int) []float32 {
	cls := scratchClass(n)
	sz := 1 << (cls + 10)
	// If the rounded-up class size is smaller than n (overflow past maxPoolClass),
	// fall back to a plain allocation — it will not be pooled on return.
	if sz < n {
		return make([]float32, n)
	}

	if v := scratchPools[cls].Get(); v != nil {
		buf := *v.(*[]float32)
		// The backing array may be larger than n; re-slice and zero.
		buf = buf[:n]
		for i := range buf {
			buf[i] = 0
		}

		return buf
	}
	// Allocate at the class size so the buffer is reusable for smaller requests
	// in the same class.
	buf := make([]float32, sz)

	return buf[:n]
}

// putScratch returns a buffer obtained from getScratch back to the pool.
// Oversized buffers (that bypassed the pool in getScratch) are silently dropped.
func putScratch(buf []float32) {
	c := cap(buf)

	cls := scratchClass(c)
	if 1<<(cls+10) < c {
		return // oversized — let GC reclaim it
	}
	// Restore full backing-array capacity so the next getScratch can re-slice.
	buf = buf[:c]
	scratchPools[cls].Put(&buf)
}

// scratchClass returns the pool index for a buffer of n elements.
func scratchClass(n int) int {
	if n <= 1<<10 {
		return 0
	}
	// Bit length of (n-1) gives the exponent for the next power of two.
	bits := 0

	v := n - 1
	for v > 0 {
		v >>= 1
		bits++
	}

	cls := max(bits-10, 0)

	if cls > 16 {
		cls = 16
	}

	return cls
}

// conv1DFastGroups1 is the im2col fast path for Conv1D with groups=1.
//
// It rearranges the convolution into a GEMM by building a patch matrix
// (im2col) of shape [outLength, inChannels*kernelSize] where each row contains
// the gathered input values for one output position.  The GEMM then becomes:
//
//	out[oc, ox] = dotProduct(kernel[oc, :], imcol[ox, :]) + bias[oc]
//
// Both the kernel row and the im2col row are contiguous in memory, so the
// AVX2/FMA dotProduct kernel runs at full throughput.
func conv1DFastGroups1(
	inputData, kernelData, biasData []float32,
	batch, inCh, length, outCh, kSize, outLen,
	stride, padding, dilation int64,
	outData []float32,
) {
	patchLen := int(inCh * kSize)
	imcolSize := int(outLen) * patchLen

	imcol := getScratch(imcolSize) // [outLen, inCh*kSize]
	defer putScratch(imcol)

	kSizeI := int(kSize)
	outChI := int(outCh)
	outLenI := int(outLen)
	lenI := int(length)

	for b := range batch {
		// Zero im2col (ensures padding positions stay 0).
		// getScratch already zeroed, but we must re-zero for b > 0.
		if b > 0 {
			for i := range imcol {
				imcol[i] = 0
			}
		}

		// Build im2col: for each (ic, kx) column, copy valid input positions.
		// Iterating (ic, kx) in outer loops and ox in inner loop keeps the
		// writes to imcol sequential (stride = patchLen across rows, consecutive
		// columns within a row).
		for ic := range inCh {
			inBase := int(b*inCh+ic) * lenI
			for kx := range kSize {
				col := int(ic)*kSizeI + int(kx)
				for ox := range outLen {
					inPos := ox*stride - padding + kx*dilation
					if inPos >= 0 && inPos < length {
						imcol[int(ox)*patchLen+col] = inputData[inBase+int(inPos)]
					}
				}
			}
		}

		// GEMM: kernel [outCh, patchLen] × imcol^T [patchLen, outLen] → out [outCh, outLen].
		// The oc loop is embarrassingly parallel: each output channel writes to
		// a disjoint slice of outData and reads shared (immutable) imcol + kernel.
		outBase := int(b) * outChI * outLenI
		parallelFor(outChI, getConvWorkers(), func(ocLo, ocHi int) {
			for oc := ocLo; oc < ocHi; oc++ {
				kernelRow := kernelData[oc*patchLen : (oc+1)*patchLen]

				biasVal := float32(0)
				if biasData != nil {
					biasVal = biasData[oc]
				}

				outOC := outData[outBase+oc*outLenI : outBase+(oc+1)*outLenI]
				for ox := range outLenI {
					outOC[ox] = tensor.DotProduct(kernelRow, imcol[ox*patchLen:(ox+1)*patchLen]) + biasVal
				}
			}
		})
	}
}

// RepackConvTransposeKernel repacks a ConvTranspose1D weight tensor from the
// standard [inCh, outCh, kSize] layout to [kSize, outCh, inCh] so that each
// (kx, oc) slice is contiguous for AVX2 dot-product acceleration.
//
// Call this once at model load time and pass the result to ConvTranspose1DPrePacked
// to avoid the per-call repack cost.
func RepackConvTransposeKernel(kernel *tensor.Tensor) []float32 {
	s := kernel.Shape() // [inCh, outCh, kSize]
	inChI := int(s[0])
	outChI := int(s[1])
	kSizeI := int(s[2])
	data := kernel.RawData()

	kernelT := make([]float32, kSizeI*outChI*inChI)
	for ic := range inChI {
		for oc := range outChI {
			for kx := range kSizeI {
				kernelT[(kx*outChI+oc)*inChI+ic] = data[(ic*outChI+oc)*kSizeI+kx]
			}
		}
	}

	return kernelT
}

// ConvTranspose1DPrePacked is like ConvTranspose1D but accepts a pre-packed
// kernelT (from RepackConvTransposeKernel). Only valid for groups=1.
func ConvTranspose1DPrePacked(input, kernel, bias *tensor.Tensor, kernelT []float32, stride, padding, outputPadding, dilation, groups int64) (*tensor.Tensor, error) {
	if groups != 1 {
		return nil, fmt.Errorf("ops: ConvTranspose1DPrePacked requires groups=1, got %d", groups)
	}

	if kernelT == nil {
		return ConvTranspose1D(input, kernel, bias, stride, padding, outputPadding, dilation, groups)
	}
	// Validate shapes (same as ConvTranspose1D).
	if input == nil || kernel == nil {
		return nil, errors.New("ops: convtranspose1d requires non-nil input/kernel")
	}

	if stride <= 0 || dilation <= 0 {
		return nil, errors.New("ops: convtranspose1d stride/dilation must be > 0")
	}

	if outputPadding < 0 || outputPadding >= stride {
		return nil, fmt.Errorf("ops: convtranspose1d output_padding must be in [0, stride), got %d", outputPadding)
	}

	inShape := input.Shape()

	kShape := kernel.Shape()
	if len(inShape) != 3 || len(kShape) != 3 {
		return nil, fmt.Errorf("ops: convtranspose1d expects input/kernel rank 3, got %v and %v", inShape, kShape)
	}

	batch, inChannels, inLength := inShape[0], inShape[1], inShape[2]

	kInChannels, outPerGroup, kernelSize := kShape[0], kShape[1], kShape[2]
	if kInChannels != inChannels {
		return nil, fmt.Errorf("ops: convtranspose1d kernel in_channels mismatch %d vs %d", kInChannels, inChannels)
	}

	outChannels := outPerGroup // groups=1

	if bias != nil {
		bShape := bias.Shape()
		if len(bShape) != 1 || bShape[0] != outChannels {
			return nil, fmt.Errorf("ops: convtranspose1d bias shape %v does not match out_channels %d", bShape, outChannels)
		}
	}

	outLength := (inLength-1)*stride - 2*padding + dilation*(kernelSize-1) + outputPadding + 1
	if outLength <= 0 {
		return nil, fmt.Errorf("ops: convtranspose1d produced non-positive output length %d", outLength)
	}

	out, err := tensor.Zeros([]int64{batch, outChannels, outLength})
	if err != nil {
		return nil, err
	}

	var biasData []float32
	if bias != nil {
		biasData = bias.RawData()
	}

	convTranspose1DGroups1(input.RawData(), nil, biasData, kernelT,
		batch, inChannels, inLength, outChannels, kernelSize, outLength,
		stride, padding, dilation, out.RawData())

	return out, nil
}

// convTranspose1DGroups1 is the fast path for ConvTranspose1D with groups=1.
//
// If prePackedKernelT is non-nil, it is used directly (skipping the repack).
// Otherwise the kernel is repacked from kernelData on the fly.
//
// To enable AVX2 dot-product acceleration the function:
//  1. Uses kernelT in [kSize, outCh, inCh] layout so that kernelT[kx, oc, :]
//     is a contiguous float32 slice of length inCh.
//  2. Transposes the input from [batch, inCh, inLen] → inputT [batch, inLen, inCh]
//     so that inputT[b, ix, :] is contiguous.
//  3. For each (kx, ix) pair computes a GEMV (outCh dot-products of length inCh)
//     and scatters the result into the output column outPos = ix*stride + kx*dilation − padding.
//
// Bias (if non-nil) is added in a final vectorised pass.
func convTranspose1DGroups1(
	inputData, kernelData, biasData []float32,
	prePackedKernelT []float32,
	batch, inCh, inLen, outCh, kSize, outLen,
	stride, padding, dilation int64,
	outData []float32,
) {
	inChI := int(inCh)
	outChI := int(outCh)
	kSizeI := int(kSize)
	outLenI := int(outLen)
	inLenI := int(inLen)

	// Step 1: use pre-packed kernel or repack on the fly.
	kernelT := prePackedKernelT
	if kernelT == nil {
		kernelTSize := kSizeI * outChI * inChI

		kernelT = getScratch(kernelTSize)
		defer putScratch(kernelT)

		for ic := range inChI {
			for oc := range outChI {
				for kx := range kSizeI {
					kernelT[(kx*outChI+oc)*inChI+ic] = kernelData[(ic*outChI+oc)*kSizeI+kx]
				}
			}
		}
	}

	// Allocate inputT once and reuse across batch elements.
	inputTSize := inLenI * inChI

	inputT := getScratch(inputTSize)
	defer putScratch(inputT)

	for b := 0; b < int(batch); b++ {
		// Step 2: transpose input [inCh, inLen] → inputT [inLen, inCh].
		// Zero first (getScratch zeroes on first use, but re-zero for b > 0).
		if b > 0 {
			for i := range inputT {
				inputT[i] = 0
			}
		}

		for ic := range inChI {
			src := inputData[(b*inChI+ic)*inLenI : (b*inChI+ic+1)*inLenI]
			for ix, v := range src {
				inputT[ix*inChI+ic] = v
			}
		}

		// Step 3: GEMV scatter with oc as the outer dimension.
		// Each oc range writes only to outBatch[oc*outLenI : (oc+1)*outLenI],
		// so the loop is embarrassingly parallel across output channels.
		outBatch := outData[b*outChI*outLenI : (b+1)*outChI*outLenI]
		parallelFor(outChI, getConvWorkers(), func(ocLo, ocHi int) {
			for oc := ocLo; oc < ocHi; oc++ {
				outRow := outBatch[oc*outLenI : (oc+1)*outLenI]
				for kx := range kSizeI {
					kOff := (kx*outChI + oc) * inChI
					kernelTRow := kernelT[kOff : kOff+inChI]

					for ix := range inLenI {
						outPos := int64(ix)*stride - padding + int64(kx)*dilation
						if outPos < 0 || outPos >= outLen {
							continue
						}

						inputRow := inputT[ix*inChI : (ix+1)*inChI]
						outRow[outPos] += tensor.DotProduct(kernelTRow, inputRow)
					}
				}
				// Add bias for this channel.
				if biasData != nil {
					bv := biasData[oc]
					for i := range outRow {
						outRow[i] += bv
					}
				}
			}
		})
	}
}

// convTranspose1DFastDepthwise is the fast path for ConvTranspose1D when
// groups == inChannels (pure depthwise transposed convolution).
// Each channel is completely independent, so the inner loops collapse to a
// simple scatter-accumulate with a per-channel kernel of length kSize.
func convTranspose1DFastDepthwise(
	inputData, kernelData, biasData []float32,
	batch, inCh, inLen, outPerGroup, kSize, outLen,
	stride, padding, dilation int64,
	outData []float32,
) {
	outChI := int(inCh * outPerGroup)
	inLenI := int(inLen)
	outLenI := int(outLen)
	kSizeI := int(kSize)

	for b := range batch {
		for g := range inCh {
			ocBase := int(g * outPerGroup)

			inBase := int(b*inCh+g) * inLenI
			for ix := range inLen {
				inVal := inputData[inBase+int(ix)]
				if inVal == 0 {
					continue
				}

				for ocg := range outPerGroup {
					oc := int(g*outPerGroup + ocg)
					kBase := oc * kSizeI
					outSlice := outData[int(b)*outChI*outLenI+(ocBase+int(ocg))*outLenI:]

					for kx := range kSizeI {
						outPos := ix*stride - padding + int64(kx)*dilation
						if outPos >= 0 && outPos < outLen {
							outSlice[outPos] += inVal * kernelData[kBase+kx]
						}
					}
				}
			}
		}
		// Add bias after scatter.
		if biasData != nil {
			for oc := range outChI {
				outSlice := outData[int(b)*outChI*outLenI+oc*outLenI : int(b)*outChI*outLenI+(oc+1)*outLenI]

				bv := biasData[oc]
				for i := range outSlice {
					outSlice[i] += bv
				}
			}
		}
	}
}

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

	return out, nil
}

// RoPE applies rotary position embedding to the last dimension in interleaved
// pair format: (..., seq, dim) where dim must be even.
// cos/sin are expected as [max_seq, dim/2].
func RoPE(x, cos, sin *tensor.Tensor, pos int64) (*tensor.Tensor, error) {
	if x == nil || cos == nil || sin == nil {
		return nil, errors.New("ops: rope requires non-nil x/cos/sin")
	}

	if pos < 0 {
		return nil, errors.New("ops: rope position must be >= 0")
	}

	xShape := x.Shape()
	if len(xShape) < 2 {
		return nil, fmt.Errorf("ops: rope requires rank >= 2 input, got %d", len(xShape))
	}

	seq := xShape[len(xShape)-2]

	dim := xShape[len(xShape)-1]
	if dim%2 != 0 {
		return nil, fmt.Errorf("ops: rope last dimension must be even, got %d", dim)
	}

	half := dim / 2

	cosShape := cos.Shape()

	sinShape := sin.Shape()
	if len(cosShape) != 2 || len(sinShape) != 2 {
		return nil, fmt.Errorf("ops: rope cos/sin must be rank 2, got %v and %v", cosShape, sinShape)
	}

	if cosShape[0] < pos+seq || sinShape[0] < pos+seq {
		return nil, fmt.Errorf("ops: rope cos/sin sequence length too small for pos=%d seq=%d", pos, seq)
	}

	if cosShape[1] != half || sinShape[1] != half {
		return nil, fmt.Errorf("ops: rope cos/sin width mismatch, want %d got %d and %d", half, cosShape[1], sinShape[1])
	}

	out := x.Clone()
	outData := out.RawData()
	cosData := cos.RawData()
	sinData := sin.RawData()

	prefix := int64(len(outData)) / (seq * dim)
	seqI := int(seq)
	dimI := int(dim)
	halfI := int(half)

	for p := range prefix {
		prefixBase := int(p * seq * dim)

		for t := range seqI {
			trigBase := int((pos + int64(t)) * half)

			xBase := prefixBase + t*dimI
			for j := range halfI {
				i0 := xBase + 2*j
				i1 := i0 + 1
				a := outData[i0]
				b := outData[i1]
				c := cosData[trigBase+j]
				s := sinData[trigBase+j]
				outData[i0] = a*c - b*s
				outData[i1] = a*s + b*c
			}
		}
	}

	return out, nil
}

// Attention computes scaled dot-product attention.
// q shape: [..., tq, d], k shape: [..., tk, d], v shape: [..., tk, dv]
// output: [..., tq, dv]
func Attention(q, k, v *tensor.Tensor, causal bool, offset int64) (*tensor.Tensor, error) {
	if q == nil || k == nil || v == nil {
		return nil, errors.New("ops: attention requires non-nil q/k/v")
	}

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

	scaled := scores.Clone()

	scale := float32(1.0 / math.Sqrt(float64(d)))
	for i := range scaled.RawData() {
		scaled.RawData()[i] *= scale
	}

	if causal {
		scaled, err = CausalMask(scaled, offset)
		if err != nil {
			return nil, fmt.Errorf("ops: attention causal mask: %w", err)
		}
	}

	probs, err := tensor.Softmax(scaled, -1)
	if err != nil {
		return nil, fmt.Errorf("ops: attention softmax: %w", err)
	}

	out, err := tensor.MatMul(probs, v)
	if err != nil {
		return nil, fmt.Errorf("ops: attention probs*v: %w", err)
	}

	return out, nil
}

// MLP computes linear(silu(linear(x))).
func MLP(x, w1, b1, w2, b2 *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := tensor.Linear(x, w1, b1)
	if err != nil {
		return nil, fmt.Errorf("ops: mlp first linear: %w", err)
	}

	hAct := h.Clone()
	for i, v := range hAct.RawData() {
		hAct.RawData()[i] = silu(v)
	}

	out, err := tensor.Linear(hAct, w2, b2)
	if err != nil {
		return nil, fmt.Errorf("ops: mlp second linear: %w", err)
	}

	return out, nil
}

// Conv1D performs a deterministic CPU Conv1d.
// input: [batch, in_channels, length]
// kernel: [out_channels, in_channels/groups, kernel_size]
func Conv1D(input, kernel, bias *tensor.Tensor, stride, padding, dilation, groups int64) (*tensor.Tensor, error) {
	if input == nil || kernel == nil {
		return nil, errors.New("ops: conv1d requires non-nil input/kernel")
	}

	if stride <= 0 || dilation <= 0 || groups <= 0 {
		return nil, errors.New("ops: conv1d stride/dilation/groups must be > 0")
	}

	inShape := input.Shape()

	kShape := kernel.Shape()
	if len(inShape) != 3 || len(kShape) != 3 {
		return nil, fmt.Errorf("ops: conv1d expects input/kernel rank 3, got %v and %v", inShape, kShape)
	}

	batch, inChannels, length := inShape[0], inShape[1], inShape[2]
	outChannels, kInChannels, kernelSize := kShape[0], kShape[1], kShape[2]

	if inChannels%groups != 0 || outChannels%groups != 0 {
		return nil, fmt.Errorf("ops: conv1d channels not divisible by groups (%d, %d, groups=%d)", inChannels, outChannels, groups)
	}

	if kInChannels != inChannels/groups {
		return nil, fmt.Errorf("ops: conv1d kernel in_channels/groups mismatch: got %d want %d", kInChannels, inChannels/groups)
	}

	if bias != nil {
		bShape := bias.Shape()
		if len(bShape) != 1 || bShape[0] != outChannels {
			return nil, fmt.Errorf("ops: conv1d bias shape %v does not match out_channels %d", bShape, outChannels)
		}
	}

	outLength := (length+2*padding-dilation*(kernelSize-1)-1)/stride + 1
	if outLength <= 0 {
		return nil, fmt.Errorf("ops: conv1d produced non-positive output length %d", outLength)
	}

	out, err := tensor.Zeros([]int64{batch, outChannels, outLength})
	if err != nil {
		return nil, err
	}

	inputData := input.RawData()
	kernelData := kernel.RawData()
	outData := out.RawData()

	var biasData []float32
	if bias != nil {
		biasData = bias.RawData()
	}

	// Fast path: groups=1 uses im2col + AVX2 dot-product GEMM.
	if groups == 1 {
		conv1DFastGroups1(inputData, kernelData, biasData,
			batch, inChannels, length, outChannels, kernelSize, outLength,
			stride, padding, dilation, outData)

		return out, nil
	}

	inPerGroup := inChannels / groups

	outPerGroup := outChannels / groups
	for b := range batch {
		for oc := range outChannels {
			g := oc / outPerGroup
			inStart := g * inPerGroup

			for ox := range outLength {
				sum := float32(0)
				if biasData != nil {
					sum = biasData[oc]
				}

				for ic := range inPerGroup {
					inC := inStart + ic

					for kx := range kernelSize {
						inPos := ox*stride - padding + kx*dilation
						if inPos < 0 || inPos >= length {
							continue
						}

						inputIdx := ((b*inChannels + inC) * length) + inPos
						kernelIdx := ((oc*kInChannels + ic) * kernelSize) + kx
						sum += inputData[inputIdx] * kernelData[kernelIdx]
					}
				}

				outIdx := ((b*outChannels + oc) * outLength) + ox
				outData[outIdx] = sum
			}
		}
	}

	return out, nil
}

// ConvTranspose1D performs a deterministic CPU ConvTranspose1d.
// input: [batch, in_channels, length]
// kernel: [in_channels, out_channels/groups, kernel_size]
func ConvTranspose1D(input, kernel, bias *tensor.Tensor, stride, padding, outputPadding, dilation, groups int64) (*tensor.Tensor, error) {
	if input == nil || kernel == nil {
		return nil, errors.New("ops: convtranspose1d requires non-nil input/kernel")
	}

	if stride <= 0 || dilation <= 0 || groups <= 0 {
		return nil, errors.New("ops: convtranspose1d stride/dilation/groups must be > 0")
	}

	if outputPadding < 0 || outputPadding >= stride {
		return nil, fmt.Errorf("ops: convtranspose1d output_padding must be in [0, stride), got %d", outputPadding)
	}

	inShape := input.Shape()

	kShape := kernel.Shape()
	if len(inShape) != 3 || len(kShape) != 3 {
		return nil, fmt.Errorf("ops: convtranspose1d expects input/kernel rank 3, got %v and %v", inShape, kShape)
	}

	batch, inChannels, inLength := inShape[0], inShape[1], inShape[2]

	kInChannels, outPerGroup, kernelSize := kShape[0], kShape[1], kShape[2]
	if kInChannels != inChannels {
		return nil, fmt.Errorf("ops: convtranspose1d kernel in_channels mismatch %d vs %d", kInChannels, inChannels)
	}

	if inChannels%groups != 0 {
		return nil, fmt.Errorf("ops: convtranspose1d in_channels %d must be divisible by groups %d", inChannels, groups)
	}

	outChannels := outPerGroup * groups

	if bias != nil {
		bShape := bias.Shape()
		if len(bShape) != 1 || bShape[0] != outChannels {
			return nil, fmt.Errorf("ops: convtranspose1d bias shape %v does not match out_channels %d", bShape, outChannels)
		}
	}

	outLength := (inLength-1)*stride - 2*padding + dilation*(kernelSize-1) + outputPadding + 1
	if outLength <= 0 {
		return nil, fmt.Errorf("ops: convtranspose1d produced non-positive output length %d", outLength)
	}

	out, err := tensor.Zeros([]int64{batch, outChannels, outLength})
	if err != nil {
		return nil, err
	}

	inPerGroup := inChannels / groups
	inputData := input.RawData()
	kernelData := kernel.RawData()
	outData := out.RawData()

	var biasData []float32
	if bias != nil {
		biasData = bias.RawData()
	}

	// Fast path: groups=1 uses kernel-repack + GEMV scatter with AVX2 dot-products.
	if groups == 1 {
		convTranspose1DGroups1(inputData, kernelData, biasData, nil,
			batch, inChannels, inLength, outChannels, kernelSize, outLength,
			stride, padding, dilation, outData)

		return out, nil
	}

	// Fast path: depthwise (groups == inChannels) avoids redundant group arithmetic.
	if groups == inChannels {
		convTranspose1DFastDepthwise(inputData, kernelData, biasData,
			batch, inChannels, inLength, outPerGroup, kernelSize, outLength,
			stride, padding, dilation, outData)

		return out, nil
	}

	for b := range batch {
		for ic := range inChannels {
			g := ic / inPerGroup
			ocBase := g * outPerGroup

			for ix := range inLength {
				inVal := inputData[((b*inChannels+ic)*inLength)+ix]

				for ocg := range outPerGroup {
					oc := ocBase + ocg

					for kx := range kernelSize {
						outPos := ix*stride - padding + kx*dilation
						if outPos < 0 || outPos >= outLength {
							continue
						}

						kernelIdx := ((ic*outPerGroup + ocg) * kernelSize) + kx
						outIdx := ((b*outChannels + oc) * outLength) + outPos
						outData[outIdx] += inVal * kernelData[kernelIdx]
					}
				}
			}
		}
	}

	if biasData != nil {
		for b := range batch {
			for oc := range outChannels {
				base := ((b*outChannels + oc) * outLength)
				for ox := range outLength {
					outData[base+ox] += biasData[oc]
				}
			}
		}
	}

	return out, nil
}

func silu(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}
