package ops

import (
	"fmt"
	"math"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

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
	imcol := make([]float32, int(outLen)*patchLen) // [outLen, inCh*kSize]

	kSizeI := int(kSize)
	outChI := int(outCh)
	outLenI := int(outLen)
	lenI := int(length)

	for b := int64(0); b < batch; b++ {
		// Zero im2col (ensures padding positions stay 0).
		for i := range imcol {
			imcol[i] = 0
		}

		// Build im2col: for each (ic, kx) column, copy valid input positions.
		// Iterating (ic, kx) in outer loops and ox in inner loop keeps the
		// writes to imcol sequential (stride = patchLen across rows, consecutive
		// columns within a row).
		for ic := int64(0); ic < inCh; ic++ {
			inBase := int(b*inCh+ic) * lenI
			for kx := int64(0); kx < kSize; kx++ {
				col := int(ic)*kSizeI + int(kx)
				for ox := int64(0); ox < outLen; ox++ {
					inPos := ox*stride - padding + kx*dilation
					if inPos >= 0 && inPos < length {
						imcol[int(ox)*patchLen+col] = inputData[inBase+int(inPos)]
					}
				}
			}
		}

		// GEMM: kernel [outCh, patchLen] × imcol^T [patchLen, outLen] → out [outCh, outLen].
		outBase := int(b) * outChI * outLenI
		for oc := 0; oc < outChI; oc++ {
			kernelRow := kernelData[oc*patchLen : (oc+1)*patchLen]
			biasVal := float32(0)
			if biasData != nil {
				biasVal = biasData[oc]
			}
			outOC := outData[outBase+oc*outLenI : outBase+(oc+1)*outLenI]
			for ox := 0; ox < outLenI; ox++ {
				outOC[ox] = tensor.DotProduct(kernelRow, imcol[ox*patchLen:(ox+1)*patchLen]) + biasVal
			}
		}
	}
}

// convTranspose1DGroups1 is the fast path for ConvTranspose1D with groups=1.
//
// To enable AVX2 dot-product acceleration the function:
//  1. Repacks the kernel from [inCh, outCh, kSize] → kernelT [kSize, outCh, inCh]
//     so that kernelT[kx, oc, :] is a contiguous float32 slice of length inCh.
//  2. Transposes the input from [batch, inCh, inLen] → inputT [batch, inLen, inCh]
//     so that inputT[b, ix, :] is contiguous.
//  3. For each (kx, ix) pair computes a GEMV (outCh dot-products of length inCh)
//     and scatters the result into the output column outPos = ix*stride + kx*dilation − padding.
//
// Bias (if non-nil) is added in a final vectorised pass.
func convTranspose1DGroups1(
	inputData, kernelData, biasData []float32,
	batch, inCh, inLen, outCh, kSize, outLen,
	stride, padding, dilation int64,
	outData []float32,
) {
	inChI := int(inCh)
	outChI := int(outCh)
	kSizeI := int(kSize)
	outLenI := int(outLen)
	inLenI := int(inLen)

	// Step 1: repack kernel [inCh, outCh, kSize] → kernelT [kSize, outCh, inCh].
	kernelT := make([]float32, kSizeI*outChI*inChI)
	for ic := 0; ic < inChI; ic++ {
		for oc := 0; oc < outChI; oc++ {
			for kx := 0; kx < kSizeI; kx++ {
				kernelT[(kx*outChI+oc)*inChI+ic] = kernelData[(ic*outChI+oc)*kSizeI+kx]
			}
		}
	}

	temp := make([]float32, outChI)

	for b := 0; b < int(batch); b++ {
		// Step 2: transpose input [inCh, inLen] → inputT [inLen, inCh].
		inputT := make([]float32, inLenI*inChI)
		for ic := 0; ic < inChI; ic++ {
			src := inputData[(b*inChI+ic)*inLenI : (b*inChI+ic+1)*inLenI]
			for ix, v := range src {
				inputT[ix*inChI+ic] = v
			}
		}

		// Step 3: GEMV scatter — for each valid (kx, ix) pair:
		//   temp[oc] = dot(kernelT[kx, oc, :], inputT[ix, :])
		//   out[oc, outPos] += temp[oc]
		outBatch := outData[b*outChI*outLenI : (b+1)*outChI*outLenI]
		for kx := 0; kx < kSizeI; kx++ {
			kxBase := kx * outChI * inChI
			for ix := 0; ix < inLenI; ix++ {
				outPos := int64(ix)*stride - padding + int64(kx)*dilation
				if outPos < 0 || outPos >= outLen {
					continue
				}
				inputRow := inputT[ix*inChI : (ix+1)*inChI]
				for oc := 0; oc < outChI; oc++ {
					temp[oc] = tensor.DotProduct(kernelT[kxBase+oc*inChI:kxBase+(oc+1)*inChI], inputRow)
				}
				// Scatter accumulated values into the output column.
				iOutPos := int(outPos)
				for oc := 0; oc < outChI; oc++ {
					outBatch[oc*outLenI+iOutPos] += temp[oc]
				}
			}
		}

		// Add bias (final vectorised pass over each output channel row).
		if biasData != nil {
			for oc := 0; oc < outChI; oc++ {
				bv := biasData[oc]
				row := outBatch[oc*outLenI : (oc+1)*outLenI]
				for i := range row {
					row[i] += bv
				}
			}
		}
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

	for b := int64(0); b < batch; b++ {
		for g := int64(0); g < inCh; g++ {
			ocBase := int(g * outPerGroup)
			inBase := int(b*inCh+g) * inLenI
			for ix := int64(0); ix < inLen; ix++ {
				inVal := inputData[inBase+int(ix)]
				if inVal == 0 {
					continue
				}
				for ocg := int64(0); ocg < outPerGroup; ocg++ {
					oc := int(g*outPerGroup + ocg)
					kBase := oc * kSizeI
					outSlice := outData[int(b)*outChI*outLenI+(ocBase+int(ocg))*outLenI:]
					for kx := 0; kx < kSizeI; kx++ {
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
			for oc := 0; oc < outChI; oc++ {
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
		return nil, fmt.Errorf("ops: causal mask scores is nil")
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
	for b := 0; b < blocks; b++ {
		base := b * q * k
		for qi := 0; qi < q; qi++ {
			maxKey := int64(qi) + offset
			row := base + qi*k
			for ki := 0; ki < k; ki++ {
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
		return nil, fmt.Errorf("ops: rope requires non-nil x/cos/sin")
	}
	if pos < 0 {
		return nil, fmt.Errorf("ops: rope position must be >= 0")
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
	for p := int64(0); p < prefix; p++ {
		prefixBase := int(p * seq * dim)
		for t := 0; t < seqI; t++ {
			trigBase := int((pos + int64(t)) * half)
			xBase := prefixBase + t*dimI
			for j := 0; j < halfI; j++ {
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
		return nil, fmt.Errorf("ops: attention requires non-nil q/k/v")
	}
	qShape := q.Shape()
	kShape := k.Shape()
	vShape := v.Shape()
	if len(qShape) < 2 || len(kShape) < 2 || len(vShape) < 2 {
		return nil, fmt.Errorf("ops: attention requires rank >= 2 inputs")
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
		return nil, fmt.Errorf("ops: conv1d requires non-nil input/kernel")
	}
	if stride <= 0 || dilation <= 0 || groups <= 0 {
		return nil, fmt.Errorf("ops: conv1d stride/dilation/groups must be > 0")
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
	for b := int64(0); b < batch; b++ {
		for oc := int64(0); oc < outChannels; oc++ {
			g := oc / outPerGroup
			inStart := g * inPerGroup
			for ox := int64(0); ox < outLength; ox++ {
				sum := float32(0)
				if biasData != nil {
					sum = biasData[oc]
				}
				for ic := int64(0); ic < inPerGroup; ic++ {
					inC := inStart + ic
					for kx := int64(0); kx < kernelSize; kx++ {
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
		return nil, fmt.Errorf("ops: convtranspose1d requires non-nil input/kernel")
	}
	if stride <= 0 || dilation <= 0 || groups <= 0 {
		return nil, fmt.Errorf("ops: convtranspose1d stride/dilation/groups must be > 0")
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
		convTranspose1DGroups1(inputData, kernelData, biasData,
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

	for b := int64(0); b < batch; b++ {
		for ic := int64(0); ic < inChannels; ic++ {
			g := ic / inPerGroup
			ocBase := g * outPerGroup
			for ix := int64(0); ix < inLength; ix++ {
				inVal := inputData[((b*inChannels+ic)*inLength)+ix]
				for ocg := int64(0); ocg < outPerGroup; ocg++ {
					oc := ocBase + ocg
					for kx := int64(0); kx < kernelSize; kx++ {
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
		for b := int64(0); b < batch; b++ {
			for oc := int64(0); oc < outChannels; oc++ {
				base := ((b*outChannels + oc) * outLength)
				for ox := int64(0); ox < outLength; ox++ {
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
