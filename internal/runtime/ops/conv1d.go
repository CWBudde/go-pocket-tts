package ops

import (
	"errors"
	"fmt"

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

		// GEMM: kernel [outCh, patchLen] x imcol^T [patchLen, outLen] -> out [outCh, outLen].
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

// Conv1D performs a deterministic CPU Conv1d.
// input: [batch, in_channels, length]
// kernel: [out_channels, in_channels/groups, kernel_size]
func Conv1D(input, kernel, bias *tensor.Tensor, stride, padding, dilation, groups int64) (*tensor.Tensor, error) {
	p, out, biasData, err := prepareConv1D(input, kernel, bias, stride, padding, dilation, groups)
	if err != nil {
		return nil, err
	}

	inputData := input.RawData()
	kernelData := kernel.RawData()
	outData := out.RawData()

	if groups == 1 {
		conv1DFastGroups1(inputData, kernelData, biasData,
			p.batch, p.inChannels, p.length, p.outChannels, p.kernelSize, p.outLength,
			stride, padding, dilation, outData)

		return out, nil
	}

	conv1DGrouped(inputData, kernelData, biasData, outData,
		p.batch, p.inChannels, p.length, p.outChannels, p.kernelSize, p.outLength,
		p.kInChannels, p.inPerGroup, p.outPerGroup, stride, padding, dilation)

	return out, nil
}

type conv1DParams struct {
	batch       int64
	inChannels  int64
	length      int64
	outChannels int64
	kInChannels int64
	kernelSize  int64
	outLength   int64
	inPerGroup  int64
	outPerGroup int64
}

func prepareConv1D(
	input, kernel, bias *tensor.Tensor,
	stride, padding, dilation, groups int64,
) (conv1DParams, *tensor.Tensor, []float32, error) {
	if input == nil || kernel == nil {
		return conv1DParams{}, nil, nil, errors.New("ops: conv1d requires non-nil input/kernel")
	}

	if stride <= 0 || dilation <= 0 || groups <= 0 {
		return conv1DParams{}, nil, nil, errors.New("ops: conv1d stride/dilation/groups must be > 0")
	}

	inShape := input.Shape()
	kShape := kernel.Shape()

	if len(inShape) != 3 || len(kShape) != 3 {
		return conv1DParams{}, nil, nil, fmt.Errorf("ops: conv1d expects input/kernel rank 3, got %v and %v", inShape, kShape)
	}

	p := conv1DParams{
		batch:       inShape[0],
		inChannels:  inShape[1],
		length:      inShape[2],
		outChannels: kShape[0],
		kInChannels: kShape[1],
		kernelSize:  kShape[2],
	}

	if p.inChannels%groups != 0 || p.outChannels%groups != 0 {
		return conv1DParams{}, nil, nil, fmt.Errorf("ops: conv1d channels not divisible by groups (%d, %d, groups=%d)", p.inChannels, p.outChannels, groups)
	}

	if p.kInChannels != p.inChannels/groups {
		return conv1DParams{}, nil, nil, fmt.Errorf("ops: conv1d kernel in_channels/groups mismatch: got %d want %d", p.kInChannels, p.inChannels/groups)
	}

	p.inPerGroup = p.inChannels / groups
	p.outPerGroup = p.outChannels / groups

	if bias != nil {
		bShape := bias.Shape()
		if len(bShape) != 1 || bShape[0] != p.outChannels {
			return conv1DParams{}, nil, nil, fmt.Errorf("ops: conv1d bias shape %v does not match out_channels %d", bShape, p.outChannels)
		}
	}

	p.outLength = (p.length+2*padding-dilation*(p.kernelSize-1)-1)/stride + 1
	if p.outLength <= 0 {
		return conv1DParams{}, nil, nil, fmt.Errorf("ops: conv1d produced non-positive output length %d", p.outLength)
	}

	out, err := tensor.Zeros([]int64{p.batch, p.outChannels, p.outLength})
	if err != nil {
		return conv1DParams{}, nil, nil, err
	}

	var biasData []float32
	if bias != nil {
		biasData = bias.RawData()
	}

	return p, out, biasData, nil
}

func conv1DGrouped(
	inputData, kernelData, biasData, outData []float32,
	batch, inChannels, length, outChannels, kernelSize, outLength, kInChannels, inPerGroup, outPerGroup, stride, padding, dilation int64,
) {
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
}
