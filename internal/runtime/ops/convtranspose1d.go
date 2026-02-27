package ops

import (
	"errors"
	"fmt"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

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

	p, out, biasData, err := prepareConvTranspose1D(input, kernel, bias, stride, padding, outputPadding, dilation, groups)
	if err != nil {
		return nil, err
	}

	inShape := input.Shape()
	kShape := kernel.Shape()

	expected := int(inShape[1] * kShape[1] * kShape[2])
	if len(kernelT) != expected {
		return nil, fmt.Errorf("ops: prepacked kernel length mismatch: got %d want %d", len(kernelT), expected)
	}

	convTranspose1DGroups1(input.RawData(), nil, biasData, kernelT,
		p.batch, p.inChannels, p.inLength, p.outChannels, p.kernelSize, p.outLength,
		stride, padding, dilation, out.RawData())

	return out, nil
}

func convTranspose1DGroups1(
	inputData, kernelData, biasData, kernelT []float32,
	batch, inCh, inLen, outCh, kSize, outLen,
	stride, padding, dilation int64,
	outData []float32,
) {
	inChI := int(inCh)
	inLenI := int(inLen)
	outChI := int(outCh)
	outLenI := int(outLen)
	kSizeI := int(kSize)

	if kernelT == nil {
		kernelT = getScratch(kSizeI * outChI * inChI)
		defer putScratch(kernelT)

		for ic := range inChI {
			for oc := range outChI {
				for kx := range kSizeI {
					kernelT[(kx*outChI+oc)*inChI+ic] = kernelData[(ic*outChI+oc)*kSizeI+kx]
				}
			}
		}
	}

	inputT := getScratch(inLenI * inChI)
	defer putScratch(inputT)

	for b := range batch {
		bI := int(b)

		if b > 0 {
			for i := range inputT {
				inputT[i] = 0
			}
		}

		for ic := range inChI {
			base := (bI*inChI + ic) * inLenI

			src := inputData[base : base+inLenI]
			for ix, v := range src {
				inputT[ix*inChI+ic] = v
			}
		}

		outBase := bI * outChI * outLenI
		outBatch := outData[outBase : outBase+outChI*outLenI]
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

// ConvTranspose1D performs a deterministic CPU ConvTranspose1d.
// input: [batch, in_channels, length]
// kernel: [in_channels, out_channels/groups, kernel_size]
func ConvTranspose1D(input, kernel, bias *tensor.Tensor, stride, padding, outputPadding, dilation, groups int64) (*tensor.Tensor, error) {
	p, out, biasData, err := prepareConvTranspose1D(input, kernel, bias, stride, padding, outputPadding, dilation, groups)
	if err != nil {
		return nil, err
	}

	inputData := input.RawData()
	kernelData := kernel.RawData()
	outData := out.RawData()

	if groups == 1 {
		convTranspose1DGroups1(inputData, kernelData, biasData, nil,
			p.batch, p.inChannels, p.inLength, p.outChannels, p.kernelSize, p.outLength,
			stride, padding, dilation, outData)

		return out, nil
	}

	if groups == p.inChannels {
		convTranspose1DFastDepthwise(inputData, kernelData, biasData,
			p.batch, p.inChannels, p.inLength, p.outPerGroup, p.kernelSize, p.outLength,
			stride, padding, dilation, outData)

		return out, nil
	}

	convTranspose1DGrouped(inputData, kernelData, outData,
		p.batch, p.inChannels, p.inLength, p.outChannels, p.kernelSize, p.outLength,
		p.inPerGroup, p.outPerGroup, stride, padding, dilation)

	addConvTransposeBias(outData, biasData, p.batch, p.outChannels, p.outLength)

	return out, nil
}

type convTranspose1DParams struct {
	batch       int64
	inChannels  int64
	inLength    int64
	outChannels int64
	outPerGroup int64
	inPerGroup  int64
	kernelSize  int64
	outLength   int64
}

func prepareConvTranspose1D(
	input, kernel, bias *tensor.Tensor,
	stride, padding, outputPadding, dilation, groups int64,
) (convTranspose1DParams, *tensor.Tensor, []float32, error) {
	if input == nil || kernel == nil {
		return convTranspose1DParams{}, nil, nil, errors.New("ops: convtranspose1d requires non-nil input/kernel")
	}

	if stride <= 0 || dilation <= 0 || groups <= 0 {
		return convTranspose1DParams{}, nil, nil, errors.New("ops: convtranspose1d stride/dilation/groups must be > 0")
	}

	if outputPadding < 0 || outputPadding >= stride {
		return convTranspose1DParams{}, nil, nil, fmt.Errorf("ops: convtranspose1d output_padding must be in [0, stride), got %d", outputPadding)
	}

	inShape := input.Shape()

	kShape := kernel.Shape()
	if len(inShape) != 3 || len(kShape) != 3 {
		return convTranspose1DParams{}, nil, nil, fmt.Errorf("ops: convtranspose1d expects input/kernel rank 3, got %v and %v", inShape, kShape)
	}

	p := convTranspose1DParams{
		batch:       inShape[0],
		inChannels:  inShape[1],
		inLength:    inShape[2],
		outPerGroup: kShape[1],
		kernelSize:  kShape[2],
	}

	kInChannels := kShape[0]
	if kInChannels != p.inChannels {
		return convTranspose1DParams{}, nil, nil, fmt.Errorf("ops: convtranspose1d kernel in_channels mismatch %d vs %d", kInChannels, p.inChannels)
	}

	if p.inChannels%groups != 0 {
		return convTranspose1DParams{}, nil, nil, fmt.Errorf("ops: convtranspose1d in_channels %d must be divisible by groups %d", p.inChannels, groups)
	}

	p.outChannels = p.outPerGroup * groups
	p.inPerGroup = p.inChannels / groups

	if bias != nil {
		bShape := bias.Shape()
		if len(bShape) != 1 || bShape[0] != p.outChannels {
			return convTranspose1DParams{}, nil, nil, fmt.Errorf("ops: convtranspose1d bias shape %v does not match out_channels %d", bShape, p.outChannels)
		}
	}

	p.outLength = (p.inLength-1)*stride - 2*padding + dilation*(p.kernelSize-1) + outputPadding + 1
	if p.outLength <= 0 {
		return convTranspose1DParams{}, nil, nil, fmt.Errorf("ops: convtranspose1d produced non-positive output length %d", p.outLength)
	}

	out, err := tensor.Zeros([]int64{p.batch, p.outChannels, p.outLength})
	if err != nil {
		return convTranspose1DParams{}, nil, nil, err
	}

	var biasData []float32
	if bias != nil {
		biasData = bias.RawData()
	}

	return p, out, biasData, nil
}

func convTranspose1DGrouped(
	inputData, kernelData, outData []float32,
	batch, inChannels, inLength, outChannels, kernelSize, outLength, inPerGroup, outPerGroup, stride, padding, dilation int64,
) {
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
}

func addConvTransposeBias(outData, biasData []float32, batch, outChannels, outLength int64) {
	if biasData == nil {
		return
	}

	for b := range batch {
		for oc := range outChannels {
			base := ((b*outChannels + oc) * outLength)
			for ox := range outLength {
				outData[base+ox] += biasData[oc]
			}
		}
	}
}
