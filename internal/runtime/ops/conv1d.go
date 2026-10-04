package ops

import (
	"errors"
	"fmt"

	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
)

// convTileFloats bounds one worker's im2col tile (128 KiB) so the GEMM over it
// stays in cache. A full im2col of a SEANet residual conv over a 30 s prompt
// at 24 kHz would be about 550 MB.
const convTileFloats = 1 << 15

// conv1DFastGroups1 is the im2col fast path for Conv1D with groups=1.
//
// It rearranges the convolution into a GEMM by building a patch matrix
// (im2col) of shape [outLength, inChannels*kernelSize] where each row contains
// the gathered input values for one output position.  The GEMM then becomes:
//
//	out[oc, ox] = dotProduct(kernel[oc, :], imcol[ox, :]) + bias[oc]
//
// Both the kernel row and the im2col row are contiguous in memory. Short
// outputs (the streaming decoder) build one im2col and split the output
// channels across workers; long ones are cut into tiles of output positions,
// each built and multiplied by one worker.
func conv1DFastGroups1(
	inputData, kernelData, biasData []float32,
	batch, inCh, length, outCh, kSize, outLen,
	stride, leftPadding, dilation int64,
	outData []float32,
) {
	// An empty patch (no input channels) still yields the bias everywhere.
	tileRows := max(1, convTileFloats/max(1, int(inCh*kSize)))
	if int(outLen) <= tileRows {
		conv1DIm2colFull(inputData, kernelData, biasData, batch, inCh, length, outCh, kSize, outLen,
			stride, leftPadding, dilation, outData)

		return
	}

	conv1DIm2colTiled(inputData, kernelData, biasData, batch, inCh, length, outCh, kSize, outLen,
		stride, leftPadding, dilation, outData, tileRows)
}

func conv1DIm2colFull(
	inputData, kernelData, biasData []float32,
	batch, inCh, length, outCh, kSize, outLen,
	stride, leftPadding, dilation int64,
	outData []float32,
) {
	patchLen := int(inCh * kSize)
	outChI := int(outCh)
	outLenI := int(outLen)
	inBatch := int(inCh * length)

	imcol := getScratch(int(outLen) * patchLen) // [outLen, inCh*kSize]
	defer putScratch(imcol)

	for b := range int(batch) {
		fillIm2colRows(imcol, inputData[b*inBatch:(b+1)*inBatch], inCh, length, kSize,
			stride, leftPadding, dilation, 0, outLenI)

		// GEMM: kernel [outCh, patchLen] x imcol^T [patchLen, outLen] -> out [outCh, outLen].
		// The oc loop is embarrassingly parallel: each output channel writes to
		// a disjoint slice of outData and reads shared (immutable) imcol + kernel.
		outB := outData[b*outChI*outLenI : (b+1)*outChI*outLenI]
		parallelFor(outChI, getConvWorkers(), func(ocLo, ocHi int) {
			im2colGEMM(imcol, kernelData, biasData, patchLen, ocLo, ocHi, outB, outLenI, 0, outLenI)
		})
	}
}

func conv1DIm2colTiled(
	inputData, kernelData, biasData []float32,
	batch, inCh, length, outCh, kSize, outLen,
	stride, leftPadding, dilation int64,
	outData []float32, tileRows int,
) {
	patchLen := int(inCh * kSize)
	outChI := int(outCh)
	outLenI := int(outLen)
	inBatch := int(inCh * length)
	tiles := (outLenI + tileRows - 1) / tileRows

	for b := range int(batch) {
		in := inputData[b*inBatch : (b+1)*inBatch]
		outB := outData[b*outChI*outLenI : (b+1)*outChI*outLenI]

		parallelFor(tiles, getConvWorkers(), func(tLo, tHi int) {
			imcol := getScratch(tileRows * patchLen)
			defer putScratch(imcol)

			for tile := tLo; tile < tHi; tile++ {
				ox0 := tile * tileRows
				rows := min(tileRows, outLenI-ox0)
				fillIm2colRows(imcol, in, inCh, length, kSize, stride, leftPadding, dilation, ox0, rows)
				im2colGEMM(imcol, kernelData, biasData, patchLen, 0, outChI, outB, outLenI, ox0, rows)
			}
		})
	}
}

// fillIm2colRows writes the im2col rows of output positions ox0..ox0+rows-1
// of one batch item, one [inCh*kSize] patch per row; taps outside the input
// are 0.
func fillIm2colRows(
	imcol, in []float32,
	inCh, length, kSize, stride, leftPadding, dilation int64,
	ox0, rows int,
) {
	kSizeI := int(kSize)
	lenI := int(length)
	patchLen := int(inCh) * kSizeI

	for r := range rows {
		start := int64(ox0+r)*stride - leftPadding
		inside := start >= 0 && start+(kSize-1)*dilation < length
		row := imcol[r*patchLen : (r+1)*patchLen]

		for ic := range int(inCh) {
			dst := row[ic*kSizeI : (ic+1)*kSizeI]
			src := in[ic*lenI : (ic+1)*lenI]

			if inside && dilation == 1 {
				copy(dst, src[start:start+kSize])
				continue
			}

			for kx := range dst {
				pos := start + int64(kx)*dilation
				if pos >= 0 && pos < length {
					dst[kx] = src[pos]
				} else {
					dst[kx] = 0
				}
			}
		}
	}
}

// im2colGEMM computes output channels ocLo..ocHi-1 at positions
// ox0..ox0+rows-1 from imcol rows 0..rows-1: the kernel rows times the
// transposed patch rows land straight in the output (row stride outLen),
// then each channel gets its bias.
func im2colGEMM(
	imcol, kernelData, biasData []float32,
	patchLen, ocLo, ocHi int,
	out []float32, outLen, ox0, rows int,
) {
	m := ocHi - ocLo
	kRows := kernelData[ocLo*patchLen : ocHi*patchLen]
	patches := imcol[:rows*patchLen]

	if rows < 4 && m >= 4 {
		// Too few positions (streaming decode) for MatMulTransB's 4-wide
		// blocks: let the channels take that role and transpose the result.
		prod := getScratch(rows * m)
		tensor.MatMulTransB(prod, m, patches, kRows, rows, m, patchLen)

		for r := range rows {
			for i, v := range prod[r*m : (r+1)*m] {
				out[(ocLo+i)*outLen+ox0+r] = v
			}
		}

		putScratch(prod)
	} else {
		tensor.MatMulTransB(out[ocLo*outLen+ox0:], outLen, kRows, patches, m, rows, patchLen)
	}

	if biasData == nil {
		return
	}

	for oc := ocLo; oc < ocHi; oc++ {
		bv := biasData[oc]

		outRow := out[oc*outLen+ox0 : oc*outLen+ox0+rows]
		for r := range outRow {
			outRow[r] += bv
		}
	}
}

// Conv1D performs a deterministic CPU Conv1d.
// input: [batch, in_channels, length]
// kernel: [out_channels, in_channels/groups, kernel_size]
func Conv1D(input, kernel, bias *tensor.Tensor, stride, padding, dilation, groups int64) (*tensor.Tensor, error) {
	return conv1DWithAsymmetricPadding(input, kernel, bias, stride, padding, padding, dilation, groups)
}

// Conv1DLeftPad is Conv1D with left-only zero padding and no right padding.
// This is useful for streaming decode paths where history padding is required
// without extending the right boundary.
func Conv1DLeftPad(input, kernel, bias *tensor.Tensor, stride, leftPadding, dilation, groups int64) (*tensor.Tensor, error) {
	return conv1DWithAsymmetricPadding(input, kernel, bias, stride, leftPadding, 0, dilation, groups)
}

func conv1DWithAsymmetricPadding(
	input, kernel, bias *tensor.Tensor,
	stride, leftPadding, rightPadding, dilation, groups int64,
) (*tensor.Tensor, error) {
	p, out, biasData, err := prepareConv1D(input, kernel, bias, stride, leftPadding, rightPadding, dilation, groups)
	if err != nil {
		return nil, err
	}

	inputData := input.RawData()
	kernelData := kernel.RawData()
	outData := out.RawData()

	if groups == 1 {
		conv1DFastGroups1(inputData, kernelData, biasData,
			p.batch, p.inChannels, p.length, p.outChannels, p.kernelSize, p.outLength,
			stride, leftPadding, dilation, outData)

		return out, nil
	}

	conv1DGrouped(inputData, kernelData, biasData, outData,
		p.batch, p.inChannels, p.length, p.outChannels, p.kernelSize, p.outLength,
		p.kInChannels, p.inPerGroup, p.outPerGroup, stride, leftPadding, dilation)

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
	stride, leftPadding, rightPadding, dilation, groups int64,
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

	p.outLength = (p.length+leftPadding+rightPadding-dilation*(p.kernelSize-1)-1)/stride + 1
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
	batch, inChannels, length, outChannels, kernelSize, outLength, kInChannels, inPerGroup, outPerGroup, stride, leftPadding, dilation int64,
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
						inPos := ox*stride - leftPadding + kx*dilation
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
