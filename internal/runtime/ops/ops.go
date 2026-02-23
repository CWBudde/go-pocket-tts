package ops

import (
	"fmt"
	"math"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

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
