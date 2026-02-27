package ops

import (
	"errors"
	"fmt"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

// RoPE applies rotary position embedding to the last dimension in interleaved
// pair format: (..., seq, dim) where dim must be even.
// cos/sin are expected as [max_seq, dim/2].
func RoPE(x, cos, sin *tensor.Tensor, pos int64) (*tensor.Tensor, error) {
	p, out, outData, cosData, sinData, err := prepareRoPE(x, cos, sin, pos)
	if err != nil {
		return nil, err
	}

	applyRoPE(outData, cosData, sinData, p)

	return out, nil
}

type ropeParams struct {
	seq    int64
	dim    int64
	half   int64
	pos    int64
	prefix int64
}

func prepareRoPE(x, cos, sin *tensor.Tensor, pos int64) (ropeParams, *tensor.Tensor, []float32, []float32, []float32, error) {
	if x == nil || cos == nil || sin == nil {
		return ropeParams{}, nil, nil, nil, nil, errors.New("ops: rope requires non-nil x/cos/sin")
	}

	if pos < 0 {
		return ropeParams{}, nil, nil, nil, nil, errors.New("ops: rope position must be >= 0")
	}

	xShape := x.Shape()
	if len(xShape) < 2 {
		return ropeParams{}, nil, nil, nil, nil, fmt.Errorf("ops: rope requires rank >= 2 input, got %d", len(xShape))
	}

	p := ropeParams{
		seq: xShape[len(xShape)-2],
		dim: xShape[len(xShape)-1],
		pos: pos,
	}
	if p.dim%2 != 0 {
		return ropeParams{}, nil, nil, nil, nil, fmt.Errorf("ops: rope last dimension must be even, got %d", p.dim)
	}

	p.half = p.dim / 2

	cosShape := cos.Shape()
	sinShape := sin.Shape()

	if len(cosShape) != 2 || len(sinShape) != 2 {
		return ropeParams{}, nil, nil, nil, nil, fmt.Errorf("ops: rope cos/sin must be rank 2, got %v and %v", cosShape, sinShape)
	}

	if cosShape[0] < p.pos+p.seq || sinShape[0] < p.pos+p.seq {
		return ropeParams{}, nil, nil, nil, nil, fmt.Errorf("ops: rope cos/sin sequence length too small for pos=%d seq=%d", p.pos, p.seq)
	}

	if cosShape[1] != p.half || sinShape[1] != p.half {
		return ropeParams{}, nil, nil, nil, nil, fmt.Errorf("ops: rope cos/sin width mismatch, want %d got %d and %d", p.half, cosShape[1], sinShape[1])
	}

	out := x.Clone()
	outData := out.RawData()
	cosData := cos.RawData()
	sinData := sin.RawData()
	p.prefix = int64(len(outData)) / (p.seq * p.dim)

	return p, out, outData, cosData, sinData, nil
}

func applyRoPE(outData, cosData, sinData []float32, p ropeParams) {
	seqI := int(p.seq)
	dimI := int(p.dim)
	halfI := int(p.half)

	for pre := range p.prefix {
		prefixBase := int(pre * p.seq * p.dim)

		for t := range seqI {
			trigBase := int((p.pos + int64(t)) * p.half)

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
}
