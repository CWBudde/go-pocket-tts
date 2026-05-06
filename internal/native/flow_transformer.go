package native

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/cwbudde/go-pocket-tts/internal/runtime/ops"
	"github.com/cwbudde/go-pocket-tts/internal/runtime/tensor"
	"github.com/cwbudde/go-pocket-tts/internal/safetensors"
)

type flowTransformerLayer struct {
	norm1   *LayerNorm
	norm2   *LayerNorm
	inProj  *Linear // self_attn.in_proj
	outProj *Linear // self_attn.out_proj
	linear1 *Linear
	linear2 *Linear
	nHeads  int64
	headDim int64
}

type flowTransformerLayerState struct {
	kCache *tensor.Tensor // [B, H, Tk, Dh]
	vCache *tensor.Tensor // [B, H, Tk, Dh]
	offset int64
}

func (s *flowTransformerLayerState) appendKV(k, v *tensor.Tensor) error {
	if s == nil {
		return errors.New("native: flow transformer layer state is nil")
	}

	if k == nil || v == nil {
		return errors.New("native: appendKV requires non-nil k/v tensors")
	}

	if s.kCache == nil || s.vCache == nil {
		s.kCache = k
		s.vCache = v
		s.offset += k.Shape()[2]

		return nil
	}

	err := s.ensureKVCapacity(k.Shape()[2])
	if err != nil {
		return err
	}

	err = copyKVAt(s.kCache, k, s.offset)
	if err != nil {
		return fmt.Errorf("native: append key cache: %w", err)
	}

	err = copyKVAt(s.vCache, v, s.offset)
	if err != nil {
		return fmt.Errorf("native: append value cache: %w", err)
	}

	s.offset += k.Shape()[2]

	return nil
}

func (s *flowTransformerLayerState) ensureKVCapacity(extra int64) error {
	if s == nil {
		return errors.New("native: flow transformer layer state is nil")
	}

	needed := s.offset + extra
	if needed < 0 {
		return fmt.Errorf("native: invalid KV cache size %d", needed)
	}

	kShape := s.kCache.Shape()
	vShape := s.vCache.Shape()
	if len(kShape) != 4 || len(vShape) != 4 {
		return fmt.Errorf("native: KV cache rank mismatch k=%v v=%v", kShape, vShape)
	}

	if !equalKVCacheLayout(kShape, vShape) {
		return fmt.Errorf("native: KV cache shape mismatch k=%v v=%v", kShape, vShape)
	}

	if needed <= kShape[2] {
		return nil
	}

	newCap := growCacheCapacity(kShape[2], needed)
	kNew, err := growKVCache(s.kCache, newCap)
	if err != nil {
		return fmt.Errorf("native: grow key cache: %w", err)
	}

	vNew, err := growKVCache(s.vCache, newCap)
	if err != nil {
		return fmt.Errorf("native: grow value cache: %w", err)
	}

	s.kCache = kNew
	s.vCache = vNew

	return nil
}

func loadFlowTransformerLayer(vb *VarBuilder, nHeads int64) (*flowTransformerLayer, error) {
	norm1, err := loadLayerNorm(vb, "norm1", 1e-5)
	if err != nil {
		return nil, err
	}

	norm2, err := loadLayerNorm(vb, "norm2", 1e-5)
	if err != nil {
		return nil, err
	}

	inProj, err := loadLinear(vb, "self_attn.in_proj", false)
	if err != nil {
		return nil, err
	}

	outProj, err := loadLinear(vb, "self_attn.out_proj", false)
	if err != nil {
		return nil, err
	}

	linear1, err := loadLinear(vb, "linear1", false)
	if err != nil {
		return nil, err
	}

	linear2, err := loadLinear(vb, "linear2", false)
	if err != nil {
		return nil, err
	}

	dModel := outProj.Weight.Shape()[0]
	if dModel%nHeads != 0 {
		return nil, fmt.Errorf("native: d_model %d not divisible by num_heads %d", dModel, nHeads)
	}

	return &flowTransformerLayer{
		norm1:   norm1,
		norm2:   norm2,
		inProj:  inProj,
		outProj: outProj,
		linear1: linear1,
		linear2: linear2,
		nHeads:  nHeads,
		headDim: dModel / nHeads,
	}, nil
}

func (l *flowTransformerLayer) forward(x, ropeCos, ropeSin *tensor.Tensor) (*tensor.Tensor, error) {
	n1, err := l.norm1.Forward(x)
	if err != nil {
		return nil, err
	}

	attn, err := l.selfAttention(n1, ropeCos, ropeSin)
	if err != nil {
		return nil, err
	}

	x, err = addSameShape(x, attn)
	if err != nil {
		return nil, err
	}

	n2, err := l.norm2.Forward(x)
	if err != nil {
		return nil, err
	}

	ff, err := l.linear1.Forward(n2)
	if err != nil {
		return nil, err
	}

	ff = geluErfTensor(ff)

	ff, err = l.linear2.Forward(ff)
	if err != nil {
		return nil, err
	}

	return addSameShape(x, ff)
}

func (l *flowTransformerLayer) projectQKV(
	x, ropeCos, ropeSin *tensor.Tensor,
	pos int64,
) (q, k, v *tensor.Tensor, err error) {
	shape := x.Shape() // [B, T, D]
	if len(shape) != 3 {
		return nil, nil, nil, fmt.Errorf("native: selfAttention expects [B, T, D], got %v", shape)
	}

	b, t := shape[0], shape[1]

	qkv, err := l.inProj.Forward(x)
	if err != nil {
		return nil, nil, nil, err
	}

	q, k, v, err = splitLastDim3(qkv)
	if err != nil {
		return nil, nil, nil, err
	}

	q, err = q.Reshape([]int64{b, t, l.nHeads, l.headDim})
	if err != nil {
		return nil, nil, nil, err
	}

	k, err = k.Reshape([]int64{b, t, l.nHeads, l.headDim})
	if err != nil {
		return nil, nil, nil, err
	}

	v, err = v.Reshape([]int64{b, t, l.nHeads, l.headDim})
	if err != nil {
		return nil, nil, nil, err
	}

	q, err = q.Transpose(1, 2) // [B, H, T, Dh]
	if err != nil {
		return nil, nil, nil, err
	}

	k, err = k.Transpose(1, 2)
	if err != nil {
		return nil, nil, nil, err
	}

	v, err = v.Transpose(1, 2)
	if err != nil {
		return nil, nil, nil, err
	}

	q, err = ops.RoPE(q, ropeCos, ropeSin, pos)
	if err != nil {
		return nil, nil, nil, err
	}

	k, err = ops.RoPE(k, ropeCos, ropeSin, pos)
	if err != nil {
		return nil, nil, nil, err
	}

	return q, k, v, nil
}

func (l *flowTransformerLayer) attentionFromQKV(
	q, k, v *tensor.Tensor,
	causal bool,
	offset int64,
) (*tensor.Tensor, error) {
	qShape := q.Shape()
	if len(qShape) != 4 {
		return nil, fmt.Errorf("native: attentionFromQKV expects q rank 4, got %v", qShape)
	}

	b, t := qShape[0], qShape[2]
	d := l.nHeads * l.headDim

	var a *tensor.Tensor
	var err error
	if causal {
		a, err = ops.Attention(q, k, v, true, offset)
	} else {
		a, err = ops.Attention(q, k, v, false, 0)
	}
	if err != nil {
		return nil, err
	}

	a, err = a.Transpose(1, 2) // [B, T, H, Dh]
	if err != nil {
		return nil, err
	}

	a, err = a.Reshape([]int64{b, t, d})
	if err != nil {
		return nil, err
	}

	return l.outProj.Forward(a)
}

func (l *flowTransformerLayer) attentionFromPositions(
	q, k, v *tensor.Tensor,
	posQ, posK []int64,
	context int64,
) (*tensor.Tensor, error) {
	qShape := q.Shape()
	if len(qShape) != 4 {
		return nil, fmt.Errorf("native: attentionFromPositions expects q rank 4, got %v", qShape)
	}

	b, t := qShape[0], qShape[2]
	d := l.nHeads * l.headDim

	a, err := ops.AttentionWithPositions(q, k, v, posQ, posK, context)
	if err != nil {
		return nil, err
	}

	a, err = a.Transpose(1, 2) // [B, T, H, Dh]
	if err != nil {
		return nil, err
	}

	a, err = a.Reshape([]int64{b, t, d})
	if err != nil {
		return nil, err
	}

	return l.outProj.Forward(a)
}

func (l *flowTransformerLayer) forwardWithState(
	x, ropeCos, ropeSin *tensor.Tensor,
	state *flowTransformerLayerState,
	incremental bool,
) (*tensor.Tensor, error) {
	if state == nil {
		return nil, errors.New("native: flow transformer layer state is nil")
	}

	n1, err := l.norm1.Forward(x)
	if err != nil {
		return nil, err
	}

	pos := state.offset

	q, k, v, err := l.projectQKV(n1, ropeCos, ropeSin, pos)
	if err != nil {
		return nil, err
	}

	err = state.appendKV(k, v)
	if err != nil {
		return nil, err
	}

	qLen := int64(0)
	if shape := q.Shape(); len(shape) == 4 {
		qLen = shape[2]
	}

	kLen := state.offset
	posQ := positionsRange(pos, qLen)
	posK := cachePositions(state.kCache, kLen)

	attn, err := l.attentionFromPositions(q, state.kCache, state.vCache, posQ, posK, ops.AttentionNoContext)
	if err != nil {
		return nil, err
	}

	x, err = addSameShape(x, attn)
	if err != nil {
		return nil, err
	}

	n2, err := l.norm2.Forward(x)
	if err != nil {
		return nil, err
	}

	ff, err := l.linear1.Forward(n2)
	if err != nil {
		return nil, err
	}

	ff = geluErfTensor(ff)

	ff, err = l.linear2.Forward(ff)
	if err != nil {
		return nil, err
	}

	return addSameShape(x, ff)
}

func positionsRange(start, count int64) []int64 {
	if count <= 0 {
		return nil
	}

	out := make([]int64, count)
	for i := range out {
		out[i] = start + int64(i)
	}

	return out
}

func cachePositions(cache *tensor.Tensor, validLen int64) []int64 {
	shape := cache.Shape()
	if len(shape) != 4 || shape[2] <= 0 {
		return nil
	}

	out := make([]int64, shape[2])
	for i := range out {
		if int64(i) < validLen {
			out[i] = int64(i)
		} else {
			out[i] = -1
		}
	}

	return out
}

func (l *flowTransformerLayer) selfAttention(x, ropeCos, ropeSin *tensor.Tensor) (*tensor.Tensor, error) {
	q, k, v, err := l.projectQKV(x, ropeCos, ropeSin, 0)
	if err != nil {
		return nil, err
	}

	return l.attentionFromQKV(q, k, v, true, 0)
}

type flowTransformer struct {
	layers  []*flowTransformerLayer
	ropeCos *tensor.Tensor // [max_seq, head_dim/2]
	ropeSin *tensor.Tensor // [max_seq, head_dim/2]
}

type flowTransformerState struct {
	layers []flowTransformerLayerState
}

func (t *flowTransformer) initState() (*flowTransformerState, error) {
	if t == nil {
		return nil, errors.New("native: flow transformer is nil")
	}

	return &flowTransformerState{
		layers: make([]flowTransformerLayerState, len(t.layers)),
	}, nil
}

func (t *flowTransformer) initStateFromVoiceModelState(voiceState *safetensors.VoiceModelState) (*flowTransformerState, error) {
	if t == nil {
		return nil, errors.New("native: flow transformer is nil")
	}

	if voiceState == nil {
		return nil, errors.New("native: voice model state is nil")
	}

	state := &flowTransformerState{
		layers: make([]flowTransformerLayerState, len(t.layers)),
	}

	for i, layer := range t.layers {
		moduleName := flowAttentionModuleName(i)
		module := voiceState.Modules[moduleName]
		if module == nil {
			return nil, fmt.Errorf("native: voice model state missing module %q", moduleName)
		}

		layerState, err := layerStateFromVoiceModule(moduleName, module, layer)
		if err != nil {
			return nil, err
		}

		state.layers[i] = layerState
	}

	return state, nil
}

func loadFlowTransformer(vb *VarBuilder, nHeads int64, maxPeriod float64) (*flowTransformer, error) {
	layers := make([]*flowTransformerLayer, 0, 8)

	for i := 0; ; i++ {
		layerPath := vb.Path("transformer", "layers", strconv.Itoa(i))
		if !layerPath.Has("norm1.weight") {
			break
		}

		layer, err := loadFlowTransformerLayer(layerPath, nHeads)
		if err != nil {
			return nil, fmt.Errorf("native: load flow transformer layer %d: %w", i, err)
		}

		layers = append(layers, layer)
	}

	if len(layers) == 0 {
		return nil, errors.New("native: no flow_lm transformer layers found")
	}

	headDim := layers[0].headDim

	cos, sin, err := buildRoPE(int64(8192), headDim, maxPeriod)
	if err != nil {
		return nil, err
	}

	return &flowTransformer{layers: layers, ropeCos: cos, ropeSin: sin}, nil
}

func flowAttentionModuleName(layer int) string {
	return "transformer.layers." + strconv.Itoa(layer) + ".self_attn"
}

func layerStateFromVoiceModule(moduleName string, module map[string]*safetensors.Tensor, layer *flowTransformerLayer) (flowTransformerLayerState, error) {
	cache := module["cache"]
	if cache == nil {
		return flowTransformerLayerState{}, fmt.Errorf("native: voice model state module %q missing cache", moduleName)
	}

	offsetTensor := module["offset"]
	if offsetTensor == nil {
		return flowTransformerLayerState{}, fmt.Errorf("native: voice model state module %q missing offset", moduleName)
	}

	offset, err := readVoiceStateOffset(moduleName, offsetTensor)
	if err != nil {
		return flowTransformerLayerState{}, err
	}

	kCache, vCache, err := splitVoiceKVCache(moduleName, cache, layer)
	if err != nil {
		return flowTransformerLayerState{}, err
	}

	if offset < 0 {
		return flowTransformerLayerState{}, fmt.Errorf("native: voice model state module %q has negative offset %d", moduleName, offset)
	}

	cacheShape := kCache.Shape()
	if len(cacheShape) != 4 {
		return flowTransformerLayerState{}, fmt.Errorf("native: internal key cache rank %d, want 4", len(cacheShape))
	}

	if offset > cacheShape[2] {
		return flowTransformerLayerState{}, fmt.Errorf("native: voice model state module %q offset %d exceeds cache length %d", moduleName, offset, cacheShape[2])
	}

	return flowTransformerLayerState{kCache: kCache, vCache: vCache, offset: offset}, nil
}

func readVoiceStateOffset(moduleName string, offset *safetensors.Tensor) (int64, error) {
	if len(offset.Data) == 0 {
		return 0, fmt.Errorf("native: voice model state module %q has empty offset tensor", moduleName)
	}

	v := offset.Data[0]
	i := int64(v)
	if float32(i) != v {
		return 0, fmt.Errorf("native: voice model state module %q offset %v is not an integer", moduleName, v)
	}

	return i, nil
}

func splitVoiceKVCache(moduleName string, cache *safetensors.Tensor, layer *flowTransformerLayer) (*tensor.Tensor, *tensor.Tensor, error) {
	shape := cache.Shape
	if len(shape) != 5 {
		return nil, nil, fmt.Errorf("native: voice model state module %q cache shape %v, want [2,B,T,H,D]", moduleName, shape)
	}

	if shape[0] != 2 {
		return nil, nil, fmt.Errorf("native: voice model state module %q cache first dim %d, want 2", moduleName, shape[0])
	}

	b, steps, heads, headDim := shape[1], shape[2], shape[3], shape[4]
	if b <= 0 || steps < 0 || heads <= 0 || headDim <= 0 {
		return nil, nil, fmt.Errorf("native: voice model state module %q has invalid cache shape %v", moduleName, shape)
	}

	if layer != nil {
		if layer.nHeads != 0 && heads != layer.nHeads {
			return nil, nil, fmt.Errorf("native: voice model state module %q heads %d, want %d", moduleName, heads, layer.nHeads)
		}

		if layer.headDim != 0 && headDim != layer.headDim {
			return nil, nil, fmt.Errorf("native: voice model state module %q head dim %d, want %d", moduleName, headDim, layer.headDim)
		}
	}

	want := int(shape[0] * b * steps * heads * headDim)
	if len(cache.Data) != want {
		return nil, nil, fmt.Errorf("native: voice model state module %q cache data length %d, want %d", moduleName, len(cache.Data), want)
	}

	outLen := int(b * heads * steps * headDim)
	kData := make([]float32, outLen)
	vData := make([]float32, outLen)

	for batch := int64(0); batch < b; batch++ {
		for step := int64(0); step < steps; step++ {
			for head := int64(0); head < heads; head++ {
				for dim := int64(0); dim < headDim; dim++ {
					dst := (((batch*heads+head)*steps+step)*headDim + dim)
					kSrc := voiceKVIndex(0, batch, step, head, dim, b, steps, heads, headDim)
					vSrc := voiceKVIndex(1, batch, step, head, dim, b, steps, heads, headDim)
					kData[dst] = cache.Data[kSrc]
					vData[dst] = cache.Data[vSrc]
				}
			}
		}
	}

	k, err := tensor.New(kData, []int64{b, heads, steps, headDim})
	if err != nil {
		return nil, nil, err
	}

	v, err := tensor.New(vData, []int64{b, heads, steps, headDim})
	if err != nil {
		return nil, nil, err
	}

	return k, v, nil
}

func voiceKVIndex(kv, batch, step, head, dim, batches, steps, heads, headDim int64) int {
	return int(((((kv*batches+batch)*steps+step)*heads+head)*headDim + dim))
}

func equalKVCacheLayout(a, b []int64) bool {
	return len(a) == 4 &&
		len(b) == 4 &&
		a[0] == b[0] &&
		a[1] == b[1] &&
		a[2] == b[2] &&
		a[3] == b[3]
}

func growCacheCapacity(current, needed int64) int64 {
	if current < 1 {
		current = 1
	}

	next := current
	for next < needed {
		next *= 2
	}

	return next
}

func growKVCache(cache *tensor.Tensor, capacity int64) (*tensor.Tensor, error) {
	shape := cache.Shape()
	if len(shape) != 4 {
		return nil, fmt.Errorf("cache shape %v, want [B,H,T,D]", shape)
	}

	if capacity == shape[2] {
		return cache, nil
	}

	batches, heads, fullSteps, headDim := shape[0], shape[1], shape[2], shape[3]
	if capacity < fullSteps {
		return nil, fmt.Errorf("capacity %d smaller than current cache length %d", capacity, fullSteps)
	}

	data := cache.RawData()
	out := make([]float32, int(batches*heads*capacity*headDim))
	for batch := int64(0); batch < batches; batch++ {
		for head := int64(0); head < heads; head++ {
			for step := int64(0); step < fullSteps; step++ {
				src := int((((batch*heads+head)*fullSteps + step) * headDim))
				dst := int((((batch*heads+head)*capacity + step) * headDim))
				copy(out[dst:dst+int(headDim)], data[src:src+int(headDim)])
			}
		}
	}

	return tensor.New(out, []int64{batches, heads, capacity, headDim})
}

func copyKVAt(dst, src *tensor.Tensor, offset int64) error {
	dstShape := dst.Shape()
	srcShape := src.Shape()
	if len(dstShape) != 4 || len(srcShape) != 4 {
		return fmt.Errorf("cache rank mismatch dst=%v src=%v", dstShape, srcShape)
	}

	if dstShape[0] != srcShape[0] || dstShape[1] != srcShape[1] || dstShape[3] != srcShape[3] {
		return fmt.Errorf("cache layout mismatch dst=%v src=%v", dstShape, srcShape)
	}

	if offset < 0 || offset+srcShape[2] > dstShape[2] {
		return fmt.Errorf("write range [%d:%d] outside cache length %d", offset, offset+srcShape[2], dstShape[2])
	}

	dstData := dst.RawData()
	srcData := src.RawData()
	batches, heads, dstSteps, headDim := dstShape[0], dstShape[1], dstShape[2], dstShape[3]
	srcSteps := srcShape[2]
	for batch := int64(0); batch < batches; batch++ {
		for head := int64(0); head < heads; head++ {
			for step := int64(0); step < srcSteps; step++ {
				dstOff := int((((batch*heads+head)*dstSteps + offset + step) * headDim))
				srcOff := int((((batch*heads+head)*srcSteps + step) * headDim))
				copy(dstData[dstOff:dstOff+int(headDim)], srcData[srcOff:srcOff+int(headDim)])
			}
		}
	}

	return nil
}

func flowLayerIndexFromModule(module string) (int, bool) {
	const prefix = "transformer.layers."
	const suffix = ".self_attn"
	if !strings.HasPrefix(module, prefix) || !strings.HasSuffix(module, suffix) {
		return 0, false
	}

	raw := strings.TrimSuffix(strings.TrimPrefix(module, prefix), suffix)
	idx, err := strconv.Atoi(raw)
	if err != nil {
		return 0, false
	}

	return idx, true
}

func (t *flowTransformer) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if t == nil {
		return nil, errors.New("native: flow transformer is nil")
	}

	var err error
	for i, layer := range t.layers {
		x, err = layer.forward(x, t.ropeCos, t.ropeSin)
		if err != nil {
			return nil, fmt.Errorf("native: transformer layer %d: %w", i, err)
		}
	}

	return x, nil
}

func (t *flowTransformer) prefill(x *tensor.Tensor, state *flowTransformerState) error {
	if t == nil {
		return errors.New("native: flow transformer is nil")
	}

	if state == nil {
		return errors.New("native: flow transformer state is nil")
	}

	if len(state.layers) != len(t.layers) {
		return fmt.Errorf("native: flow transformer state layer count %d does not match transformer layers %d", len(state.layers), len(t.layers))
	}

	var err error
	for i, layer := range t.layers {
		x, err = layer.forwardWithState(x, t.ropeCos, t.ropeSin, &state.layers[i], false)
		if err != nil {
			return fmt.Errorf("native: transformer prefill layer %d: %w", i, err)
		}
	}

	return nil
}

func (t *flowTransformer) step(x *tensor.Tensor, state *flowTransformerState) (*tensor.Tensor, error) {
	if t == nil {
		return nil, errors.New("native: flow transformer is nil")
	}

	if state == nil {
		return nil, errors.New("native: flow transformer state is nil")
	}

	if len(state.layers) != len(t.layers) {
		return nil, fmt.Errorf("native: flow transformer state layer count %d does not match transformer layers %d", len(state.layers), len(t.layers))
	}

	var err error
	for i, layer := range t.layers {
		x, err = layer.forwardWithState(x, t.ropeCos, t.ropeSin, &state.layers[i], true)
		if err != nil {
			return nil, fmt.Errorf("native: transformer step layer %d: %w", i, err)
		}
	}

	return x, nil
}

func buildRoPE(maxSeq, headDim int64, maxPeriod float64) (*tensor.Tensor, *tensor.Tensor, error) {
	if headDim%2 != 0 {
		return nil, nil, fmt.Errorf("native: rope head dim must be even, got %d", headDim)
	}

	half := int(headDim / 2)

	invFreq := make([]float64, half)
	for i := range half {
		invFreq[i] = 1.0 / math.Pow(maxPeriod, float64(i)/float64(half))
	}

	cos := make([]float32, int(maxSeq)*half)

	sin := make([]float32, int(maxSeq)*half)
	for pos := range maxSeq {
		base := int(pos) * half
		for i, f := range invFreq {
			angle := float64(pos) * f
			cos[base+i] = float32(math.Cos(angle))
			sin[base+i] = float32(math.Sin(angle))
		}
	}

	cosT, err := tensor.New(cos, []int64{maxSeq, headDim / 2})
	if err != nil {
		return nil, nil, err
	}

	sinT, err := tensor.New(sin, []int64{maxSeq, headDim / 2})
	if err != nil {
		return nil, nil, err
	}

	return cosT, sinT, nil
}

func detectNumHeads(vb *VarBuilder, fallback int64) int64 {
	if vb == nil {
		return fallback
	}

	first := vb.Path("transformer", "layers", "0")

	w, err := first.Tensor("self_attn.in_proj.weight")
	if err != nil {
		return fallback
	}

	shape := w.Shape()
	if len(shape) != 2 {
		return fallback
	}

	dModel := shape[1]
	if dModel <= 0 {
		return fallback
	}

	// Heuristic from known PocketTTS configs.
	for _, n := range []int64{16, 8, 4, 2, 1} {
		if dModel%n == 0 {
			return n
		}
	}

	return fallback
}
