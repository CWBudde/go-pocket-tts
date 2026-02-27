package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
)

const (
	dtypeF32  = "F32"
	dtypeF16  = "F16"
	dtypeBF16 = "BF16"
)

type KeyMapper func(name string) (mapped string, keep bool)

type RemapMode string

const (
	RemapLenient RemapMode = "lenient"
	RemapStrict  RemapMode = "strict"
)

type StoreOptions struct {
	KeyMapper KeyMapper
	RemapMode RemapMode
}

type Store struct {
	raw     []byte
	entries map[string]storeEntry
	names   []string
}

type storeEntry struct {
	OriginalName string
	DType        string
	Shape        []int64
	Start        int
	End          int
}

type storeHeaderEntry struct {
	DType   string  `json:"dtype"`
	Shape   []int64 `json:"shape"`
	Offsets [2]int  `json:"data_offsets"`
}

func OpenStore(path string, opts StoreOptions) (*Store, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: read %s: %w", path, err)
	}

	return OpenStoreFromBytes(data, opts)
}

func OpenStoreFromBytes(data []byte, opts StoreOptions) (*Store, error) {
	keyMapper := opts.KeyMapper
	if keyMapper == nil {
		keyMapper = func(name string) (string, bool) { return name, true }
	}

	mode := opts.RemapMode
	if mode == "" {
		mode = RemapLenient
	}

	headerEnd, header, err := decodeHeader(data)
	if err != nil {
		return nil, err
	}

	keys := make([]string, 0, len(header))
	for name := range header {
		keys = append(keys, name)
	}

	sort.Strings(keys)

	entries := make(map[string]storeEntry, len(keys))
	names := make([]string, 0, len(keys))

	for _, original := range keys {
		if original == "__metadata__" {
			continue
		}

		entry, err := parseHeaderEntry(header[original])
		if err != nil {
			return nil, fmt.Errorf("safetensors: decode header entry %q: %w", original, err)
		}

		if err := validateHeaderEntry(original, entry); err != nil {
			return nil, err
		}

		mapped, keep := keyMapper(original)
		if !keep {
			if mode == RemapStrict {
				return nil, fmt.Errorf("safetensors: strict remap rejected tensor %q", original)
			}

			continue
		}

		mapped = strings.TrimSpace(mapped)
		if mapped == "" {
			return nil, fmt.Errorf("safetensors: remapped tensor name for %q is empty", original)
		}

		if _, exists := entries[mapped]; exists {
			if mode == RemapStrict {
				return nil, fmt.Errorf("safetensors: strict remap collision for %q", mapped)
			}

			continue
		}

		start := headerEnd + entry.Offsets[0]

		end := headerEnd + entry.Offsets[1]
		if start < headerEnd || end < start || end > len(data) {
			return nil, fmt.Errorf(
				"safetensors: tensor %q data [%d:%d] exceeds file size %d",
				original,
				start,
				end,
				len(data),
			)
		}

		elemCount, err := shapeElementCount(entry.Shape)
		if err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", original, err)
		}

		elemBytes, err := dtypeBytes(entry.DType)
		if err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", original, err)
		}

		expectedBytes := int(elemCount) * elemBytes

		actualBytes := end - start
		if actualBytes < expectedBytes {
			return nil, fmt.Errorf(
				"safetensors: tensor %q needs %d bytes but data has %d",
				original,
				expectedBytes,
				actualBytes,
			)
		}

		entries[mapped] = storeEntry{
			OriginalName: original,
			DType:        strings.ToUpper(entry.DType),
			Shape:        append([]int64(nil), entry.Shape...),
			Start:        start,
			End:          end,
		}
		names = append(names, mapped)
	}

	if len(entries) == 0 {
		return nil, errors.New("safetensors: no tensors found")
	}

	sort.Strings(names)

	return &Store{
		raw:     data,
		entries: entries,
		names:   names,
	}, nil
}

func (s *Store) Names() []string {
	return append([]string(nil), s.names...)
}

func (s *Store) Has(name string) bool {
	_, ok := s.entries[name]
	return ok
}

func (s *Store) Tensor(name string) (*Tensor, error) {
	entry, ok := s.entries[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found (available: %s)", name, summarizeNames(s.names))
	}

	data, err := decodeTensorData(s.raw[entry.Start:entry.End], entry.DType, entry.Shape)
	if err != nil {
		return nil, fmt.Errorf("safetensors: tensor %q decode: %w", name, err)
	}

	return &Tensor{
		Name:  name,
		Shape: append([]int64(nil), entry.Shape...),
		Data:  data,
	}, nil
}

func (s *Store) TensorWithShape(name string, wantShape []int64) (*Tensor, error) {
	t, err := s.Tensor(name)
	if err != nil {
		return nil, err
	}

	if !equalShape(t.Shape, wantShape) {
		return nil, fmt.Errorf("safetensors: tensor %q shape %v does not match expected %v", name, t.Shape, wantShape)
	}

	return t, nil
}

func (s *Store) ReadAll() (map[string]*Tensor, error) {
	out := make(map[string]*Tensor, len(s.names))
	for _, name := range s.names {
		t, err := s.Tensor(name)
		if err != nil {
			return nil, err
		}

		out[name] = t
	}

	return out, nil
}

func (s *Store) Close() {
	s.raw = nil
	s.entries = nil
	s.names = nil
}

func decodeHeader(data []byte) (int, map[string]json.RawMessage, error) {
	if len(data) < 8 {
		return 0, nil, fmt.Errorf("safetensors: file too short (%d bytes)", len(data))
	}

	headerLen := binary.LittleEndian.Uint64(data[:8])

	headerEnd := 8 + int(headerLen)
	if headerEnd > len(data) {
		return 0, nil, fmt.Errorf("safetensors: header length %d exceeds file size %d", headerLen, len(data))
	}

	var header map[string]json.RawMessage

	err := json.Unmarshal(data[8:headerEnd], &header)
	if err != nil {
		return 0, nil, fmt.Errorf("safetensors: parse header: %w", err)
	}

	return headerEnd, header, nil
}

func parseHeaderEntry(raw json.RawMessage) (storeHeaderEntry, error) {
	var e storeHeaderEntry

	err := json.Unmarshal(raw, &e)
	if err != nil {
		return storeHeaderEntry{}, err
	}

	return e, nil
}

func validateHeaderEntry(name string, entry storeHeaderEntry) error {
	switch strings.ToUpper(entry.DType) {
	case dtypeF32, dtypeF16, dtypeBF16:
	default:
		return fmt.Errorf("safetensors: tensor %q has unsupported dtype %q", name, entry.DType)
	}

	if entry.Offsets[0] < 0 || entry.Offsets[1] < entry.Offsets[0] {
		return fmt.Errorf("safetensors: tensor %q has invalid data offsets %v", name, entry.Offsets)
	}

	for _, d := range entry.Shape {
		if d < 0 {
			return fmt.Errorf("safetensors: tensor %q has negative shape dimension in %v", name, entry.Shape)
		}
	}

	return nil
}

func shapeElementCount(shape []int64) (int64, error) {
	total := int64(1)

	for _, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("negative dimension %d", d)
		}

		if d == 0 {
			return 0, nil
		}

		if total > math.MaxInt64/d {
			return 0, fmt.Errorf("shape %v overflows element count", shape)
		}

		total *= d
	}

	return total, nil
}

func dtypeBytes(dtype string) (int, error) {
	switch strings.ToUpper(dtype) {
	case dtypeF32:
		return 4, nil
	case dtypeF16, dtypeBF16:
		return 2, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %q", dtype)
	}
}

func decodeTensorData(raw []byte, dtype string, shape []int64) ([]float32, error) {
	elemCount, err := shapeElementCount(shape)
	if err != nil {
		return nil, err
	}

	n := int(elemCount)
	out := make([]float32, n)

	switch strings.ToUpper(dtype) {
	case dtypeF32:
		if len(raw) < n*4 {
			return nil, fmt.Errorf("need %d bytes for F32, got %d", n*4, len(raw))
		}

		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}

		return out, nil
	case dtypeF16:
		if len(raw) < n*2 {
			return nil, fmt.Errorf("need %d bytes for F16, got %d", n*2, len(raw))
		}

		for i := range out {
			bits := binary.LittleEndian.Uint16(raw[i*2:])
			out[i] = float16ToFloat32(bits)
		}

		return out, nil
	case dtypeBF16:
		if len(raw) < n*2 {
			return nil, fmt.Errorf("need %d bytes for BF16, got %d", n*2, len(raw))
		}

		for i := range out {
			bits := binary.LittleEndian.Uint16(raw[i*2:])
			out[i] = math.Float32frombits(uint32(bits) << 16)
		}

		return out, nil
	default:
		return nil, fmt.Errorf("unsupported dtype %q", dtype)
	}
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h & 0x03ff)

	var bits uint32

	switch exp {
	case 0:
		if frac == 0 {
			bits = sign << 31
		} else {
			// Subnormal: normalize.
			e := int32(-14)

			for (frac & 0x0400) == 0 {
				frac <<= 1
				e--
			}

			frac &= 0x03ff
			exp32 := uint32(e + 127)
			bits = (sign << 31) | (exp32 << 23) | (frac << 13)
		}
	case 0x1f:
		// Inf / NaN.
		bits = (sign << 31) | 0x7f800000 | (frac << 13)
	default:
		exp32 := exp + (127 - 15)
		bits = (sign << 31) | (exp32 << 23) | (frac << 13)
	}

	return math.Float32frombits(bits)
}

func equalShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

func summarizeNames(names []string) string {
	if len(names) == 0 {
		return "none"
	}

	const maxNames = 8
	if len(names) <= maxNames {
		return strings.Join(names, ", ")
	}

	return strings.Join(names[:maxNames], ", ") + ", ..."
}
