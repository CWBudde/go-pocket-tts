package tokenizer

// Minimal reader for serialized SentencePiece ModelProto files
// (sentencepiece_model.proto). Only the fields the UNIGRAM encoder needs are
// decoded; everything else is skipped. Field numbers were checked against the
// shipped tokenizer models.

import (
	"errors"
	"fmt"
	"math"

	"google.golang.org/protobuf/encoding/protowire"
)

// spPieceType is ModelProto.SentencePiece.Type.
type spPieceType uint64

const (
	spNormal      spPieceType = 1
	spUnknown     spPieceType = 2
	spControl     spPieceType = 3
	spUserDefined spPieceType = 4
	spUnused      spPieceType = 5
	spByte        spPieceType = 6
)

// spModelTypeUnigram is TrainerSpec.ModelType UNIGRAM.
const spModelTypeUnigram = 1

// ModelProto field numbers.
const (
	fieldModelPieces     protowire.Number = 1
	fieldModelTrainer    protowire.Number = 2
	fieldModelNormalizer protowire.Number = 3

	fieldPiecePiece protowire.Number = 1
	fieldPieceScore protowire.Number = 2
	fieldPieceType  protowire.Number = 3

	fieldTrainerModelType       protowire.Number = 3
	fieldTrainerWhitespaceAsSfx protowire.Number = 24
	fieldTrainerByteFallback    protowire.Number = 35

	fieldNormName              protowire.Number = 1
	fieldNormCharsmap          protowire.Number = 2
	fieldNormAddDummyPrefix    protowire.Number = 3
	fieldNormRemoveExtraSpaces protowire.Number = 4
	fieldNormEscapeWhitespaces protowire.Number = 5
)

var errWireType = errors.New("unexpected wire type")

type spPieceProto struct {
	piece string
	score float32
	typ   spPieceType
}

// spModelProto holds the decoded ModelProto fields, initialised with the
// proto2 defaults.
type spModelProto struct {
	pieces []spPieceProto

	modelType               uint64
	byteFallback            bool
	treatWhitespaceAsSuffix bool

	normalizerName         string
	precompiledCharsmap    []byte
	addDummyPrefix         bool
	removeExtraWhitespaces bool
	escapeWhitespaces      bool
}

// spField is one decoded field; only the member matching its wire type is set.
type spField struct {
	num    protowire.Number
	typ    protowire.Type
	varint uint64
	fixed  uint32
	bytes  []byte
}

func (f spField) expect(typ protowire.Type) error {
	if f.typ != typ {
		return fmt.Errorf("field %d: %w %d", f.num, errWireType, f.typ)
	}

	return nil
}

// spEachField decodes the fields of one serialized message in order.
func spEachField(b []byte, fn func(f spField) error) error {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return protowire.ParseError(n)
		}

		b = b[n:]
		f := spField{num: num, typ: typ}

		switch typ {
		case protowire.VarintType:
			f.varint, n = protowire.ConsumeVarint(b)
		case protowire.Fixed32Type:
			f.fixed, n = protowire.ConsumeFixed32(b)
		case protowire.BytesType:
			f.bytes, n = protowire.ConsumeBytes(b)
		default:
			n = protowire.ConsumeFieldValue(num, typ, b)
		}

		if n < 0 {
			return protowire.ParseError(n)
		}

		b = b[n:]

		err := fn(f)
		if err != nil {
			return err
		}
	}

	return nil
}

// parseSpModelProto decodes a serialized ModelProto. Repeated occurrences of
// the singular sub-messages merge, as in proto2.
func parseSpModelProto(data []byte) (*spModelProto, error) {
	m := &spModelProto{
		modelType:              spModelTypeUnigram,
		addDummyPrefix:         true,
		removeExtraWhitespaces: true,
		escapeWhitespaces:      true,
	}

	err := spEachField(data, func(f spField) error {
		switch f.num {
		case fieldModelPieces:
			return m.parsePiece(f)
		case fieldModelTrainer:
			return m.parseSubMessage(f, m.trainerField)
		case fieldModelNormalizer:
			return m.parseSubMessage(f, m.normalizerField)
		default:
			return nil
		}
	})
	if err != nil {
		return nil, fmt.Errorf("parse sentencepiece model: %w", err)
	}

	return m, nil
}

func (m *spModelProto) parseSubMessage(f spField, fn func(f spField) error) error {
	err := f.expect(protowire.BytesType)
	if err != nil {
		return err
	}

	return spEachField(f.bytes, fn)
}

func (m *spModelProto) parsePiece(f spField) error {
	p := spPieceProto{typ: spNormal}

	err := m.parseSubMessage(f, func(f spField) error {
		switch f.num {
		case fieldPiecePiece:
			p.piece = string(f.bytes)

			return f.expect(protowire.BytesType)
		case fieldPieceScore:
			p.score = math.Float32frombits(f.fixed)

			return f.expect(protowire.Fixed32Type)
		case fieldPieceType:
			p.typ = spPieceType(f.varint)

			return f.expect(protowire.VarintType)
		default:
			return nil
		}
	})
	if err != nil {
		return fmt.Errorf("piece %d: %w", len(m.pieces), err)
	}

	m.pieces = append(m.pieces, p)

	return nil
}

func (m *spModelProto) trainerField(f spField) error {
	switch f.num {
	case fieldTrainerModelType:
		m.modelType = f.varint
	case fieldTrainerWhitespaceAsSfx:
		m.treatWhitespaceAsSuffix = f.varint != 0
	case fieldTrainerByteFallback:
		m.byteFallback = f.varint != 0
	default:
		return nil
	}

	return f.expect(protowire.VarintType)
}

func (m *spModelProto) normalizerField(f spField) error {
	switch f.num {
	case fieldNormName:
		m.normalizerName = string(f.bytes)

		return f.expect(protowire.BytesType)
	case fieldNormCharsmap:
		m.precompiledCharsmap = f.bytes

		return f.expect(protowire.BytesType)
	case fieldNormAddDummyPrefix:
		m.addDummyPrefix = f.varint != 0
	case fieldNormRemoveExtraSpaces:
		m.removeExtraWhitespaces = f.varint != 0
	case fieldNormEscapeWhitespaces:
		m.escapeWhitespaces = f.varint != 0
	default:
		return nil
	}

	return f.expect(protowire.VarintType)
}
