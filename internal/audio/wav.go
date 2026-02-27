package audio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

func toUint32Checked(value int64, label string) (uint32, error) {
	const maxUint32 = int64(^uint32(0))
	if value < 0 || value > maxUint32 {
		return 0, fmt.Errorf("%s exceeds uint32: %d", label, value)
	}

	return uint32(value), nil
}

type Hook func(samples []float32) []float32

func ApplyHooks(samples []float32, hooks ...Hook) []float32 {
	out := samples
	for _, hook := range hooks {
		out = hook(out)
	}

	return out
}

//nolint:funlen // WAV header construction stays explicit and validated in one place.
func EncodeWAVPCM16(samples []float32, sampleRate int) ([]byte, error) {
	if sampleRate < 1 {
		return nil, fmt.Errorf("invalid sample rate: %d", sampleRate)
	}

	pcm := make([]int16, len(samples))
	for i, s := range samples {
		clamped := math.Max(-1.0, math.Min(1.0, float64(s)))
		pcm[i] = int16(clamped * 32767)
	}

	const channels = 1
	const bitsPerSample = 16
	byteRate := int64(sampleRate) * channels * bitsPerSample / 8
	blockAlign := channels * bitsPerSample / 8
	dataSize := int64(len(pcm)) * 2
	riffSize := int64(4+(8+16)+8) + dataSize

	riffSizeU32, err := toUint32Checked(riffSize, "riff size")
	if err != nil {
		return nil, err
	}

	sampleRateU32, err := toUint32Checked(int64(sampleRate), "sample rate")
	if err != nil {
		return nil, err
	}

	byteRateU32, err := toUint32Checked(byteRate, "byte rate")
	if err != nil {
		return nil, err
	}

	dataSizeU32, err := toUint32Checked(dataSize, "data size")
	if err != nil {
		return nil, err
	}

	buf := &bytes.Buffer{}
	buf.WriteString("RIFF")
	_ = binary.Write(buf, binary.LittleEndian, riffSizeU32)
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(buf, binary.LittleEndian, uint16(1))
	_ = binary.Write(buf, binary.LittleEndian, uint16(channels))
	_ = binary.Write(buf, binary.LittleEndian, sampleRateU32)
	_ = binary.Write(buf, binary.LittleEndian, byteRateU32)
	_ = binary.Write(buf, binary.LittleEndian, uint16(blockAlign))
	_ = binary.Write(buf, binary.LittleEndian, uint16(bitsPerSample))
	buf.WriteString("data")

	_ = binary.Write(buf, binary.LittleEndian, dataSizeU32)
	for _, s := range pcm {
		_ = binary.Write(buf, binary.LittleEndian, s)
	}

	return buf.Bytes(), nil
}
