package audio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

type Hook func(samples []float32) []float32

func ApplyHooks(samples []float32, hooks ...Hook) []float32 {
	out := samples
	for _, hook := range hooks {
		out = hook(out)
	}

	return out
}

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
	byteRate := sampleRate * channels * bitsPerSample / 8
	blockAlign := channels * bitsPerSample / 8
	dataSize := len(pcm) * 2
	riffSize := 4 + (8 + 16) + (8 + dataSize)

	buf := &bytes.Buffer{}
	buf.WriteString("RIFF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(riffSize))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(buf, binary.LittleEndian, uint16(1))
	_ = binary.Write(buf, binary.LittleEndian, uint16(channels))
	_ = binary.Write(buf, binary.LittleEndian, uint32(sampleRate))
	_ = binary.Write(buf, binary.LittleEndian, uint32(byteRate))
	_ = binary.Write(buf, binary.LittleEndian, uint16(blockAlign))
	_ = binary.Write(buf, binary.LittleEndian, uint16(bitsPerSample))
	buf.WriteString("data")

	_ = binary.Write(buf, binary.LittleEndian, uint32(dataSize))
	for _, s := range pcm {
		_ = binary.Write(buf, binary.LittleEndian, s)
	}

	return buf.Bytes(), nil
}
