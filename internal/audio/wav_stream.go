package audio

import (
	"encoding/binary"
	"io"
	"math"
)

// WriteWAVHeaderStreaming writes a 44-byte WAV header suitable for streaming
// where the total data length is not known in advance.  Both the RIFF chunk
// size and the data sub-chunk size are set to 0xFFFFFFFF, which is the
// conventional marker for an unknown/streaming length.
//
// Format: 24 kHz, mono, 16-bit PCM (matching ExpectedSampleRate).
func WriteWAVHeaderStreaming(w io.Writer) (int, error) {
	const (
		channels      = ExpectedChannels
		bitsPerSample = ExpectedBitDepth
		sampleRate    = ExpectedSampleRate
		byteRate      = sampleRate * channels * bitsPerSample / 8
		blockAlign    = channels * bitsPerSample / 8
	)

	var hdr [44]byte
	copy(hdr[0:4], "RIFF")
	binary.LittleEndian.PutUint32(hdr[4:8], 0xFFFFFFFF)
	copy(hdr[8:12], "WAVE")
	copy(hdr[12:16], "fmt ")
	binary.LittleEndian.PutUint32(hdr[16:20], 16)
	binary.LittleEndian.PutUint16(hdr[20:22], 1) // PCM
	binary.LittleEndian.PutUint16(hdr[22:24], channels)
	binary.LittleEndian.PutUint32(hdr[24:28], sampleRate)
	binary.LittleEndian.PutUint32(hdr[28:32], byteRate)
	binary.LittleEndian.PutUint16(hdr[32:34], blockAlign)
	binary.LittleEndian.PutUint16(hdr[34:36], bitsPerSample)
	copy(hdr[36:40], "data")
	binary.LittleEndian.PutUint32(hdr[40:44], 0xFFFFFFFF)

	return w.Write(hdr[:])
}

// WritePCM16Samples encodes float32 samples as little-endian 16-bit signed
// integers and writes them to w.  Samples are clamped to [-1, 1].
func WritePCM16Samples(w io.Writer, samples []float32) (int, error) {
	buf := make([]byte, len(samples)*2)
	for i, s := range samples {
		clamped := math.Max(-1.0, math.Min(1.0, float64(s)))
		v := int16(clamped * 32767)
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
	}

	return w.Write(buf)
}
