package audio

import (
	"encoding/binary"
	"io"
	"math"
)

// RIFF/WAVE chunk identifiers.
const (
	riffChunkID = "RIFF"
	waveFormID  = "WAVE"
	dataChunkID = "data"
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
	copy(hdr[0:4], riffChunkID)
	binary.LittleEndian.PutUint32(hdr[4:8], 0xFFFFFFFF)
	copy(hdr[8:12], waveFormID)
	copy(hdr[12:16], "fmt ")
	binary.LittleEndian.PutUint32(hdr[16:20], 16)
	binary.LittleEndian.PutUint16(hdr[20:22], 1) // PCM
	binary.LittleEndian.PutUint16(hdr[22:24], channels)
	binary.LittleEndian.PutUint32(hdr[24:28], sampleRate)
	binary.LittleEndian.PutUint32(hdr[28:32], byteRate)
	binary.LittleEndian.PutUint16(hdr[32:34], blockAlign)
	binary.LittleEndian.PutUint16(hdr[34:36], bitsPerSample)
	copy(hdr[36:40], dataChunkID)
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
		buf[i*2] = byte(v)
		buf[i*2+1] = byte(v >> 8)
	}

	return w.Write(buf)
}

// FixStreamedWAVSizes rewrites the RIFF and data chunk sizes of a streamed WAV
// to the bytes actually present, in place, and returns data. Streaming writers
// put a placeholder there: WriteWAVHeaderStreaming uses 0xFFFFFFFF, and
// pocket-tts 3.x (`generate --output-path -`) declares 1e9 frames. Sizes are
// only ever shrunk; a correct header and anything that is not a RIFF/WAVE
// file come back unchanged.
func FixStreamedWAVSizes(data []byte) []byte {
	if len(data) < 12 || len(data) > math.MaxUint32 || string(data[0:4]) != riffChunkID || string(data[8:12]) != waveFormID {
		return data
	}

	for off := 12; off+8 <= len(data); {
		size := int64(binary.LittleEndian.Uint32(data[off+4 : off+8]))
		body := off + 8

		if string(data[off:off+4]) == dataChunkID {
			if remaining := int64(len(data) - body); size > remaining {
				binary.LittleEndian.PutUint32(data[off+4:off+8], uint32(remaining))
			}

			break
		}

		next := int64(body) + size + size%2
		if next > int64(len(data)) {
			return data
		}

		off = int(next)
	}

	if riff := int64(len(data) - 8); int64(binary.LittleEndian.Uint32(data[4:8])) > riff {
		binary.LittleEndian.PutUint32(data[4:8], uint32(riff))
	}

	return data
}
