package audio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/cwbudde/wav"
	goaudio "github.com/go-audio/audio"
)

// EncodeWAV encodes float32 PCM samples as a WAV byte slice
// using 24000 Hz, mono, 16-bit PCM format.
func EncodeWAV(samples []float32) ([]byte, error) {
	var buf bytes.Buffer

	// wav.NewEncoder requires an io.WriteSeeker; bytes.Buffer is not one.
	// Use a seekable wrapper.
	sw := &seekBuffer{buf: &buf}

	enc := wav.NewEncoder(sw, ExpectedSampleRate, ExpectedBitDepth, ExpectedChannels, 1) // 1 = PCM

	pcmBuf := &goaudio.Float32Buffer{
		Data:           samples,
		Format:         &goaudio.Format{SampleRate: ExpectedSampleRate, NumChannels: ExpectedChannels},
		SourceBitDepth: ExpectedBitDepth,
	}

	err := enc.Write(pcmBuf)
	if err != nil {
		return nil, fmt.Errorf("writing PCM: %w", err)
	}

	err = enc.Close()
	if err != nil {
		return nil, fmt.Errorf("closing encoder: %w", err)
	}

	return buf.Bytes(), nil
}

// AppendTrailingSilenceWAV appends TrailingSilenceSec of zero frames to a
// PCM WAV whose data chunk is its last chunk, as EncodeWAV writes it, and
// patches the RIFF and data sizes in a new slice; wav is not modified. Unlike
// decoding and re-encoding, it holds one extra copy of the audio.
func AppendTrailingSilenceWAV(wav []byte) ([]byte, error) {
	if len(wav) < 12 || string(wav[0:4]) != "RIFF" || string(wav[8:12]) != "WAVE" {
		return nil, errors.New("invalid WAV file")
	}

	sampleRate, blockAlign := 0, 0

	for off := 12; off+8 <= len(wav); {
		id := string(wav[off : off+4])
		size := int(binary.LittleEndian.Uint32(wav[off+4 : off+8]))

		switch {
		case id == "fmt " && size >= 16 && off+8+size <= len(wav):
			sampleRate = int(binary.LittleEndian.Uint32(wav[off+12 : off+16]))
			blockAlign = int(binary.LittleEndian.Uint16(wav[off+20 : off+22]))
		case id == "data":
			if off+8+size != len(wav) || sampleRate == 0 || blockAlign == 0 {
				return nil, errors.New("WAV data chunk is not the last chunk after a fmt chunk")
			}

			pad := int(float64(sampleRate)*TrailingSilenceSec) * blockAlign
			out := make([]byte, len(wav)+pad)
			copy(out, wav)
			binary.LittleEndian.PutUint32(out[off+4:off+8], uint32(size+pad))
			binary.LittleEndian.PutUint32(out[4:8], uint32(len(out)-8))

			return out, nil
		}

		off += 8 + size + size%2
	}

	return nil, errors.New("WAV has no data chunk")
}

// seekBuffer wraps a bytes.Buffer to satisfy io.WriteSeeker.
type seekBuffer struct {
	buf *bytes.Buffer
	pos int
}

func (s *seekBuffer) Write(p []byte) (int, error) {
	// If writing at the end, just append.
	if s.pos == s.buf.Len() {
		n, err := s.buf.Write(p)
		s.pos += n

		return n, err
	}
	// Writing in the middle: overwrite existing bytes.
	data := s.buf.Bytes()

	n := copy(data[s.pos:], p)
	if n < len(p) {
		// Extend the buffer for the remainder.
		data = append(data, p[n:]...)
		// Reset buffer with extended data.
		s.buf.Reset()
		s.buf.Write(data)

		n = len(p)
	}

	s.pos += n

	return n, nil
}

func (s *seekBuffer) Seek(offset int64, whence int) (int64, error) {
	var newPos int

	switch whence {
	case 0: // io.SeekStart
		newPos = int(offset)
	case 1: // io.SeekCurrent
		newPos = s.pos + int(offset)
	case 2: // io.SeekEnd
		newPos = s.buf.Len() + int(offset)
	}

	if newPos < 0 {
		return 0, errors.New("seek before start")
	}

	s.pos = newPos

	return int64(newPos), nil
}
