package audio

import (
	"bytes"
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

	if err := enc.Write(pcmBuf); err != nil {
		return nil, fmt.Errorf("writing PCM: %w", err)
	}
	if err := enc.Close(); err != nil {
		return nil, fmt.Errorf("closing encoder: %w", err)
	}

	return buf.Bytes(), nil
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
		return 0, fmt.Errorf("seek before start")
	}
	s.pos = newPos
	return int64(newPos), nil
}
