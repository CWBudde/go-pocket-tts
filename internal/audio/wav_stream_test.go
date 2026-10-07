package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestWriteWAVHeaderStreaming_Size(t *testing.T) {
	var buf bytes.Buffer

	n, err := WriteWAVHeaderStreaming(&buf)
	if err != nil {
		t.Fatalf("WriteWAVHeaderStreaming error: %v", err)
	}

	if n != 44 {
		t.Fatalf("wrote %d bytes; want 44", n)
	}

	if buf.Len() != 44 {
		t.Fatalf("buffer length %d; want 44", buf.Len())
	}
}

func TestWriteWAVHeaderStreaming_Markers(t *testing.T) {
	var buf bytes.Buffer

	_, err := WriteWAVHeaderStreaming(&buf)
	if err != nil {
		t.Fatal(err)
	}

	hdr := buf.Bytes()

	if string(hdr[0:4]) != "RIFF" {
		t.Errorf("RIFF marker = %q; want RIFF", hdr[0:4])
	}

	if string(hdr[8:12]) != "WAVE" {
		t.Errorf("WAVE marker = %q; want WAVE", hdr[8:12])
	}

	if string(hdr[12:16]) != "fmt " {
		t.Errorf("fmt marker = %q; want 'fmt '", hdr[12:16])
	}

	if string(hdr[36:40]) != "data" {
		t.Errorf("data marker = %q; want data", hdr[36:40])
	}
}

func TestWriteWAVHeaderStreaming_UnknownLength(t *testing.T) {
	var buf bytes.Buffer

	_, err := WriteWAVHeaderStreaming(&buf)
	if err != nil {
		t.Fatal(err)
	}

	hdr := buf.Bytes()

	riffSize := binary.LittleEndian.Uint32(hdr[4:8])
	if riffSize != 0xFFFFFFFF {
		t.Errorf("RIFF size = 0x%08X; want 0xFFFFFFFF", riffSize)
	}

	dataSize := binary.LittleEndian.Uint32(hdr[40:44])
	if dataSize != 0xFFFFFFFF {
		t.Errorf("data size = 0x%08X; want 0xFFFFFFFF", dataSize)
	}
}

func TestWriteWAVHeaderStreaming_Format(t *testing.T) {
	var buf bytes.Buffer

	_, err := WriteWAVHeaderStreaming(&buf)
	if err != nil {
		t.Fatal(err)
	}

	hdr := buf.Bytes()

	audioFormat := binary.LittleEndian.Uint16(hdr[20:22])
	if audioFormat != 1 {
		t.Errorf("audio format = %d; want 1 (PCM)", audioFormat)
	}

	channels := binary.LittleEndian.Uint16(hdr[22:24])
	if channels != 1 {
		t.Errorf("channels = %d; want 1", channels)
	}

	sampleRate := binary.LittleEndian.Uint32(hdr[24:28])
	if sampleRate != 24000 {
		t.Errorf("sample rate = %d; want 24000", sampleRate)
	}

	bitsPerSample := binary.LittleEndian.Uint16(hdr[34:36])
	if bitsPerSample != 16 {
		t.Errorf("bits per sample = %d; want 16", bitsPerSample)
	}
}

func TestWritePCM16Samples_Encoding(t *testing.T) {
	samples := []float32{0.0, 1.0, -1.0, 0.5, -0.5}
	var buf bytes.Buffer

	n, err := WritePCM16Samples(&buf, samples)
	if err != nil {
		t.Fatalf("WritePCM16Samples error: %v", err)
	}

	if n != len(samples)*2 {
		t.Fatalf("wrote %d bytes; want %d", n, len(samples)*2)
	}

	data := buf.Bytes()
	for i, want := range []int16{0, 32767, -32767, 16383, -16383} {
		got := int16(binary.LittleEndian.Uint16(data[i*2 : i*2+2]))
		if abs16(got-want) > 1 {
			t.Errorf("sample[%d] = %d; want ~%d", i, got, want)
		}
	}
}

func TestWritePCM16Samples_Clamping(t *testing.T) {
	samples := []float32{2.0, -3.0}

	var buf bytes.Buffer

	_, err := WritePCM16Samples(&buf, samples)
	if err != nil {
		t.Fatal(err)
	}

	data := buf.Bytes()

	got0 := int16(binary.LittleEndian.Uint16(data[0:2]))
	if got0 != 32767 {
		t.Errorf("clamped +2.0 = %d; want 32767", got0)
	}

	got1 := int16(binary.LittleEndian.Uint16(data[2:4]))
	if got1 != -32767 {
		t.Errorf("clamped -3.0 = %d; want -32767", got1)
	}
}

func TestWritePCM16Samples_Empty(t *testing.T) {
	var buf bytes.Buffer

	n, err := WritePCM16Samples(&buf, nil)
	if err != nil {
		t.Fatalf("WritePCM16Samples(nil) error: %v", err)
	}

	if n != 0 {
		t.Errorf("wrote %d bytes for nil; want 0", n)
	}
}

func TestWritePCM16Samples_NaN(t *testing.T) {
	samples := []float32{float32(math.NaN())}
	var buf bytes.Buffer

	_, err := WritePCM16Samples(&buf, samples)
	if err != nil {
		t.Fatalf("WritePCM16Samples(NaN) error: %v", err)
	}
	// NaN comparisons return false, so math.Max/Min will clamp to 0
	got := int16(binary.LittleEndian.Uint16(buf.Bytes()[0:2]))
	// NaN clamped — exact value is implementation-defined, just verify no panic
	_ = got
}

func abs16(v int16) int16 {
	if v < 0 {
		return -v
	}

	return v
}

// pcmWAV returns a 16-bit mono 24 kHz WAV whose RIFF and data chunk sizes
// are riffSize and dataSize, followed by pcmBytes bytes of PCM and an
// optional LIST chunk before the data chunk.
func pcmWAV(riffSize, dataSize uint32, pcmBytes int, withList bool) []byte {
	var buf bytes.Buffer

	_, _ = WriteWAVHeaderStreaming(&buf)
	hdr := buf.Bytes()
	binary.LittleEndian.PutUint32(hdr[4:8], riffSize)

	out := append([]byte(nil), hdr[:36]...)
	if withList {
		out = append(out, "LIST"...)
		out = binary.LittleEndian.AppendUint32(out, 4)
		out = append(out, "INFO"...)
	}

	out = append(out, "data"...)
	out = binary.LittleEndian.AppendUint32(out, dataSize)

	for i := range pcmBytes {
		out = append(out, byte(i))
	}

	return out
}

func TestFixStreamedWAVSizes(t *testing.T) {
	const placeholder = 1_000_000_000 * 2 // pocket-tts 3.x: setnframes(1e9), 16-bit mono

	tests := []struct {
		name               string
		in                 []byte
		wantRIFF, wantData uint32
		dataOffset         int
	}{
		{"pocket-tts 3.x placeholder", pcmWAV(36+placeholder, placeholder, 200, false), 36 + 200, 200, 40},
		{"streaming 0xFFFFFFFF", pcmWAV(0xFFFFFFFF, 0xFFFFFFFF, 64, false), 36 + 64, 64, 40},
		{"LIST chunk before data", pcmWAV(48+placeholder, placeholder, 10, true), 48 + 10, 10, 52},
		{"odd byte count", pcmWAV(36+placeholder, placeholder, 7, false), 36 + 7, 7, 40},
		{"correct sizes stay", pcmWAV(36+100, 100, 100, false), 36 + 100, 100, 40},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FixStreamedWAVSizes(tt.in)

			if len(got) != len(tt.in) {
				t.Fatalf("length %d, want %d", len(got), len(tt.in))
			}

			if r := binary.LittleEndian.Uint32(got[4:8]); r != tt.wantRIFF {
				t.Errorf("RIFF size %d, want %d", r, tt.wantRIFF)
			}

			if d := binary.LittleEndian.Uint32(got[tt.dataOffset : tt.dataOffset+4]); d != tt.wantData {
				t.Errorf("data size %d, want %d", d, tt.wantData)
			}
		})
	}
}

func TestFixStreamedWAVSizes_LeavesNonWAVAlone(t *testing.T) {
	for _, in := range [][]byte{nil, []byte("RIFF"), []byte("not a wav file at all, just some text bytes!!")} {
		want := append([]byte(nil), in...)
		if got := FixStreamedWAVSizes(in); !bytes.Equal(got, want) {
			t.Errorf("FixStreamedWAVSizes(%q) = %q, want unchanged", want, got)
		}
	}
}

func TestFixStreamedWAVSizes_DecodesToActualLength(t *testing.T) {
	samples := []float32{0.1, -0.2, 0.3, -0.4, 0.5}

	var buf bytes.Buffer

	_, _ = WriteWAVHeaderStreaming(&buf)
	_, _ = WritePCM16Samples(&buf, samples)

	data := FixStreamedWAVSizes(buf.Bytes())

	got, err := DecodeWAV(data)
	if err != nil {
		t.Fatalf("DecodeWAV: %v", err)
	}

	if len(got) != len(samples) {
		t.Errorf("decoded %d samples, want %d", len(got), len(samples))
	}
}
