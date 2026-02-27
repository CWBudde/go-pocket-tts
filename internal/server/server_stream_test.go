package server_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/example/go-pocket-tts/internal/server"
	"github.com/example/go-pocket-tts/internal/tts"
)

// stubStreamingSynthesizer implements server.StreamingSynthesizer for tests.
type stubStreamingSynthesizer struct {
	chunks []tts.PCMChunk
	err    error
	delay  time.Duration // per-chunk delay to simulate generation time
}

func (s *stubStreamingSynthesizer) SynthesizeStream(ctx context.Context, _, _ string, out chan<- tts.PCMChunk) error {
	defer close(out)

	for _, chunk := range s.chunks {
		if s.delay > 0 {
			select {
			case <-time.After(s.delay):
			case <-ctx.Done():
				return ctx.Err()
			}
		}

		select {
		case out <- chunk:
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return s.err
}

func postStreamJSON(h http.Handler, body any) *httptest.ResponseRecorder {
	b, err := json.Marshal(body)
	if err != nil {
		panic(err)
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/tts/stream", bytes.NewReader(b))
	h.ServeHTTP(rec, req)

	return rec
}

func TestTTSStream_NoStreamer_Returns501(t *testing.T) {
	h := server.NewHandler(&stubSynthesizer{}, &stubVoiceLister{})
	rec := postStreamJSON(h, map[string]string{"text": "hello"})

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("want 501, got %d", rec.Code)
	}
}

func TestTTSStream_MethodNotAllowed(t *testing.T) {
	streamer := &stubStreamingSynthesizer{}
	h := server.NewHandler(&stubSynthesizer{}, &stubVoiceLister{}, server.WithStreamer(streamer))

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/tts/stream", nil)
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("want 405, got %d", rec.Code)
	}
}

func TestTTSStream_EmptyText_Returns400(t *testing.T) {
	streamer := &stubStreamingSynthesizer{}
	h := server.NewHandler(&stubSynthesizer{}, &stubVoiceLister{}, server.WithStreamer(streamer))
	rec := postStreamJSON(h, map[string]string{"text": ""})

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", rec.Code)
	}
}

func TestTTSStream_ProducesWAVWithChunkedPCM(t *testing.T) {
	samples := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	streamer := &stubStreamingSynthesizer{
		chunks: []tts.PCMChunk{
			{Samples: samples[:3], ChunkIndex: 0, Final: false},
			{Samples: samples[3:], ChunkIndex: 1, Final: true},
		},
	}
	h := server.NewHandler(&stubSynthesizer{}, &stubVoiceLister{}, server.WithStreamer(streamer))
	rec := postStreamJSON(h, map[string]string{"text": "hello world"})

	if rec.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", rec.Code)
	}

	if ct := rec.Header().Get("Content-Type"); ct != "audio/wav" {
		t.Errorf("Content-Type = %q; want audio/wav", ct)
	}

	body := rec.Body.Bytes()
	// Should have 44-byte WAV header + 5 samples * 2 bytes = 54 bytes total
	expectedLen := 44 + len(samples)*2
	if len(body) != expectedLen {
		t.Fatalf("body length = %d; want %d", len(body), expectedLen)
	}

	// Verify RIFF header
	if string(body[0:4]) != "RIFF" {
		t.Error("missing RIFF marker")
	}

	// Verify data follows header — check first sample
	pcmStart := 44
	got := int16(binary.LittleEndian.Uint16(body[pcmStart : pcmStart+2]))

	want := int16(math.Round(0.1 * 32767))
	if abs16(got-want) > 1 {
		t.Errorf("first PCM sample = %d; want ~%d", got, want)
	}
}

func TestTTSStream_SemaphoreEnforced(t *testing.T) {
	// Use a streamer with delay to hold the worker slot
	streamer := &stubStreamingSynthesizer{
		chunks: []tts.PCMChunk{{Samples: []float32{0.1}, Final: true}},
		delay:  200 * time.Millisecond,
	}
	h := server.NewHandler(
		&stubSynthesizer{},
		&stubVoiceLister{},
		server.WithStreamer(streamer),
		server.WithWorkers(1),
	)

	var wg sync.WaitGroup
	results := make([]*httptest.ResponseRecorder, 2)

	for i := range 2 {
		wg.Add(1)

		go func(idx int) {
			defer wg.Done()

			results[idx] = postStreamJSON(h, map[string]string{"text": "hello"})
		}(i)
	}

	wg.Wait()

	// Both should succeed (second waits for first)
	for i, rec := range results {
		if rec.Code != http.StatusOK {
			t.Errorf("request[%d] status = %d; want 200", i, rec.Code)
		}
	}
}

func TestTTSStream_TextTooLarge(t *testing.T) {
	streamer := &stubStreamingSynthesizer{}
	h := server.NewHandler(
		&stubSynthesizer{},
		&stubVoiceLister{},
		server.WithStreamer(streamer),
		server.WithMaxTextBytes(10),
	)
	rec := postStreamJSON(h, map[string]string{"text": "this text is way too long"})

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("want 413, got %d", rec.Code)
	}
}

func abs16(v int16) int16 {
	if v < 0 {
		return -v
	}

	return v
}
