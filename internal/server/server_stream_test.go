package server_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/server"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
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
	// 44-byte WAV header + 5 samples, then upstream's 0.2 s of trailing
	// silence (4800 samples at 24 kHz), 2 bytes each.
	const silence = 4800

	expectedLen := 44 + (len(samples)+silence)*2
	if len(body) != expectedLen {
		t.Fatalf("body length = %d; want %d", len(body), expectedLen)
	}

	for i := 44 + len(samples)*2; i < len(body); i++ {
		if body[i] != 0 {
			t.Fatalf("byte %d = %d; want trailing silence", i, body[i])
		}
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

func TestTTSStream_ErrorSkipsTrailingSilence(t *testing.T) {
	samples := []float32{0.1, 0.2}
	streamer := &stubStreamingSynthesizer{
		chunks: []tts.PCMChunk{{Samples: samples}},
		err:    errors.New("generation failed"),
	}
	h := server.NewHandler(&stubSynthesizer{}, &stubVoiceLister{}, server.WithStreamer(streamer))
	rec := postStreamJSON(h, map[string]string{"text": "hello"})

	// The header is already sent; a failed stream just ends, without the
	// silence that marks a finished one.
	if got, want := rec.Body.Len(), 44+len(samples)*2; got != want {
		t.Fatalf("body length = %d; want %d", got, want)
	}
}

// steppingStreamer generates one chunk per step until steps run out or ctx
// is cancelled, recording which request (its text) ran each step.
type steppingStreamer struct {
	steps int
	delay time.Duration

	mu      sync.Mutex
	ran     []string
	stopped map[string]chan struct{}
}

func (s *steppingStreamer) SynthesizeStream(ctx context.Context, text, _ string, out chan<- tts.PCMChunk) error {
	defer close(out)
	defer close(s.done(text))

	for i := range s.steps {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		s.mu.Lock()
		s.ran = append(s.ran, text)
		s.mu.Unlock()

		select {
		case out <- tts.PCMChunk{Samples: []float32{0.5, 0.5}, ChunkIndex: i}:
		case <-ctx.Done():
			return ctx.Err()
		}

		select {
		case <-time.After(s.delay):
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return nil
}

func (s *steppingStreamer) done(text string) chan struct{} {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped == nil {
		s.stopped = map[string]chan struct{}{}
	}

	if s.stopped[text] == nil {
		s.stopped[text] = make(chan struct{})
	}

	return s.stopped[text]
}

func (s *steppingStreamer) stepsOf(text string) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	n := 0

	for _, r := range s.ran {
		if r == text {
			n++
		}
	}

	return n
}

// Upstream test_client_disconnect_stops_the_generation: once the client goes
// away the generation stops early, and the next request runs without steps
// from the abandoned one.
func TestTTSStream_ClientDisconnectStopsGeneration(t *testing.T) {
	streamer := &steppingStreamer{steps: 1000, delay: 2 * time.Millisecond}

	ts := httptest.NewServer(server.NewHandler(
		&stubSynthesizer{},
		&stubVoiceLister{},
		server.WithStreamer(streamer),
		server.WithWorkers(1),
	))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	resp := postStream(ctx, t, ts.URL, "first")

	// The header and the first chunk arrive, then the client disconnects.
	_, err := io.ReadFull(resp.Body, make([]byte, 44+4))
	if err != nil {
		t.Fatalf("read first chunk: %v", err)
	}

	cancel()
	resp.Body.Close()

	select {
	case <-streamer.done("first"):
	case <-time.After(5 * time.Second):
		t.Fatal("generation still running 5 s after the client disconnected")
	}

	stepsWhenStopped := streamer.stepsOf("first")
	if stepsWhenStopped >= streamer.steps {
		t.Fatalf("first request ran all %d steps; want it stopped early", stepsWhenStopped)
	}

	// A second request has the streamer to itself.
	streamer.steps = 3

	second := postStream(context.Background(), t, ts.URL, "second")
	defer second.Body.Close()

	body, err := io.ReadAll(second.Body)
	if err != nil {
		t.Fatalf("read second stream: %v", err)
	}

	if got, want := len(body), 44+(3*2+4800)*2; got != want {
		t.Errorf("second stream = %d bytes; want %d (3 chunks + trailing silence)", got, want)
	}

	if got := streamer.stepsOf("first"); got != stepsWhenStopped {
		t.Errorf("first request ran %d more steps after it stopped", got-stepsWhenStopped)
	}
}

func postStream(ctx context.Context, t *testing.T, baseURL, text string) *http.Response {
	t.Helper()

	b, err := json.Marshal(map[string]string{"text": text})
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/tts/stream", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /tts/stream: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("POST /tts/stream: status %d", resp.StatusCode)
	}

	return resp
}

func abs16(v int16) int16 {
	if v < 0 {
		return -v
	}

	return v
}
