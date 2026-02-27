package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/example/go-pocket-tts/internal/server"
	"github.com/example/go-pocket-tts/internal/tts"
)

// stubSynthesizer implements server.Synthesizer for tests.
type stubSynthesizer struct {
	wav []byte
	err error
}

func (s *stubSynthesizer) Synthesize(_ context.Context, _, _ string) ([]byte, error) {
	return s.wav, s.err
}

// stubVoiceLister implements server.VoiceLister for tests.
type stubVoiceLister struct {
	voices []tts.Voice
}

func (v *stubVoiceLister) ListVoices() []tts.Voice {
	return v.voices
}

func newTestHandler(synth server.Synthesizer, voices server.VoiceLister) http.Handler {
	return server.NewHandler(synth, voices)
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

func TestHealth_Returns200WithStatusOK(t *testing.T) {
	h := newTestHandler(&stubSynthesizer{}, &stubVoiceLister{})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", rec.Code)
	}

	var body map[string]string
	err := json.NewDecoder(rec.Body).Decode(&body)
	if err != nil {
		t.Fatalf("decode body: %v", err)
	}

	if body["status"] != "ok" {
		t.Errorf("want status=ok, got %q", body["status"])
	}

	if _, ok := body["version"]; !ok {
		t.Error("want version field in response")
	}
}

// ---------------------------------------------------------------------------
// GET /voices
// ---------------------------------------------------------------------------

func TestVoices_ReturnsJSONArray(t *testing.T) {
	voices := []tts.Voice{
		{ID: "en-default", Path: "voices/en-default.safetensors", License: "cc-by-4.0"},
		{ID: "de-default", Path: "voices/de-default.safetensors", License: "cc-by-4.0"},
	}
	h := newTestHandler(&stubSynthesizer{}, &stubVoiceLister{voices: voices})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/voices", nil)
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", rec.Code)
	}

	var got []tts.Voice
	err := json.NewDecoder(rec.Body).Decode(&got)
	if err != nil {
		t.Fatalf("decode body: %v", err)
	}

	if len(got) != 2 {
		t.Fatalf("want 2 voices, got %d", len(got))
	}

	if got[0].ID != "en-default" || got[1].ID != "de-default" {
		t.Errorf("unexpected voice IDs: %v", got)
	}
}

func TestVoices_ReturnsEmptyArrayWhenNoVoices(t *testing.T) {
	h := newTestHandler(&stubSynthesizer{}, &stubVoiceLister{voices: []tts.Voice{}})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/voices", nil)
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", rec.Code)
	}

	var got []tts.Voice
	err := json.NewDecoder(rec.Body).Decode(&got)
	if err != nil {
		t.Fatalf("decode body: %v", err)
	}

	if len(got) != 0 {
		t.Errorf("want empty array, got %v", got)
	}
}

// ---------------------------------------------------------------------------
// POST /tts
// ---------------------------------------------------------------------------

func TestTTS_ReturnsMissingBodyAs400(t *testing.T) {
	h := newTestHandler(&stubSynthesizer{}, &stubVoiceLister{})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/tts", nil)
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", rec.Code)
	}

	var body map[string]string
	err := json.NewDecoder(rec.Body).Decode(&body)
	if err != nil {
		t.Fatalf("decode error body: %v", err)
	}

	if body["error"] == "" {
		t.Error("want non-empty error field")
	}
}

func TestTTS_ReturnsEmptyTextAs400(t *testing.T) {
	h := newTestHandler(&stubSynthesizer{}, &stubVoiceLister{})

	body := bytes.NewBufferString(`{"text":"","voice":"en-default"}`)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/tts", body)
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", rec.Code)
	}
}

func TestTTS_ReturnsWAVBytesOnSuccess(t *testing.T) {
	fakeWAV := []byte("RIFF\x00\x00\x00\x00WAVEfmt ")
	synth := &stubSynthesizer{wav: fakeWAV}
	h := newTestHandler(synth, &stubVoiceLister{})

	body := bytes.NewBufferString(`{"text":"Hello world.","voice":"en-default"}`)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/tts", body)
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("want 200, got %d (body: %s)", rec.Code, rec.Body.String())
	}

	if ct := rec.Header().Get("Content-Type"); ct != "audio/wav" {
		t.Errorf("want Content-Type audio/wav, got %q", ct)
	}

	if !bytes.Equal(rec.Body.Bytes(), fakeWAV) {
		t.Errorf("want WAV bytes back, got %d bytes", rec.Body.Len())
	}
}

func TestTTS_SynthesizerErrorReturns500(t *testing.T) {
	synth := &stubSynthesizer{err: errSynthFailed}
	h := newTestHandler(synth, &stubVoiceLister{})

	body := bytes.NewBufferString(`{"text":"Hello.","voice":"en-default"}`)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/tts", body)
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("want 500, got %d", rec.Code)
	}

	var errBody map[string]string
	err := json.NewDecoder(rec.Body).Decode(&errBody)
	if err != nil {
		t.Fatalf("decode error body: %v", err)
	}

	if errBody["error"] == "" {
		t.Error("want non-empty error field")
	}
}

var errSynthFailed = &synthError{"synthesis failed"}

type synthError struct{ msg string }

func (e *synthError) Error() string { return e.msg }
