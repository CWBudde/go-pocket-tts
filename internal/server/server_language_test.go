package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/server"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
)

// fakeRouter serves fixed backends; "broken" fails to load.
type fakeRouter struct {
	backends map[string]server.LanguageBackend
	voices   map[string]server.VoiceLister
	released atomic.Int32
}

func (f *fakeRouter) Acquire(_ context.Context, language string) (server.LanguageBackend, func(), error) {
	if language == "broken" {
		return server.LanguageBackend{}, nil, errors.New("model missing")
	}

	backend, ok := f.backends[language]
	if !ok {
		return server.LanguageBackend{}, nil, fmt.Errorf("%w: %q", server.ErrUnknownLanguage, language)
	}

	return backend, func() { f.released.Add(1) }, nil
}

// Voices serves the languages of backends (and "broken"), listing the
// voices given for them.
func (f *fakeRouter) Voices(language string) (server.VoiceLister, error) {
	if _, ok := f.backends[language]; !ok && language != "broken" {
		return nil, fmt.Errorf("%w: %q", server.ErrUnknownLanguage, language)
	}

	if voices, ok := f.voices[language]; ok {
		return voices, nil
	}

	return &stubVoiceLister{}, nil
}

func newLanguageHandler() (http.Handler, *fakeRouter) {
	router := &fakeRouter{
		backends: map[string]server.LanguageBackend{
			"":       {Synth: &stubSynthesizer{wav: []byte("english")}},
			"german": {Synth: &stubSynthesizer{wav: []byte("german")}, Streamer: &stubStreamingSynthesizer{}},
		},
		voices: map[string]server.VoiceLister{
			"":       &stubVoiceLister{voices: []tts.Voice{{ID: "alba"}}},
			"german": &stubVoiceLister{voices: []tts.Voice{{ID: "juergen"}}},
		},
	}

	return server.NewHandler(&stubSynthesizer{wav: []byte("unused")}, &stubVoiceLister{}, server.WithLanguages(router)), router
}

func postTTS(h http.Handler, path string, body map[string]string) *httptest.ResponseRecorder {
	b, err := json.Marshal(body)
	if err != nil {
		panic(err)
	}

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, bytes.NewReader(b)))

	return rec
}

func TestHandler_LanguageRoutesTTS(t *testing.T) {
	h, router := newLanguageHandler()

	for language, want := range map[string]string{"": "english", "german": "german"} {
		rec := postTTS(h, "/tts", map[string]string{"text": "hi", "language": language})
		if rec.Code != http.StatusOK || rec.Body.String() != want {
			t.Errorf("language %q: %d %q; want 200 %q", language, rec.Code, rec.Body.String(), want)
		}
	}

	if n := router.released.Load(); n != 2 {
		t.Errorf("released %d backends; want 2 (one per request)", n)
	}
}

func TestHandler_LanguageErrors(t *testing.T) {
	h, _ := newLanguageHandler()

	for language, want := range map[string]int{"french": http.StatusBadRequest, "broken": http.StatusInternalServerError} {
		for _, path := range []string{"/tts", "/tts/stream"} {
			rec := postTTS(h, path, map[string]string{"text": "hi", "language": language})
			if rec.Code != want {
				t.Errorf("%s language %q: %d; want %d", path, language, rec.Code, want)
			}
		}
	}

	rec := postTTS(h, "/tts", map[string]string{"text": "hi", "language": "french"})
	if !strings.Contains(rec.Body.String(), "language not served") {
		t.Errorf("unknown language body = %s; want the ErrUnknownLanguage message", rec.Body.String())
	}
}

func TestHandler_LanguageWithoutRouter(t *testing.T) {
	h := server.NewHandler(&stubSynthesizer{wav: []byte("wav")}, &stubVoiceLister{}, server.WithStreamer(&stubStreamingSynthesizer{}))

	for _, path := range []string{"/tts", "/tts/stream"} {
		rec := postTTS(h, path, map[string]string{"text": "hi", "language": "german"})
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s with a language and no router: %d; want 400", path, rec.Code)
		}
	}

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/voices?language=german", nil))

	if rec.Code != http.StatusBadRequest {
		t.Errorf("/voices?language= without a router: %d; want 400", rec.Code)
	}

	if rec := postTTS(h, "/tts", map[string]string{"text": "hi"}); rec.Code != http.StatusOK {
		t.Errorf("/tts without a language: %d; want 200", rec.Code)
	}
}

func TestHandler_VoicesByLanguage(t *testing.T) {
	h, _ := newLanguageHandler()

	for query, want := range map[string]string{"": "alba", "?language=german": "juergen"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/voices"+query, nil))

		var voices []tts.Voice

		err := json.NewDecoder(rec.Body).Decode(&voices)
		if err != nil || rec.Code != http.StatusOK || len(voices) != 1 || voices[0].ID != want {
			t.Errorf("/voices%s = %d %v, %v; want %s", query, rec.Code, voices, err, want)
		}
	}

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/voices?language=french", nil))

	if rec.Code != http.StatusBadRequest {
		t.Errorf("/voices?language=french: %d; want 400", rec.Code)
	}
}

func TestHandler_LanguageStream(t *testing.T) {
	h, router := newLanguageHandler()

	rec := postTTS(h, "/tts/stream", map[string]string{"text": "hi", "language": "german"})
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "audio/wav" {
		t.Errorf("german stream: %d %q; want 200 audio/wav", rec.Code, rec.Header().Get("Content-Type"))
	}

	// The startup backend here cannot stream.
	rec = postTTS(h, "/tts/stream", map[string]string{"text": "hi"})
	if rec.Code != http.StatusNotImplemented {
		t.Errorf("stream without a streamer: %d; want 501", rec.Code)
	}

	if n := router.released.Load(); n != 2 {
		t.Errorf("released %d backends; want 2", n)
	}
}

// countingRouter counts Acquire calls of an inner router.
type countingRouter struct {
	*fakeRouter

	acquired atomic.Int32
}

func (c *countingRouter) Acquire(ctx context.Context, language string) (server.LanguageBackend, func(), error) {
	c.acquired.Add(1)
	return c.fakeRouter.Acquire(ctx, language)
}

func TestHandler_LanguageLoadsOnlyWithAWorkerSlot(t *testing.T) {
	blocked := &blockingSynthesizer{blocked: make(chan struct{}), wav: []byte("english")}
	router := &countingRouter{fakeRouter: &fakeRouter{backends: map[string]server.LanguageBackend{
		"":       {Synth: blocked},
		"german": {Synth: &stubSynthesizer{wav: []byte("german")}},
	}}}
	h := server.NewHandler(&stubSynthesizer{}, &stubVoiceLister{}, server.WithLanguages(router), server.WithWorkers(1))

	first := make(chan *httptest.ResponseRecorder)
	go func() { first <- postTTS(h, "/tts", map[string]string{"text": "hi"}) }()

	for router.acquired.Load() == 0 {
		time.Sleep(time.Millisecond)
	}

	// The only worker is busy: a request for another language waits for it
	// before its model is acquired (and loaded).
	second := make(chan *httptest.ResponseRecorder)
	go func() { second <- postTTS(h, "/tts", map[string]string{"text": "hi", "language": "german"}) }()

	time.Sleep(50 * time.Millisecond)

	if n := router.acquired.Load(); n != 1 {
		t.Fatalf("Acquire called %d times while the only worker was busy; want 1", n)
	}

	// An unknown language is still rejected without waiting for a worker.
	if rec := postTTS(h, "/tts", map[string]string{"text": "hi", "language": "french"}); rec.Code != http.StatusBadRequest {
		t.Errorf("unknown language while busy: %d; want 400", rec.Code)
	}

	close(blocked.blocked)

	if rec := <-first; rec.Code != http.StatusOK {
		t.Errorf("first request: %d", rec.Code)
	}

	if rec := <-second; rec.Code != http.StatusOK || rec.Body.String() != "german" {
		t.Errorf("second request: %d %q; want 200 german", rec.Code, rec.Body.String())
	}
}
