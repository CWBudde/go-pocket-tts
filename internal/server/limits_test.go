package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/example/go-pocket-tts/internal/server"
	"github.com/example/go-pocket-tts/internal/tts"
)

// ---------------------------------------------------------------------------
// Task 7.4 — request validation and limits
// ---------------------------------------------------------------------------

func TestTTS_OversizedTextRejectedAs413(t *testing.T) {
	h := server.NewHandler(
		&stubSynthesizer{},
		&stubVoiceLister{},
		server.WithMaxTextBytes(10),
	)

	bigText := strings.Repeat("x", 11)
	body := bytes.NewBufferString(`{"text":"` + bigText + `","voice":"en"}`)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/tts", body)
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("want 413, got %d", rec.Code)
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

func TestTTS_TextAtExactLimitIsAccepted(t *testing.T) {
	fakeWAV := []byte("RIFF")
	h := server.NewHandler(
		&stubSynthesizer{wav: fakeWAV},
		&stubVoiceLister{},
		server.WithMaxTextBytes(5),
	)

	body := bytes.NewBufferString(`{"text":"hello","voice":"en"}`)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/tts", body)
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("want 200 for exactly-limit text, got %d", rec.Code)
	}
}

func TestTTS_RequestTimeoutCancelsInFlight(t *testing.T) {
	// Synthesizer that blocks until its context is cancelled.
	blocked := make(chan struct{})
	synth := &blockingSynthesizer{blocked: blocked}

	h := server.NewHandler(
		synth,
		&stubVoiceLister{},
		server.WithRequestTimeout(20*time.Millisecond),
	)

	body := bytes.NewBufferString(`{"text":"Hello.","voice":"en"}`)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/tts", body)
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(rec, req)

	// Must return 504 or 408 (we test for a 5xx/4xx that signals timeout)
	if rec.Code != http.StatusGatewayTimeout && rec.Code != http.StatusRequestTimeout {
		t.Fatalf("want 504 or 408 on timeout, got %d", rec.Code)
	}
	var errBody map[string]string

	_ = json.NewDecoder(rec.Body).Decode(&errBody)
	if errBody["error"] == "" {
		t.Error("want non-empty error field")
	}
}

// ---------------------------------------------------------------------------
// Task 7.2 — worker pool / concurrency throttling
// ---------------------------------------------------------------------------

func TestTTS_ConcurrencyThrottling(t *testing.T) {
	const workers = 2
	const totalRequests = 5

	// Synthesizer that counts concurrent executions.
	var (
		mu         sync.Mutex
		peak       int
		current    int32
		releaseAll = make(chan struct{})
	)
	synth := &countingSynthesizer{
		onEnter: func() {
			n := int(atomic.AddInt32(&current, 1))

			mu.Lock()
			if n > peak {
				peak = n
			}
			mu.Unlock()
			<-releaseAll
		},
		onExit: func() { atomic.AddInt32(&current, -1) },
		wav:    []byte("RIFF"),
	}

	h := server.NewHandler(
		synth,
		&stubVoiceLister{},
		server.WithWorkers(workers),
	)

	var wg sync.WaitGroup

	codes := make([]int, totalRequests)
	for i := range totalRequests {
		wg.Add(1)

		go func(idx int) {
			defer wg.Done()

			body := bytes.NewBufferString(`{"text":"Hi.","voice":"en"}`)
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/tts", body)
			req.Header.Set("Content-Type", "application/json")
			h.ServeHTTP(rec, req)
			codes[idx] = rec.Code
		}(i)
	}

	// Give goroutines time to enter the synthesizer.
	time.Sleep(50 * time.Millisecond)
	close(releaseAll)
	wg.Wait()

	mu.Lock()
	got := peak
	mu.Unlock()

	if got > workers {
		t.Errorf("peak concurrency %d exceeded worker limit %d", got, workers)
	}

	for i, code := range codes {
		if code != http.StatusOK {
			t.Errorf("request %d: want 200, got %d", i, code)
		}
	}
}

func TestTTS_WaiterCancelledWhileThrottled(t *testing.T) {
	const workers = 1

	release := make(chan struct{})
	synth := &blockingSynthesizer{blocked: release}

	h := server.NewHandler(
		synth,
		&stubVoiceLister{},
		server.WithWorkers(workers),
	)

	// First request occupies the single worker slot.
	go func() {
		body := bytes.NewBufferString(`{"text":"First.","voice":"en"}`)
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/tts", body)
		h.ServeHTTP(rec, req)
	}()

	time.Sleep(20 * time.Millisecond)

	// Second request should be blocked waiting for a worker; cancel its context.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	body := bytes.NewBufferString(`{"text":"Second.","voice":"en"}`)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/tts", body).WithContext(ctx)
	h.ServeHTTP(rec, req)

	// The cancelled waiter must get a non-200 (503 or 499-like response).
	if rec.Code == http.StatusOK {
		t.Fatalf("expected non-200 when waiter context cancelled, got 200")
	}

	close(release) // unblock the first request
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// blockingSynthesizer blocks until blocked is closed (simulates a slow subprocess).
type blockingSynthesizer struct {
	blocked chan struct{}
	wav     []byte
}

func (b *blockingSynthesizer) Synthesize(ctx context.Context, _, _ string) ([]byte, error) {
	select {
	case <-b.blocked:
		return b.wav, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// countingSynthesizer calls onEnter/onExit around the synthesize call.
type countingSynthesizer struct {
	onEnter func()
	onExit  func()
	wav     []byte
}

func (c *countingSynthesizer) Synthesize(_ context.Context, _, _ string) ([]byte, error) {
	c.onEnter()
	defer c.onExit()

	return c.wav, nil
}

// stubVoiceLister is already defined in server_test.go (same package), reused here.
// stubSynthesizer is already defined in server_test.go (same package), reused here.
var (
	_ server.VoiceLister = (*stubVoiceLister)(nil)
	_ tts.Voice          = tts.Voice{}
)
