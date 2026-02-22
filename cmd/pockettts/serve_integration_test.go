//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/example/go-pocket-tts/internal/audio"
	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/server"
	"github.com/example/go-pocket-tts/internal/testutil"
	"github.com/example/go-pocket-tts/internal/tts"
)

// ---------------------------------------------------------------------------
// TestServe_HealthEndpoint
// ---------------------------------------------------------------------------

// TestServe_HealthEndpoint starts a real httptest server and asserts that
// GET /health returns 200 with {"status":"ok"}.
func TestServe_HealthEndpoint(t *testing.T) {
	ts := httptest.NewServer(server.NewHandler(
		&noopSynthesizer{},
		&staticVoices{},
	))
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/health") //nolint:noctx
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode /health body: %v", err)
	}
	if body["status"] != "ok" {
		t.Errorf("want status=ok, got %q", body["status"])
	}
}

// ---------------------------------------------------------------------------
// TestServe_VoicesEndpoint
// ---------------------------------------------------------------------------

// TestServe_VoicesEndpoint asserts that GET /voices returns a JSON array
// containing at least the fixture voice ID.
func TestServe_VoicesEndpoint(t *testing.T) {
	fixtureVoice := tts.Voice{ID: "fixture", Path: "fixture.safetensors", License: "cc-by-4.0"}
	ts := httptest.NewServer(server.NewHandler(
		&noopSynthesizer{},
		&staticVoices{voices: []tts.Voice{fixtureVoice}},
	))
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/voices") //nolint:noctx
	if err != nil {
		t.Fatalf("GET /voices: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var voices []tts.Voice
	if err := json.NewDecoder(resp.Body).Decode(&voices); err != nil {
		t.Fatalf("decode /voices body: %v", err)
	}
	if len(voices) == 0 {
		t.Fatal("expected at least one voice in /voices response")
	}
	found := false
	for _, v := range voices {
		if v.ID == fixtureVoice.ID {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("fixture voice %q not found in /voices response: %v", fixtureVoice.ID, voices)
	}
}

// ---------------------------------------------------------------------------
// TestServe_TTSEndpoint_CLI
// ---------------------------------------------------------------------------

// TestServe_TTSEndpoint_CLI posts to /tts via a real CLI synthesizer and
// asserts a valid WAV response with Content-Type: audio/wav.
func TestServe_TTSEndpoint_CLI(t *testing.T) {
	testutil.RequirePocketTTS(t)
	voice := synthTestVoice(t)

	ts := httptest.NewServer(server.NewHandler(
		&pocketTTSSynthesizer{defaultVoice: voice},
		&staticVoices{},
		server.WithRequestTimeout(120*time.Second),
	))
	defer ts.Close()

	body := serveJSONBody(t, map[string]any{"text": "Hello.", "voice": voice})
	resp, err := http.Post(ts.URL+"/tts", "application/json", body) //nolint:noctx
	if err != nil {
		t.Fatalf("POST /tts: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("want 200, got %d: %s", resp.StatusCode, data)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "audio/wav" {
		t.Errorf("want Content-Type audio/wav, got %q", ct)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read /tts body: %v", err)
	}
	testutil.AssertValidWAV(t, data)
}

// ---------------------------------------------------------------------------
// TestServe_TTSEndpoint_Native
// ---------------------------------------------------------------------------

// TestServe_TTSEndpoint_Native posts to /tts via the native ONNX backend.
// Skips when ONNX Runtime is unavailable.
func TestServe_TTSEndpoint_Native(t *testing.T) {
	testutil.RequireONNXRuntime(t)

	cfg := config.DefaultConfig()
	svc, err := tts.NewService(cfg)
	if err != nil {
		t.Skipf("native TTS service unavailable: %v", err)
	}

	ts := httptest.NewServer(server.NewHandler(
		&nativeTTSSynthesizer{svc: svc},
		&staticVoices{},
		server.WithRequestTimeout(120*time.Second),
	))
	defer ts.Close()

	body := serveJSONBody(t, map[string]any{"text": "Hello.", "voice": ""})
	resp, err := http.Post(ts.URL+"/tts", "application/json", body) //nolint:noctx
	if err != nil {
		t.Fatalf("POST /tts: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("want 200, got %d: %s", resp.StatusCode, data)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "audio/wav" {
		t.Errorf("want Content-Type audio/wav, got %q", ct)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read /tts body: %v", err)
	}
	testutil.AssertValidWAV(t, data)
}

// ---------------------------------------------------------------------------
// TestServe_TTSEndpoint_EmptyText
// ---------------------------------------------------------------------------

// TestServe_TTSEndpoint_EmptyText asserts that POST /tts with an empty text
// field returns 400 and a JSON error body.
func TestServe_TTSEndpoint_EmptyText(t *testing.T) {
	ts := httptest.NewServer(server.NewHandler(
		&noopSynthesizer{},
		&staticVoices{},
	))
	defer ts.Close()

	body := serveJSONBody(t, map[string]any{"text": "", "voice": "any"})
	resp, err := http.Post(ts.URL+"/tts", "application/json", body) //nolint:noctx
	if err != nil {
		t.Fatalf("POST /tts: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
	var errBody map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&errBody); err != nil {
		t.Fatalf("decode error body: %v", err)
	}
	if errBody["error"] == "" {
		t.Error("want non-empty error field in response body")
	}
}

// ---------------------------------------------------------------------------
// TestServe_TTSEndpoint_OversizedText
// ---------------------------------------------------------------------------

// TestServe_TTSEndpoint_OversizedText asserts that POST /tts with text
// exceeding WithMaxTextBytes returns 413.
func TestServe_TTSEndpoint_OversizedText(t *testing.T) {
	const limit = 20
	ts := httptest.NewServer(server.NewHandler(
		&noopSynthesizer{},
		&staticVoices{},
		server.WithMaxTextBytes(limit),
	))
	defer ts.Close()

	oversized := strings.Repeat("x", limit+1)
	body := serveJSONBody(t, map[string]any{"text": oversized, "voice": "any"})
	resp, err := http.Post(ts.URL+"/tts", "application/json", body) //nolint:noctx
	if err != nil {
		t.Fatalf("POST /tts: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("want 413, got %d", resp.StatusCode)
	}
	var errBody map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&errBody); err != nil {
		t.Fatalf("decode error body: %v", err)
	}
	if errBody["error"] == "" {
		t.Error("want non-empty error field in response body")
	}
}

// ---------------------------------------------------------------------------
// TestServe_ConcurrentRequests
// ---------------------------------------------------------------------------

// TestServe_ConcurrentRequests fires N concurrent POST /tts requests (N ==
// worker pool size) and asserts all return 200 within a bounded time window.
func TestServe_ConcurrentRequests(t *testing.T) {
	testutil.RequirePocketTTS(t)
	voice := synthTestVoice(t)

	const workers = 2
	ts := httptest.NewServer(server.NewHandler(
		&pocketTTSSynthesizer{defaultVoice: voice},
		&staticVoices{},
		server.WithWorkers(workers),
		server.WithRequestTimeout(120*time.Second),
	))
	defer ts.Close()

	var wg sync.WaitGroup
	codes := make([]int, workers)
	start := time.Now()

	for i := range workers {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			body := serveJSONBody(t, map[string]any{"text": "Hello.", "voice": voice})
			resp, err := http.Post(ts.URL+"/tts", "application/json", body) //nolint:noctx
			if err != nil {
				t.Errorf("request %d: POST /tts: %v", idx, err)
				codes[idx] = -1
				return
			}
			defer resp.Body.Close()
			io.Copy(io.Discard, resp.Body) //nolint:errcheck
			codes[idx] = resp.StatusCode
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)

	for i, code := range codes {
		if code != http.StatusOK {
			t.Errorf("request %d: want 200, got %d", i, code)
		}
	}
	// All concurrent requests must complete within 5 minutes.
	if elapsed > 5*time.Minute {
		t.Errorf("concurrent requests took too long: %v", elapsed)
	}
}

// ---------------------------------------------------------------------------
// Synthesizer adapters (integration-test-scoped)
// ---------------------------------------------------------------------------

// noopSynthesizer returns a minimal valid WAV without invoking any real backend.
type noopSynthesizer struct{}

func (n *noopSynthesizer) Synthesize(_ context.Context, _, _ string) ([]byte, error) {
	wav, err := audio.EncodeWAV([]float32{0.0, 0.0, 0.0})
	if err != nil {
		return nil, fmt.Errorf("noop encode: %w", err)
	}
	return wav, nil
}

// pocketTTSSynthesizer invokes the pocket-tts CLI subprocess.
type pocketTTSSynthesizer struct {
	defaultVoice string
}

func (p *pocketTTSSynthesizer) Synthesize(ctx context.Context, text, voice string) ([]byte, error) {
	if voice == "" {
		voice = p.defaultVoice
	}
	args := []string{"generate", "--text", "-", "--output-path", "-"}
	if voice != "" {
		args = append(args, "--voice", voice)
	}
	cmd := exec.CommandContext(ctx, "pocket-tts", args...)
	cmd.Stdin = strings.NewReader(text)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = io.Discard
	if err := cmd.Run(); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// nativeTTSSynthesizer wraps tts.Service as a server.Synthesizer.
type nativeTTSSynthesizer struct {
	svc *tts.Service
}

func (n *nativeTTSSynthesizer) Synthesize(_ context.Context, text, voice string) ([]byte, error) {
	samples, err := n.svc.Synthesize(text, voice)
	if err != nil {
		return nil, err
	}
	return audio.EncodeWAV(samples)
}

// staticVoices is a server.VoiceLister backed by a fixed slice.
type staticVoices struct {
	voices []tts.Voice
}

func (s *staticVoices) ListVoices() []tts.Voice {
	return append([]tts.Voice(nil), s.voices...)
}

// serveJSONBody encodes v as JSON and returns an io.Reader for use as a
// request body.
func serveJSONBody(t testing.TB, v any) io.Reader {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal JSON body: %v", err)
	}
	return bytes.NewReader(b)
}
