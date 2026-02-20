package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/example/go-pocket-tts/internal/server"
)

// capturingHandler captures all slog records during a test.
type capturingHandler struct {
	records []slog.Record
}

func (c *capturingHandler) Enabled(_ context.Context, _ slog.Level) bool { return true }
func (c *capturingHandler) Handle(_ context.Context, r slog.Record) error {
	c.records = append(c.records, r)
	return nil
}
func (c *capturingHandler) WithAttrs(attrs []slog.Attr) slog.Handler { return c }
func (c *capturingHandler) WithGroup(name string) slog.Handler       { return c }

func (c *capturingHandler) attrMap(idx int) map[string]any {
	m := make(map[string]any)
	c.records[idx].Attrs(func(a slog.Attr) bool {
		m[a.Key] = a.Value.Any()
		return true
	})
	return m
}

func TestTTS_LogsVoiceAndTextLen(t *testing.T) {
	cap := &capturingHandler{}
	logger := slog.New(cap)

	fakeWAV := []byte("RIFF\x00\x00\x00\x00WAVEfmt ")
	h := server.NewHandler(
		&stubSynthesizer{wav: fakeWAV},
		&stubVoiceLister{},
		server.WithLogger(logger),
	)

	body := bytes.NewBufferString(`{"text":"Hello world.","voice":"en-default"}`)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/tts", body)
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", rec.Code)
	}

	// Must have at least one log record for the request.
	if len(cap.records) == 0 {
		t.Fatal("want at least one log record, got none")
	}

	// Find the synthesis log record.
	var found bool
	for i := range cap.records {
		attrs := cap.attrMap(i)
		if _, ok := attrs["voice"]; ok {
			found = true
			if attrs["voice"] != "en-default" {
				t.Errorf("want voice=en-default, got %v", attrs["voice"])
			}
			if _, ok := attrs["text_len"]; !ok {
				t.Error("want text_len attribute in log record")
			}
			if _, ok := attrs["duration_ms"]; !ok {
				t.Error("want duration_ms attribute in log record")
			}
		}
	}
	if !found {
		t.Error("no log record contained a 'voice' attribute")
	}
}

func TestTTS_LogsStatusOnError(t *testing.T) {
	cap := &capturingHandler{}
	logger := slog.New(cap)

	h := server.NewHandler(
		&stubSynthesizer{err: errSynthFailed},
		&stubVoiceLister{},
		server.WithLogger(logger),
	)

	body := bytes.NewBufferString(`{"text":"Hello.","voice":"en"}`)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/tts", body)
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("want 500, got %d", rec.Code)
	}

	var foundError bool
	for i := range cap.records {
		attrs := cap.attrMap(i)
		if _, ok := attrs["error"]; ok {
			foundError = true
		}
	}
	if !foundError {
		t.Error("want a log record with an 'error' attribute on synthesis failure")
	}
}

func TestSetupLogger_LevelFromString(t *testing.T) {
	cases := []struct {
		level   string
		wantLvl slog.Level
	}{
		{"debug", slog.LevelDebug},
		{"info", slog.LevelInfo},
		{"warn", slog.LevelWarn},
		{"error", slog.LevelError},
		{"", slog.LevelInfo}, // default
	}

	for _, tc := range cases {
		t.Run(tc.level, func(t *testing.T) {
			lvl, err := server.ParseLogLevel(tc.level)
			if err != nil {
				t.Fatalf("ParseLogLevel(%q) error: %v", tc.level, err)
			}
			if lvl != tc.wantLvl {
				t.Errorf("ParseLogLevel(%q) = %v, want %v", tc.level, lvl, tc.wantLvl)
			}
		})
	}
}

func TestSetupLogger_InvalidLevelReturnsError(t *testing.T) {
	_, err := server.ParseLogLevel("verbose")
	if err == nil {
		t.Error("want error for unknown log level")
	}
}

// Ensure json.NewDecoder still compiles (avoids unused import warning).
var _ = json.NewDecoder
