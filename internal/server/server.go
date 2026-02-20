package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os/exec"
	"runtime/debug"
	"strings"
	"time"

	"github.com/example/go-pocket-tts/internal/audio"
	"github.com/example/go-pocket-tts/internal/config"
	"github.com/example/go-pocket-tts/internal/tts"
)

// ParseLogLevel converts a case-insensitive level string to slog.Level.
// An empty string returns slog.LevelInfo. Unknown strings return an error.
func ParseLogLevel(s string) (slog.Level, error) {
	switch strings.ToLower(s) {
	case "", "info":
		return slog.LevelInfo, nil
	case "debug":
		return slog.LevelDebug, nil
	case "warn", "warning":
		return slog.LevelWarn, nil
	case "error":
		return slog.LevelError, nil
	default:
		return slog.LevelInfo, fmt.Errorf("unknown log level %q (want debug|info|warn|error)", s)
	}
}

// Synthesizer produces WAV bytes from text and a voice ID.
type Synthesizer interface {
	Synthesize(ctx context.Context, text, voice string) ([]byte, error)
}

// VoiceLister returns the list of available voices.
type VoiceLister interface {
	ListVoices() []tts.Voice
}

// ---------------------------------------------------------------------------
// Functional options
// ---------------------------------------------------------------------------

type options struct {
	maxTextBytes   int
	workers        int
	requestTimeout time.Duration
	logger         *slog.Logger
}

func defaultOptions() options {
	return options{
		maxTextBytes:   4096,
		workers:        2,
		requestTimeout: 60 * time.Second,
		logger:         slog.Default(),
	}
}

// Option configures the HTTP handler.
type Option func(*options)

// WithMaxTextBytes sets the maximum allowed text length in bytes for POST /tts.
func WithMaxTextBytes(n int) Option {
	return func(o *options) { o.maxTextBytes = n }
}

// WithWorkers sets the maximum number of concurrent synthesis calls.
func WithWorkers(n int) Option {
	return func(o *options) { o.workers = n }
}

// WithRequestTimeout sets the per-request synthesis deadline.
func WithRequestTimeout(d time.Duration) Option {
	return func(o *options) { o.requestTimeout = d }
}

// WithLogger sets the slog.Logger used for request logging.
func WithLogger(l *slog.Logger) Option {
	return func(o *options) { o.logger = l }
}

// ---------------------------------------------------------------------------
// handler
// ---------------------------------------------------------------------------

// handler holds the dependencies needed to serve HTTP requests.
type handler struct {
	synth  Synthesizer
	voices VoiceLister
	opts   options
	sem    chan struct{} // semaphore for worker pool
	log    *slog.Logger
}

// NewHandler returns an http.Handler that serves /health, /voices, and POST /tts.
func NewHandler(synth Synthesizer, voices VoiceLister, optFns ...Option) http.Handler {
	opts := defaultOptions()
	for _, fn := range optFns {
		fn(&opts)
	}

	h := &handler{
		synth:  synth,
		voices: voices,
		opts:   opts,
		log:    opts.logger,
	}
	if opts.workers > 0 {
		h.sem = make(chan struct{}, opts.workers)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", h.handleHealth)
	mux.HandleFunc("/voices", h.handleVoices)
	mux.HandleFunc("/tts", h.handleTTS)
	return mux
}

func buildVersion() string {
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" {
		return info.Main.Version
	}
	return "dev"
}

func (h *handler) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "ok",
		"version": buildVersion(),
	})
}

func (h *handler) handleVoices(w http.ResponseWriter, _ *http.Request) {
	voices := h.voices.ListVoices()
	if voices == nil {
		voices = []tts.Voice{}
	}
	writeJSON(w, http.StatusOK, voices)
}

type ttsRequest struct {
	Text  string `json:"text"`
	Voice string `json:"voice"`
	Chunk bool   `json:"chunk"`
}

func (h *handler) handleTTS(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	if r.Body == nil {
		writeError(w, http.StatusBadRequest, "request body is required")
		return
	}

	var req ttsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	if req.Text == "" {
		writeError(w, http.StatusBadRequest, "text field is required")
		return
	}

	if len(req.Text) > h.opts.maxTextBytes {
		writeError(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("text exceeds maximum size of %d bytes", h.opts.maxTextBytes))
		return
	}

	// Acquire a worker slot — honour context cancellation while waiting.
	// In native backend mode, worker throttling can be disabled (sem == nil).
	if h.sem != nil {
		select {
		case h.sem <- struct{}{}:
			// slot acquired
		case <-r.Context().Done():
			writeError(w, http.StatusServiceUnavailable, "request cancelled while waiting for worker")
			return
		}
		defer func() { <-h.sem }()
	}

	// Apply per-request timeout.
	ctx, cancel := context.WithTimeout(r.Context(), h.opts.requestTimeout)
	defer cancel()

	start := time.Now()
	wav, err := h.synth.Synthesize(ctx, req.Text, req.Voice)
	durationMS := time.Since(start).Milliseconds()

	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			h.log.WarnContext(r.Context(), "synthesis timed out",
				slog.String("voice", req.Voice),
				slog.Int("text_len", len(req.Text)),
				slog.Int64("duration_ms", durationMS),
				slog.String("error", err.Error()),
			)
			writeError(w, http.StatusGatewayTimeout, "synthesis timed out")
			return
		}
		h.log.ErrorContext(r.Context(), "synthesis failed",
			slog.String("voice", req.Voice),
			slog.Int("text_len", len(req.Text)),
			slog.Int64("duration_ms", durationMS),
			slog.String("error", err.Error()),
		)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	h.log.InfoContext(r.Context(), "synthesis complete",
		slog.String("voice", req.Voice),
		slog.Int("text_len", len(req.Text)),
		slog.Int64("duration_ms", durationMS),
		slog.Int("wav_bytes", len(wav)),
	)

	w.Header().Set("Content-Type", "audio/wav")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(wav)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// ---------------------------------------------------------------------------
// Server — wires handler into net/http.Server with graceful shutdown
// ---------------------------------------------------------------------------

// Server wires the HTTP handler into a net/http.Server with graceful shutdown.
type Server struct {
	cfg             config.Config
	tts             *tts.Service
	shutdownTimeout time.Duration
}

func New(cfg config.Config, svc *tts.Service) *Server {
	return &Server{
		cfg:             cfg,
		tts:             svc,
		shutdownTimeout: 30 * time.Second,
	}
}

// WithShutdownTimeout overrides the graceful-shutdown drain period.
func (s *Server) WithShutdownTimeout(d time.Duration) *Server {
	s.shutdownTimeout = d
	return s
}

func (s *Server) Start(ctx context.Context) error {
	workers := s.cfg.Server.Workers
	if workers <= 0 {
		workers = s.cfg.TTS.Concurrency
	}

	handlerOpts := []Option{
		WithWorkers(workers),
		WithMaxTextBytes(s.cfg.Server.MaxTextBytes),
		WithRequestTimeout(time.Duration(s.cfg.Server.RequestTimeout) * time.Second),
	}

	h := NewHandler(&noopSynthesizer{}, &noopVoiceLister{}, handlerOpts...)

	httpServer := &http.Server{
		Addr:              s.cfg.Server.ListenAddr,
		Handler:           h,
		ReadHeaderTimeout: 5 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		errCh <- httpServer.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), s.shutdownTimeout)
		defer cancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("http shutdown: %w", err)
		}
		return nil
	case err := <-errCh:
		if err == http.ErrServerClosed {
			return nil
		}
		return fmt.Errorf("http listen: %w", err)
	}
}

func ProbeHTTP(addr string) error {
	resp, err := http.Get("http://" + addr + "/health") //nolint:noctx
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected health status: %s", resp.Status)
	}
	return nil
}

// ---------------------------------------------------------------------------
// no-op placeholders used until Phase 6 wires the real implementations
// ---------------------------------------------------------------------------

type noopSynthesizer struct{}

func (n *noopSynthesizer) Synthesize(_ context.Context, _, _ string) ([]byte, error) {
	return nil, fmt.Errorf("synthesizer not configured — wire in Phase 6")
}

type noopVoiceLister struct{}

func (n *noopVoiceLister) ListVoices() []tts.Voice { return []tts.Voice{} }
