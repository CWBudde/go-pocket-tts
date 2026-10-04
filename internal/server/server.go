package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"os/exec"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/audio"
	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
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

// StreamingSynthesizer produces audio incrementally via a channel of PCM chunks.
type StreamingSynthesizer interface {
	SynthesizeStream(ctx context.Context, text, voice string, out chan<- tts.PCMChunk) error
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
	streamer       StreamingSynthesizer
	router         LanguageRouter
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

// WithStreamer sets the streaming synthesizer for /tts/stream.
// If nil, the streaming endpoint returns 501 Not Implemented.
func WithStreamer(s StreamingSynthesizer) Option {
	return func(o *options) { o.streamer = s }
}

// WithLanguages lets requests pick a language with their language field and
// /voices?language=. Without it, requests that name a language get 400.
func WithLanguages(r LanguageRouter) Option {
	return func(o *options) { o.router = r }
}

// ---------------------------------------------------------------------------
// handler
// ---------------------------------------------------------------------------

// handler holds the dependencies needed to serve HTTP requests.
type handler struct {
	synth    Synthesizer
	streamer StreamingSynthesizer // nil when streaming is not available
	voices   VoiceLister
	opts     options
	sem      chan struct{} // semaphore for worker pool
	log      *slog.Logger
}

// NewHandler returns an http.Handler that serves /health, /voices, and POST /tts.
func NewHandler(synth Synthesizer, voices VoiceLister, optFns ...Option) http.Handler {
	opts := defaultOptions()
	for _, fn := range optFns {
		fn(&opts)
	}

	h := &handler{
		synth:    synth,
		streamer: opts.streamer,
		voices:   voices,
		opts:     opts,
		log:      opts.logger,
	}
	if opts.workers > 0 {
		h.sem = make(chan struct{}, opts.workers)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", h.handleHealth)
	mux.HandleFunc("/voices", h.handleVoices)
	mux.HandleFunc("/tts", h.handleTTS)
	mux.HandleFunc("/tts/stream", h.handleTTSStream)

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

func (h *handler) handleVoices(w http.ResponseWriter, r *http.Request) {
	lister := h.voices

	language := r.URL.Query().Get("language")
	if h.opts.router != nil {
		var err error

		lister, err = h.opts.router.Voices(language)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
	} else if language != "" {
		writeError(w, http.StatusBadRequest, errNoLanguages)
		return
	}

	voices := lister.ListVoices()
	if voices == nil {
		voices = []tts.Voice{}
	}

	writeJSON(w, http.StatusOK, voices)
}

type ttsRequest struct {
	Text  string `json:"text"`
	Voice string `json:"voice"`
	Chunk bool   `json:"chunk"`
	// Language picks one of the served model configs; empty means the
	// startup language.
	Language string `json:"language"`
}

const errNoLanguages = "this server does not take a language"

func (h *handler) handleTTS(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	if r.Body == nil {
		writeError(w, http.StatusBadRequest, "request body is required")
		return
	}

	req, ok := h.decodeTTSRequest(w, r)
	if !ok || !h.checkLanguage(w, req.Language) {
		return
	}

	// Acquire a worker slot — honour context cancellation while waiting.
	if !h.acquireWorker(r.Context(), w) {
		return
	}

	if h.sem != nil {
		defer func() { <-h.sem }()
	}

	// Only with a worker slot: loading a model while queued would let
	// request bursts load more models than workers and the language cap.
	backend, release, ok := h.acquireLanguage(r.Context(), w, req.Language)
	if !ok {
		return
	}
	defer release()

	// Apply per-request timeout.
	ctx, cancel := context.WithTimeout(r.Context(), h.opts.requestTimeout)
	defer cancel()

	start := time.Now()
	wav, err := backend.Synth.Synthesize(ctx, req.Text, req.Voice)
	durationMS := time.Since(start).Milliseconds()

	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			h.log.WarnContext(r.Context(), "synthesis timed out",
				slog.String("language", req.Language),
				slog.String("voice", req.Voice),
				slog.Int("text_len", len(req.Text)),
				slog.Int64("duration_ms", durationMS),
				slog.String("error", err.Error()),
			)
			writeError(w, http.StatusGatewayTimeout, "synthesis timed out")

			return
		}

		h.log.ErrorContext(r.Context(), "synthesis failed",
			slog.String("language", req.Language),
			slog.String("voice", req.Voice),
			slog.Int("text_len", len(req.Text)),
			slog.Int64("duration_ms", durationMS),
			slog.String("error", err.Error()),
		)
		writeError(w, http.StatusInternalServerError, err.Error())

		return
	}

	h.log.InfoContext(r.Context(), "synthesis complete",
		slog.String("language", req.Language),
		slog.String("voice", req.Voice),
		slog.Int("text_len", len(req.Text)),
		slog.Int64("duration_ms", durationMS),
		slog.Int("wav_bytes", len(wav)),
	)

	w.Header().Set("Content-Type", "audio/wav")
	w.WriteHeader(http.StatusOK)
	// #nosec G705 -- Writes binary audio bytes to an HTTP response with audio/wav content type, not HTML.
	_, _ = w.Write(wav)
}

func (h *handler) handleTTSStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := h.prepareStreamingResponse(w, r)
	if !ok {
		return
	}

	req, ok := h.decodeTTSRequest(w, r)
	if !ok || !h.checkLanguage(w, req.Language) {
		return
	}

	if !h.acquireWorker(r.Context(), w) {
		return
	}

	if h.sem != nil {
		defer func() { <-h.sem }()
	}

	backend, release, ok := h.acquireLanguage(r.Context(), w, req.Language)
	if !ok {
		return
	}
	defer release()

	if backend.Streamer == nil {
		writeError(w, http.StatusNotImplemented, "streaming not available for this backend")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), h.opts.requestTimeout)
	defer cancel()

	start := time.Now()

	// streamChunks handles the synthesis and streaming, and returns the total number of audio samples sent.
	totalSamples, err := h.streamChunks(ctx, cancel, w, flusher, backend.Streamer, req)
	if err != nil {
		h.log.ErrorContext(r.Context(), "streaming synthesis failed",
			slog.String("language", req.Language),
			slog.String("voice", req.Voice),
			slog.Int("text_len", len(req.Text)),
			slog.Int64("duration_ms", time.Since(start).Milliseconds()),
			slog.String("error", err.Error()),
		)

		return
	}

	h.log.InfoContext(r.Context(), "streaming synthesis complete",
		slog.String("language", req.Language),
		slog.String("voice", req.Voice),
		slog.Int("text_len", len(req.Text)),
		slog.Int64("duration_ms", time.Since(start).Milliseconds()),
		slog.Int("total_samples", totalSamples),
	)
}

func (h *handler) prepareStreamingResponse(w http.ResponseWriter, r *http.Request) (http.Flusher, bool) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return nil, false
	}

	// With languages, the language's backend decides after the body is read.
	if h.streamer == nil && h.opts.router == nil {
		writeError(w, http.StatusNotImplemented, "streaming not available for this backend")
		return nil, false
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return nil, false
	}

	if r.Body == nil {
		writeError(w, http.StatusBadRequest, "request body is required")
		return nil, false
	}

	return flusher, true
}

// decodeTTSRequest reads and validates the body of /tts and /tts/stream. On
// failure it writes an HTTP error and returns false.
func (h *handler) decodeTTSRequest(w http.ResponseWriter, r *http.Request) (ttsRequest, bool) {
	var req ttsRequest

	err := json.NewDecoder(r.Body).Decode(&req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return ttsRequest{}, false
	}

	if req.Text == "" {
		writeError(w, http.StatusBadRequest, "text field is required")
		return ttsRequest{}, false
	}

	if len(req.Text) > h.opts.maxTextBytes {
		writeError(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("text exceeds maximum size of %d bytes", h.opts.maxTextBytes))

		return ttsRequest{}, false
	}

	return req, true
}

func (h *handler) streamChunks(
	ctx context.Context,
	cancel context.CancelFunc,
	w http.ResponseWriter,
	flusher http.Flusher,
	streamer StreamingSynthesizer,
	req ttsRequest,
) (int, error) {
	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)

	_, err := audio.WriteWAVHeaderStreaming(w)
	if err != nil {
		h.log.ErrorContext(ctx, "failed to write WAV header", slog.String("error", err.Error()))
		return 0, err
	}

	flusher.Flush()

	chunkCh := make(chan tts.PCMChunk, 2)
	errCh := make(chan error, 1)

	go func() {
		errCh <- streamer.SynthesizeStream(ctx, req.Text, req.Voice, chunkCh)
	}()

	totalSamples := 0
	for chunk := range chunkCh {
		totalSamples += len(chunk.Samples)

		_, err := audio.WritePCM16Samples(w, chunk.Samples)
		if err != nil {
			h.log.ErrorContext(ctx, "failed to write PCM chunk", slog.String("error", err.Error()))
			cancel()

			break
		}

		flusher.Flush()
	}

	return totalSamples, <-errCh
}

// checkLanguage rejects a language the server does not serve with 400, so
// such a request does not wait for a worker slot first.
func (h *handler) checkLanguage(w http.ResponseWriter, language string) bool {
	if h.opts.router == nil {
		if language != "" {
			writeError(w, http.StatusBadRequest, errNoLanguages)
			return false
		}

		return true
	}

	// Voices reads no model; its error is the same as Acquire's.
	_, err := h.opts.router.Voices(language)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return false
	}

	return true
}

// acquireLanguage returns the backend for language and the func that ends
// its use. On failure it writes an HTTP error and returns false: 400 for a
// language the server does not serve, 503 when the request ends while the
// model loads, 500 when it fails to load.
func (h *handler) acquireLanguage(ctx context.Context, w http.ResponseWriter, language string) (LanguageBackend, func(), bool) {
	if h.opts.router == nil {
		if language != "" {
			writeError(w, http.StatusBadRequest, errNoLanguages)
			return LanguageBackend{}, nil, false
		}

		return LanguageBackend{Synth: h.synth, Streamer: h.streamer}, func() {}, true
	}

	backend, release, err := h.opts.router.Acquire(ctx, language)

	switch {
	case err == nil:
		return backend, release, true
	case errors.Is(err, ErrUnknownLanguage):
		writeError(w, http.StatusBadRequest, err.Error())
	case errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled):
		writeError(w, http.StatusServiceUnavailable, "request cancelled while loading language")
	default:
		h.log.ErrorContext(ctx, "load language failed", slog.String("language", language), slog.String("error", err.Error()))
		writeError(w, http.StatusInternalServerError, "load language: "+err.Error())
	}

	return LanguageBackend{}, nil, false
}

// acquireWorker tries to acquire a worker slot from the semaphore.
// Returns true on success. On failure (context cancelled) it writes an HTTP
// error and returns false. When sem is nil (no throttling) it returns true
// immediately.
func (h *handler) acquireWorker(ctx context.Context, w http.ResponseWriter) bool {
	if h.sem == nil {
		return true
	}

	select {
	case h.sem <- struct{}{}:
		return true
	default:
		h.log.Info("request queued for worker slot")

		select {
		case h.sem <- struct{}{}:
			return true
		case <-ctx.Done():
			writeError(w, http.StatusServiceUnavailable, "request cancelled while waiting for worker")
			return false
		}
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	err := json.NewEncoder(w).Encode(v)
	if err != nil {
		slog.Warn("encode JSON response", "error", err)
	}
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
	backend, err := config.NormalizeBackend(s.cfg.TTS.Backend)
	if err != nil {
		return err
	}

	// The other languages are checked first: that is cheap, while the
	// startup model takes seconds to load.
	startup, specs, err := s.languageSpecs(backend)
	if err != nil {
		return err
	}

	synth, voiceLister, workers, streamer, err := s.runtimeDeps(backend)
	if err != nil {
		return err
	}

	specs[startup] = languageSpec{voices: voiceLister, build: func() (loadedLanguage, error) {
		synth, _, _, streamer, err := s.runtimeDeps(backend)
		if err != nil {
			return loadedLanguage{}, err
		}

		return s.loadedStartup(synth, streamer), nil
	}}

	router := newLanguageRegistry(startup, s.cfg.Server.MaxLanguages, specs, s.loadedStartup(synth, streamer), slog.Default())
	if len(specs) > 1 {
		slog.Info("serving languages", slog.Any("languages", slices.Sorted(maps.Keys(specs))),
			slog.Int("max_loaded", s.cfg.Server.MaxLanguages))
	}

	handlerOpts := []Option{
		WithWorkers(workers),
		WithMaxTextBytes(s.cfg.Server.MaxTextBytes),
		WithRequestTimeout(time.Duration(s.cfg.Server.RequestTimeout) * time.Second),
		WithLanguages(router),
	}
	if streamer != nil {
		handlerOpts = append(handlerOpts, WithStreamer(streamer))
	}

	h := NewHandler(synth, voiceLister, handlerOpts...)

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
		shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), s.shutdownTimeout)
		defer cancel()

		err := httpServer.Shutdown(shutdownCtx)
		if err != nil {
			return fmt.Errorf("http shutdown: %w", err)
		}

		return nil
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
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

	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected health status: %s", resp.Status)
	}

	return nil
}

func (s *Server) runtimeDeps(backend string) (Synthesizer, VoiceLister, int, StreamingSynthesizer, error) {
	voices := loadVoiceLister(s.cfg.Paths.VoiceManifest)
	// /voices lists manifest IDs; the synthesizer maps them to files.
	vm, _ := voices.(*tts.VoiceManager)

	if strings.TrimSpace(s.cfg.Server.DefaultVoice) != "" && backend != config.BackendNative {
		return nil, nil, 0, nil, fmt.Errorf("--default-voice needs the %s backend, not %s", config.BackendNative, backend)
	}

	switch backend {
	case config.BackendNative, config.BackendNativeONNX:
		var defaultVoice string

		// The ONNX runtime rejects the predefined model-state voices. The
		// default voice is checked before the model loads, so a broken one
		// fails fast.
		if backend == config.BackendNative {
			var err error

			defaultVoice, err = resolveDefaultVoice(s.cfg, vm)
			if err != nil {
				return nil, nil, 0, nil, err
			}
		}

		svc := s.tts
		if svc == nil {
			var err error
			next := s.cfg
			next.TTS.Backend = backend

			svc, err = tts.NewService(next)
			if err != nil {
				return nil, nil, 0, nil, fmt.Errorf("initialize native service: %w", err)
			}
		}

		// The file loaded above; now check it fits the loaded model.
		if defaultVoice != "" {
			err := svc.CheckVoice(defaultVoice)
			if err != nil {
				if s.tts == nil {
					svc.Close()
				}

				return nil, nil, 0, nil, fmt.Errorf("default voice %s does not fit the %s model: %w",
					defaultVoice, s.cfg.TTS.Language, err)
			}
		}

		workers := s.cfg.Server.Workers
		if workers <= 0 {
			workers = 2
		}

		ns := &nativeSynthesizer{svc: svc, voices: vm, defaultVoice: defaultVoice}

		return ns, voices, workers, ns, nil
	case config.BackendCLI:
		workers := chooseWorkerLimit(s.cfg, backend)

		return &cliSynthesizer{
			executablePath: s.cfg.TTS.CLIPath,
			configPath:     s.cfg.TTS.CLIConfigPath,
			quiet:          s.cfg.TTS.Quiet,
		}, voices, workers, nil, nil
	default:
		return nil, nil, 0, nil, fmt.Errorf("unsupported backend %q", backend)
	}
}

// resolveDefaultVoice returns the file of cfg's default voice for requests
// without one: --default-voice (a manifest ID or .safetensors path; serve
// has already downloaded URLs) or else the model config's default voice
// (upstream get_default_voice_for_language). It loads the file once, so a
// voice that cannot be used fails at startup rather than per request.
func resolveDefaultVoice(cfg config.Config, vm *tts.VoiceManager) (string, error) {
	source, ref := "--default-voice", strings.TrimSpace(cfg.Server.DefaultVoice)
	if ref == "" {
		if cfg.Model == nil || cfg.Model.DefaultVoice == "" {
			return "", nil
		}

		source, ref = "default voice", cfg.Model.DefaultVoice
	}

	hint := "run 'pockettts model download' (same --language)"
	if cfg.TTS.ModelConfigPath == "" {
		hint = "run 'pockettts model download --language " + cfg.TTS.Language + "'"
	}

	// An exact manifest ID wins, as for request voices; otherwise a value
	// that looks like a file is a path and anything else must be an ID.
	path := ref
	fromManifest := hasVoiceID(vm, ref) ||
		(!strings.ContainsRune(ref, filepath.Separator) && !strings.HasSuffix(ref, ".safetensors"))

	if fromManifest {
		if vm == nil {
			return "", fmt.Errorf("%s %q: voice manifest %s not readable; %s", source, ref, cfg.Paths.VoiceManifest, hint)
		}

		var err error

		path, err = vm.ResolvePath(ref)
		if err != nil {
			return "", fmt.Errorf("%s %q (voice manifest %s): %w; %s", source, ref, cfg.Paths.VoiceManifest, err, hint)
		}
	}

	err := tts.CheckVoiceFile(path)
	if err != nil {
		if fromManifest {
			return "", fmt.Errorf("%s %q (%s): %w; %s", source, ref, path, err, hint)
		}

		return "", fmt.Errorf("%s %q: %w", source, ref, err)
	}

	return path, nil
}

// hasVoiceID reports whether the manifest (nil when unreadable) lists id.
func hasVoiceID(vm *tts.VoiceManager, id string) bool {
	if vm == nil {
		return false
	}

	for _, v := range vm.ListVoices() {
		if v.ID == id {
			return true
		}
	}

	return false
}

func chooseWorkerLimit(cfg config.Config, backend string) int {
	if backend != config.BackendCLI {
		return 0
	}

	workers := cfg.Server.Workers
	if workers <= 0 {
		workers = cfg.TTS.Concurrency
	}

	return workers
}

func loadVoiceLister(manifestPath string) VoiceLister {
	vm, err := tts.NewVoiceManager(manifestPath)
	if err != nil {
		return staticVoiceLister{}
	}

	return vm
}

type staticVoiceLister struct {
	voices []tts.Voice
}

func (s staticVoiceLister) ListVoices() []tts.Voice {
	return append([]tts.Voice(nil), s.voices...)
}

type nativeSynthesizer struct {
	svc          *tts.Service
	voices       *tts.VoiceManager // nil when the voice manifest is unreadable
	defaultVoice string            // checked default voice file for requests without one
}

func (n *nativeSynthesizer) Synthesize(ctx context.Context, text, voice string) ([]byte, error) {
	path, err := n.voicePath(voice)
	if err != nil {
		return nil, err
	}

	samples, err := n.svc.SynthesizeCtx(ctx, text, path)
	if err != nil {
		return nil, err
	}

	return audio.EncodeWAV(samples)
}

func (n *nativeSynthesizer) SynthesizeStream(ctx context.Context, text, voice string, out chan<- tts.PCMChunk) error {
	path, err := n.voicePath(voice)
	if err != nil {
		return err
	}

	return n.svc.SynthesizeStream(ctx, text, path, out)
}

// voicePath maps a voice ID from the manifest, as listed by /voices, to its
// file; tts.Service only accepts file paths. Other values, such as a direct
// .safetensors path, pass through unchanged. No voice means the default
// voice file that runtimeDeps resolved and checked at startup.
func (n *nativeSynthesizer) voicePath(voice string) (string, error) {
	if strings.TrimSpace(voice) == "" {
		return n.defaultVoice, nil
	}

	if n.voices == nil {
		return voice, nil
	}

	for _, v := range n.voices.ListVoices() {
		if v.ID == voice {
			return n.voices.ResolvePath(voice)
		}
	}

	return voice, nil
}

type cliSynthesizer struct {
	executablePath string
	configPath     string
	quiet          bool
}

func (c *cliSynthesizer) Synthesize(ctx context.Context, text, voice string) ([]byte, error) {
	exe := c.executablePath
	if exe == "" {
		exe = "pocket-tts"
	}

	args := []string{"generate", "--text", "-", "--output-path", "-"}
	if strings.TrimSpace(voice) != "" {
		args = append(args, "--voice", voice)
	}

	if c.configPath != "" {
		args = append(args, "--config", c.configPath)
	}

	if c.quiet {
		args = append(args, "--quiet")
	}

	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Stdin = strings.NewReader(text)

	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = io.Discard

	err := cmd.Run()
	if err != nil {
		return nil, err
	}

	return out.Bytes(), nil
}
