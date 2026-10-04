package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/tts"
)

// ErrUnknownLanguage reports a request language the server does not serve.
var ErrUnknownLanguage = errors.New("language not served")

// LanguageBackend serves the requests of one language.
type LanguageBackend struct {
	Synth    Synthesizer
	Streamer StreamingSynthesizer // nil when the backend cannot stream
}

// LanguageRouter picks the backend for a request's language; "" means the
// language the server started with.
type LanguageRouter interface {
	// Acquire returns language's backend, loading it first if needed. The
	// caller must call release once the request is done.
	Acquire(ctx context.Context, language string) (backend LanguageBackend, release func(), err error)
	// Voices lists language's voices without loading its model.
	Voices(language string) (VoiceLister, error)
}

// languageSpec describes a language the server may load.
type languageSpec struct {
	voices VoiceLister
	build  func() (loadedLanguage, error)
}

// loadedLanguage is a built backend and how to free it (nil: nothing to free,
// such as a tts.Service the caller owns).
type loadedLanguage struct {
	backend LanguageBackend
	close   func()
}

type languageEntry struct {
	name     string
	ready    chan struct{} // closed once loaded or err is set
	loaded   loadedLanguage
	err      error
	refs     int // in-flight requests, the loading one included
	lastUsed uint64
	evicted  bool
}

// languageRegistry loads each language's model on first use and keeps at
// most max of them. The least recently used model leaves the registry as soon
// as another language needs room, but is freed only after its in-flight
// requests finish, so memory can briefly exceed the cap.
type languageRegistry struct {
	startup string
	max     int
	specs   map[string]languageSpec
	log     *slog.Logger

	mu      sync.Mutex
	entries map[string]*languageEntry
	tick    uint64
}

// newLanguageRegistry serves specs, with startup already loaded as preloaded.
// A startup of "" (a custom model config) accepts only requests without a
// language.
func newLanguageRegistry(
	startup string, maxLanguages int, specs map[string]languageSpec, preloaded loadedLanguage, log *slog.Logger,
) *languageRegistry {
	ready := make(chan struct{})
	close(ready)

	return &languageRegistry{
		startup: startup,
		max:     max(maxLanguages, 1),
		specs:   specs,
		log:     log,
		entries: map[string]*languageEntry{
			startup: {name: startup, ready: ready, loaded: preloaded},
		},
	}
}

// Acquire implements LanguageRouter.
func (r *languageRegistry) Acquire(ctx context.Context, language string) (LanguageBackend, func(), error) {
	name := language
	if name == "" {
		name = r.startup
	}

	spec, ok := r.specs[name]
	if !ok {
		return LanguageBackend{}, nil, r.unknown(language)
	}

	r.mu.Lock()

	e, loaded := r.entries[name]
	if !loaded {
		e = &languageEntry{name: name, ready: make(chan struct{})}
		r.entries[name] = e
	}

	e.refs++
	r.tick++
	e.lastUsed = r.tick
	idle := r.evictLocked(e)

	r.mu.Unlock()

	for _, old := range idle {
		r.free(old)
	}

	if !loaded {
		r.load(e, spec)
	}

	select {
	case <-e.ready:
	case <-ctx.Done():
		r.release(e)
		return LanguageBackend{}, nil, ctx.Err()
	}

	if e.err != nil {
		r.release(e)
		return LanguageBackend{}, nil, e.err
	}

	var once sync.Once

	return e.loaded.backend, func() { once.Do(func() { r.release(e) }) }, nil
}

// Voices implements LanguageRouter.
func (r *languageRegistry) Voices(language string) (VoiceLister, error) {
	name := language
	if name == "" {
		name = r.startup
	}

	spec, ok := r.specs[name]
	if !ok {
		return nil, r.unknown(language)
	}

	return spec.voices, nil
}

// load builds e's backend. A failed load leaves the registry, so the next
// request tries again.
func (r *languageRegistry) load(e *languageEntry, spec languageSpec) {
	start := time.Now()
	loaded, err := spec.build()

	r.mu.Lock()

	e.loaded, e.err = loaded, err
	if err != nil && r.entries[e.name] == e {
		delete(r.entries, e.name)
	}

	r.mu.Unlock()
	close(e.ready)

	if err != nil {
		r.log.Error("load language failed", slog.String("language", e.name), slog.String("error", err.Error()))
		return
	}

	r.log.Info("loaded language", slog.String("language", e.name),
		slog.Int64("duration_ms", time.Since(start).Milliseconds()))
}

// evictLocked removes least recently used entries other than keep until the
// registry fits max, and returns the ones no request uses any more. The
// others are freed by their last release.
func (r *languageRegistry) evictLocked(keep *languageEntry) []*languageEntry {
	var idle []*languageEntry

	for len(r.entries) > r.max {
		var lru *languageEntry

		for _, e := range r.entries {
			if e != keep && (lru == nil || e.lastUsed < lru.lastUsed) {
				lru = e
			}
		}

		if lru == nil {
			break
		}

		delete(r.entries, lru.name)
		lru.evicted = true

		r.log.Info("unloading language", slog.String("language", lru.name),
			slog.Int("in_flight", lru.refs), slog.String("for", keep.name))

		if lru.refs == 0 {
			idle = append(idle, lru)
		}
	}

	return idle
}

func (r *languageRegistry) release(e *languageEntry) {
	r.mu.Lock()
	e.refs--
	free := e.refs == 0 && e.evicted
	r.mu.Unlock()

	if free {
		r.free(e)
	}
}

func (r *languageRegistry) free(e *languageEntry) {
	if e.loaded.close != nil {
		e.loaded.close()
	}

	r.log.Info("freed language", slog.String("language", e.name))
}

func (r *languageRegistry) unknown(language string) error {
	names := make([]string, 0, len(r.specs))

	for name := range r.specs {
		if name != "" {
			names = append(names, name)
		}
	}

	if len(names) == 0 {
		return fmt.Errorf("%w: %q (this server runs a custom model; send requests without a language)",
			ErrUnknownLanguage, language)
	}

	slices.Sort(names)

	return fmt.Errorf("%w: %q (served: %s)", ErrUnknownLanguage, language, strings.Join(names, ", "))
}

// languageSpecs returns the startup language's name ("" for a custom model
// config) and a spec for every other language in --server-languages. It
// checks each one's voice manifest, default voice, model and tokenizer now,
// before any weights load, so serve does not start with a language it
// cannot load.
func (s *Server) languageSpecs(backend string) (string, map[string]languageSpec, error) {
	if s.cfg.Server.MaxLanguages < 1 {
		return "", nil, fmt.Errorf("--server-max-languages must be at least 1, got %d", s.cfg.Server.MaxLanguages)
	}

	startup := s.cfg.TTS.Language
	if s.cfg.TTS.ModelConfigPath != "" {
		startup = ""
	}

	specs := map[string]languageSpec{}

	for _, raw := range s.cfg.Server.Languages {
		language := strings.TrimSpace(raw)
		if language == "" || language == startup {
			continue
		}

		if _, ok := specs[language]; ok {
			continue
		}

		if s.cfg.TTS.ModelConfigPath != "" {
			return "", nil, errors.New("--server-languages cannot be combined with --model-config, " +
				"which serves only its own model")
		}

		if backend != config.BackendNative {
			return "", nil, fmt.Errorf("--server-languages %s needs the %s backend, not %s",
				language, config.BackendNative, backend)
		}

		spec, err := s.otherLanguage(language)
		if err != nil {
			return "", nil, err
		}

		specs[language] = spec
	}

	return startup, specs, nil
}

// otherLanguage checks language's files and returns a spec that loads its
// model on first use, with its built-in default voice.
func (s *Server) otherLanguage(language string) (languageSpec, error) {
	cfg, err := config.ForLanguage(s.cfg, language)
	if err != nil {
		return languageSpec{}, fmt.Errorf("--server-languages: %w", err)
	}

	cfg.TTS.Backend = config.BackendNative
	hint := "run 'pockettts model download --language " + language + "'"

	vm, err := tts.NewVoiceManager(cfg.Paths.VoiceManifest)
	if err != nil {
		return languageSpec{}, fmt.Errorf("language %s: %w; %s", language, err, hint)
	}

	defaultVoice, err := resolveDefaultVoice(cfg, vm)
	if err != nil {
		return languageSpec{}, fmt.Errorf("language %s: %w", language, err)
	}

	for _, p := range []string{cfg.Paths.ModelPath, cfg.Paths.TokenizerModel} {
		_, err := os.Stat(p)
		if err != nil {
			return languageSpec{}, fmt.Errorf("language %s: %w; %s", language, err, hint)
		}
	}

	build := func() (loadedLanguage, error) {
		svc, err := tts.NewService(cfg)
		if err != nil {
			return loadedLanguage{}, fmt.Errorf("language %s: %w", language, err)
		}

		err = svc.CheckVoice(defaultVoice)
		if err != nil {
			svc.Close()

			return loadedLanguage{}, fmt.Errorf("language %s: default voice %s does not fit the model: %w",
				language, defaultVoice, err)
		}

		ns := &nativeSynthesizer{svc: svc, voices: vm, defaultVoice: defaultVoice}

		return loadedLanguage{backend: LanguageBackend{Synth: ns, Streamer: ns}, close: svc.Close}, nil
	}

	return languageSpec{voices: vm, build: build}, nil
}

// loadedStartup wraps the startup language's backend. The registry frees a
// service only if the server built it, not one passed to New.
func (s *Server) loadedStartup(synth Synthesizer, streamer StreamingSynthesizer) loadedLanguage {
	loaded := loadedLanguage{backend: LanguageBackend{Synth: synth, Streamer: streamer}}

	if ns, ok := synth.(*nativeSynthesizer); ok && s.tts == nil {
		loaded.close = ns.svc.Close
	}

	return loaded
}
