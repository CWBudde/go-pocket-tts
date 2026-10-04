package server

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/tts"
)

// nameSynth answers every request with its name, so a test can tell which
// language served it.
type nameSynth struct{ name string }

func (n nameSynth) Synthesize(context.Context, string, string) ([]byte, error) {
	return []byte(n.name), nil
}

// fakeLanguage counts the builds and frees of one language's backend.
type fakeLanguage struct {
	builds atomic.Int32
	frees  atomic.Int32
	// gate, when set, holds every build until it is closed.
	gate chan struct{}
	// fail makes builds fail while it returns an error.
	fail func() error
}

func (f *fakeLanguage) spec(name string) languageSpec {
	return languageSpec{
		voices: staticVoiceLister{voices: []tts.Voice{{ID: name + "-voice"}}},
		build: func() (loadedLanguage, error) {
			f.builds.Add(1)

			if f.gate != nil {
				<-f.gate
			}

			if f.fail != nil {
				err := f.fail()
				if err != nil {
					return loadedLanguage{}, err
				}
			}

			return loadedLanguage{
				backend: LanguageBackend{Synth: nameSynth{name}},
				close:   func() { f.frees.Add(1) },
			}, nil
		},
	}
}

func discardLogger() *slog.Logger {
	return slog.New(slog.DiscardHandler)
}

// newTestRegistry serves english as the startup language (preloaded, as
// Start does) plus the other languages.
func newTestRegistry(t *testing.T, maxLanguages int, langs map[string]*fakeLanguage) *languageRegistry {
	t.Helper()

	specs := map[string]languageSpec{}
	for name, f := range langs {
		specs[name] = f.spec(name)
	}

	preloaded, err := specs["english"].build()
	if err != nil {
		t.Fatal(err)
	}

	langs["english"].builds.Store(0)

	return newLanguageRegistry("english", maxLanguages, specs, preloaded, discardLogger())
}

func acquireOK(t *testing.T, r *languageRegistry, language string) (string, func()) {
	t.Helper()

	backend, release, err := r.Acquire(context.Background(), language)
	if err != nil {
		t.Fatalf("Acquire(%q): %v", language, err)
	}

	wav, err := backend.Synth.Synthesize(context.Background(), "hi", "")
	if err != nil {
		t.Fatal(err)
	}

	return string(wav), release
}

func TestLanguageRegistry_StartupIsPreloaded(t *testing.T) {
	english := &fakeLanguage{}
	r := newTestRegistry(t, 2, map[string]*fakeLanguage{"english": english, "german": {}})

	for _, language := range []string{"", "english"} {
		got, release := acquireOK(t, r, language)
		release()

		if got != "english" {
			t.Errorf("Acquire(%q) served %q; want english", language, got)
		}
	}

	if n := english.builds.Load(); n != 0 {
		t.Errorf("startup language built %d more times; want 0 (it is preloaded)", n)
	}
}

func TestLanguageRegistry_LoadsOnceConcurrently(t *testing.T) {
	german := &fakeLanguage{gate: make(chan struct{})}
	r := newTestRegistry(t, 2, map[string]*fakeLanguage{"english": {}, "german": german})

	var wg sync.WaitGroup

	results := make([]string, 8)
	errs := make([]error, 8)

	for i := range results {
		wg.Go(func() {
			backend, release, err := r.Acquire(context.Background(), "german")
			if err != nil {
				errs[i] = err
				return
			}
			defer release()

			wav, _ := backend.Synth.Synthesize(context.Background(), "hi", "")
			results[i] = string(wav)
		})
	}

	// Let every goroutine reach the registry before the build finishes.
	time.Sleep(50 * time.Millisecond)
	close(german.gate)
	wg.Wait()

	for i := range results {
		if errs[i] != nil || results[i] != "german" {
			t.Errorf("request %d = %q, %v; want german", i, results[i], errs[i])
		}
	}

	if n := german.builds.Load(); n != 1 {
		t.Errorf("german built %d times; want 1", n)
	}
}

func TestLanguageRegistry_EvictsLRUOnlyAfterInFlightRequests(t *testing.T) {
	english, german, french := &fakeLanguage{}, &fakeLanguage{}, &fakeLanguage{}
	r := newTestRegistry(t, 2, map[string]*fakeLanguage{"english": english, "german": german, "french": french})

	// english (startup) and german fit; german stays in flight.
	_, releaseGerman := acquireOK(t, r, "german")

	// french needs room: english is least recently used and idle, so it is
	// freed at once.
	_, releaseFrench := acquireOK(t, r, "french")
	releaseFrench()

	if n := english.frees.Load(); n != 1 {
		t.Fatalf("english freed %d times; want 1 (idle LRU entry)", n)
	}

	// english comes back and pushes out german, which is still in flight:
	// it leaves the registry but is freed only after its request ends.
	_, releaseEnglish := acquireOK(t, r, "english")
	releaseEnglish()

	if n := german.frees.Load(); n != 0 {
		t.Fatalf("german freed %d times while its request was in flight; want 0", n)
	}

	releaseGerman()
	releaseGerman() // a second call must not free twice

	if n := german.frees.Load(); n != 1 {
		t.Errorf("german freed %d times after its request ended; want 1", n)
	}

	if n := english.builds.Load(); n != 1 {
		t.Errorf("english rebuilt %d times; want 1", n)
	}

	// german was evicted, so the next request loads it again.
	_, release := acquireOK(t, r, "german")
	release()

	if n := german.builds.Load(); n != 2 {
		t.Errorf("german built %d times; want 2 (reloaded after eviction)", n)
	}
}

func TestLanguageRegistry_MaxOneSwitches(t *testing.T) {
	english, german := &fakeLanguage{}, &fakeLanguage{}
	r := newTestRegistry(t, 1, map[string]*fakeLanguage{"english": english, "german": german})

	for _, language := range []string{"german", "english", "german"} {
		got, release := acquireOK(t, r, language)
		release()

		if got != language {
			t.Errorf("Acquire(%q) served %q", language, got)
		}
	}

	// Each switch frees the previous model: english for german, german for
	// english, english again for german.
	if english.frees.Load() != 2 || german.frees.Load() != 1 {
		t.Errorf("frees english=%d german=%d; want 2 and 1 (one model loaded at a time)",
			english.frees.Load(), german.frees.Load())
	}
}

func TestLanguageRegistry_FailedLoadIsRetried(t *testing.T) {
	var broken atomic.Bool

	broken.Store(true)

	german := &fakeLanguage{fail: func() error {
		if broken.Load() {
			return errors.New("model missing")
		}

		return nil
	}}
	r := newTestRegistry(t, 2, map[string]*fakeLanguage{"english": {}, "german": german})

	_, _, err := r.Acquire(context.Background(), "german")
	if err == nil || !strings.Contains(err.Error(), "model missing") {
		t.Fatalf("Acquire(german) error = %v; want the build error", err)
	}

	broken.Store(false)

	got, release := acquireOK(t, r, "german")
	release()

	if got != "german" || german.builds.Load() != 2 {
		t.Errorf("after the fix: served %q with %d builds; want german after 2 builds", got, german.builds.Load())
	}
}

func TestLanguageRegistry_UnknownLanguage(t *testing.T) {
	r := newTestRegistry(t, 2, map[string]*fakeLanguage{"english": {}, "german": {}})

	_, _, err := r.Acquire(context.Background(), "french")
	if !errors.Is(err, ErrUnknownLanguage) || !strings.Contains(err.Error(), "english, german") {
		t.Errorf("Acquire(french) error = %v; want ErrUnknownLanguage naming english, german", err)
	}

	_, err = r.Voices("french")
	if !errors.Is(err, ErrUnknownLanguage) {
		t.Errorf("Voices(french) error = %v; want ErrUnknownLanguage", err)
	}

	voices, err := r.Voices("german")
	if err != nil || len(voices.ListVoices()) != 1 || voices.ListVoices()[0].ID != "german-voice" {
		t.Errorf("Voices(german) = %v, %v; want german-voice", voices, err)
	}
}

func TestLanguageRegistry_CustomModelServesOnlyRequestsWithoutLanguage(t *testing.T) {
	// With --model-config the startup language has no name.
	f := &fakeLanguage{}
	spec := f.spec("custom")
	preloaded, _ := spec.build()
	r := newLanguageRegistry("", 2, map[string]languageSpec{"": spec}, preloaded, discardLogger())

	got, release := acquireOK(t, r, "")
	release()

	if got != "custom" {
		t.Errorf("Acquire(\"\") served %q; want custom", got)
	}

	_, _, err := r.Acquire(context.Background(), "english_2026-01")
	if !errors.Is(err, ErrUnknownLanguage) || !strings.Contains(err.Error(), "without a language") {
		t.Errorf("Acquire(english_2026-01) error = %v; want ErrUnknownLanguage saying requests go without a language", err)
	}
}

func TestLanguageRegistry_CancelWhileWaiting(t *testing.T) {
	german := &fakeLanguage{gate: make(chan struct{})}
	r := newTestRegistry(t, 2, map[string]*fakeLanguage{"english": {}, "german": german})

	// The first request builds german and is held at the gate.
	loaded := make(chan func())

	go func() {
		_, release, err := r.Acquire(context.Background(), "german")
		if err != nil {
			t.Error(err)
		}

		loaded <- release
	}()

	time.Sleep(20 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	_, _, err := r.Acquire(ctx, "german")
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Acquire while loading = %v; want the context error", err)
	}

	close(german.gate)
	(<-loaded)()

	if german.builds.Load() != 1 {
		t.Errorf("german built %d times; want 1", german.builds.Load())
	}
}

func TestLanguageRegistry_NilCloseIsSkipped(t *testing.T) {
	// An injected tts.Service has no close func; evicting it must not panic.
	specs := map[string]languageSpec{
		"english": {voices: staticVoiceLister{}, build: func() (loadedLanguage, error) {
			return loadedLanguage{backend: LanguageBackend{Synth: nameSynth{"english"}}}, nil
		}},
		"german": (&fakeLanguage{}).spec("german"),
	}

	preloaded, _ := specs["english"].build()
	r := newLanguageRegistry("english", 1, specs, preloaded, discardLogger())

	got, release := acquireOK(t, r, "german")
	release()

	if got != "german" {
		t.Errorf("served %q; want german", got)
	}
}

func TestLanguageRegistry_FirstLoaderCanCancel(t *testing.T) {
	german := &fakeLanguage{gate: make(chan struct{})}
	r := newTestRegistry(t, 2, map[string]*fakeLanguage{"english": {}, "german": german})

	// The request that starts the load gives up while the model loads.
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	errCh := make(chan error, 1)

	go func() {
		_, _, err := r.Acquire(ctx, "german")
		errCh <- err
	}()

	select {
	case err := <-errCh:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("first Acquire while loading = %v; want the context error", err)
		}
	case <-time.After(time.Second):
		close(german.gate)
		t.Fatal("first Acquire blocked until the load finished; want it to give up with its context")
	}

	// The load goes on and serves the next request without a second build.
	close(german.gate)

	got, release := acquireOK(t, r, "german")
	release()

	if got != "german" || german.builds.Load() != 1 {
		t.Errorf("served %q after %d builds; want german after 1", got, german.builds.Load())
	}
}

func TestLanguageRegistry_LoadingModelIsNotEvicted(t *testing.T) {
	german, french := &fakeLanguage{gate: make(chan struct{})}, &fakeLanguage{}
	r := newTestRegistry(t, 1, map[string]*fakeLanguage{"english": {}, "german": german, "french": french})

	germanDone := make(chan func())

	go func() {
		_, release, err := r.Acquire(context.Background(), "german")
		if err != nil {
			t.Error(err)
		}

		germanDone <- release
	}()

	time.Sleep(20 * time.Millisecond)

	// french arrives while german loads: german must stay, so the next
	// german request joins its load instead of starting another.
	_, releaseFrench := acquireOK(t, r, "french")
	releaseFrench()

	joined := make(chan func())

	go func() {
		_, release, err := r.Acquire(context.Background(), "german")
		if err != nil {
			t.Error(err)
		}

		joined <- release
	}()

	time.Sleep(20 * time.Millisecond)
	close(german.gate)
	(<-germanDone)()
	(<-joined)()

	if n := german.builds.Load(); n != 1 {
		t.Errorf("german built %d times; want 1 (later requests join the running load)", n)
	}

	// Once german is loaded the cap holds again: french is unloaded.
	if n := french.frees.Load(); n != 1 {
		t.Errorf("french freed %d times; want 1 once german finished loading", n)
	}
}
