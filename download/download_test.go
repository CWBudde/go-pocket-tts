package download_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	pockettts "github.com/cwbudde/go-pocket-tts"
	"github.com/cwbudde/go-pocket-tts/download"
)

// pinned describes want as served at srv/name.
func pinned(srv *httptest.Server, name string, want []byte) pockettts.File {
	sum := sha256.Sum256(want)

	return pockettts.File{
		URL:    srv.URL + "/" + name,
		SHA256: hex.EncodeToString(sum[:]),
		Size:   int64(len(want)),
		Path:   "model/weights.safetensors",
	}
}

func serve(t *testing.T, bodies map[string][]byte, hits *atomic.Int32) *httptest.Server {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)

		body, ok := bodies[strings.TrimPrefix(r.URL.Path, "/")]
		if !ok {
			http.NotFound(w, r)
			return
		}

		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)

	return srv
}

func TestFileDownloadsVerifiesAndSkipsPresent(t *testing.T) {
	var hits atomic.Int32

	body := []byte(strings.Repeat("weights", 1000))
	srv := serve(t, map[string][]byte{"w": body}, &hits)
	f := pinned(srv, "w", body)
	root := t.TempDir()

	var last download.Progress

	err := download.File(context.Background(), root, f, download.Options{Progress: func(p download.Progress) { last = p }})
	if err != nil {
		t.Fatal(err)
	}

	got, err := os.ReadFile(filepath.Join(root, "model", "weights.safetensors"))
	if err != nil || string(got) != string(body) {
		t.Fatalf("downloaded file = %q, %v", got, err)
	}

	if last.Done != f.Size || last.Total != f.Size || last.File != f.Path {
		t.Fatalf("last progress %+v; want the whole file", last)
	}

	err = download.File(context.Background(), root, f, download.Options{})
	if err != nil {
		t.Fatal(err)
	}

	if hits.Load() != 1 {
		t.Fatalf("server hits = %d; a present file must not be fetched again", hits.Load())
	}
}

func TestFileRejectsWrongBytesWithoutReplacingAGoodFile(t *testing.T) {
	var hits atomic.Int32

	want := []byte("good weights")
	srv := serve(t, map[string][]byte{
		"other-content": []byte("evil weights"),
		"one-too-long":  []byte("good weights!"),
		"truncated":     []byte("good"),
	}, &hits)

	for _, name := range []string{"other-content", "one-too-long", "truncated", "missing"} {
		t.Run(name, func(t *testing.T) {
			f := pinned(srv, name, want)
			root := t.TempDir()
			dst := filepath.Join(root, filepath.FromSlash(f.Path))

			err := os.MkdirAll(filepath.Dir(dst), 0o755)
			if err != nil {
				t.Fatal(err)
			}

			// A stale file of the right size but the wrong content.
			err = os.WriteFile(dst, []byte("stale weight"), 0o600)
			if err != nil {
				t.Fatal(err)
			}

			err = download.File(context.Background(), root, f, download.Options{})
			if err == nil {
				t.Fatal("wrong bytes accepted")
			}

			got, _ := os.ReadFile(dst)
			if string(got) != "stale weight" {
				t.Fatalf("existing file replaced by %q", got)
			}

			_, err = os.Stat(dst + ".part")
			if !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("partial file left behind: %v", err)
			}
		})
	}
}

func TestFileHonoursCancellation(t *testing.T) {
	var hits atomic.Int32

	body := []byte("weights")
	srv := serve(t, map[string][]byte{"w": body}, &hits)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := download.File(ctx, t.TempDir(), pinned(srv, "w", body), download.Options{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("File(cancelled) = %v; want context.Canceled", err)
	}
}

func TestFilesSelectsDefaultOrNamedVoices(t *testing.T) {
	m, err := pockettts.LookupModel("german")
	if err != nil {
		t.Fatal(err)
	}

	files, err := download.Files(m, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(files) != 3 || files[2].Path != "german/voices/"+m.DefaultVoice+".safetensors" {
		t.Fatalf("default files = %+v", files)
	}

	_, err = download.Files(m, []string{"nobody"})
	if err == nil {
		t.Fatal("unknown voice accepted")
	}
}
