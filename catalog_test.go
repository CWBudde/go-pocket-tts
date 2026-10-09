package pockettts_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	pockettts "github.com/cwbudde/go-pocket-tts"
	"github.com/cwbudde/go-pocket-tts/internal/catalogsrc"
)

// TestCatalogMatchesManifests rebuilds the catalog from the pinned manifests
// with the embedded sizes, so a changed manifest or model config that was
// not regenerated (go generate .) fails here offline.
func TestCatalogMatchesManifests(t *testing.T) {
	embedded, err := pockettts.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}

	sizes := map[string]int64{}

	for _, m := range embedded.Models {
		for _, f := range m.Files() {
			sizes[f.URL] = f.Size
		}
	}

	rebuilt, err := catalogsrc.Build(func(r catalogsrc.Ref) (int64, error) {
		n, ok := sizes[r.URL()]
		if !ok {
			return 0, fmt.Errorf("%s is not in catalog.json; run go generate .", r.URL())
		}

		return n, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	want, err := json.Marshal(rebuilt)
	if err != nil {
		t.Fatal(err)
	}

	got, err := json.Marshal(embedded)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(got, want) {
		t.Fatal("catalog.json is stale; run go generate .")
	}
}

func TestCatalogInvariants(t *testing.T) {
	catalog, err := pockettts.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}

	_, err = pockettts.LookupModel(catalog.Default)
	if err != nil {
		t.Fatalf("default model: %v", err)
	}

	paths := map[string]bool{}

	for _, m := range catalog.Models {
		if m.Label == "" || m.Language == "" || m.Layers <= 0 || len(m.Voices) == 0 {
			t.Errorf("%s: incomplete entry %+v", m.Name, m)
		}

		if _, ok := m.Voice(m.DefaultVoice); !ok {
			t.Errorf("%s: default voice %q missing", m.Name, m.DefaultVoice)
		}

		for _, f := range m.Files() {
			switch {
			case !strings.HasPrefix(f.URL, "https://huggingface.co/") || !strings.Contains(f.URL, "/resolve/"):
				t.Errorf("%s: unpinned URL %s", m.Name, f.URL)
			case len(f.SHA256) != 64 || f.Size <= 0:
				t.Errorf("%s: %s lacks checksum or size", m.Name, f.Path)
			case !strings.HasPrefix(f.Path, m.Name+"/") || strings.Contains(f.Path, ".."):
				t.Errorf("%s: path %s escapes the model directory", m.Name, f.Path)
			case paths[f.Path]:
				t.Errorf("%s: duplicate path %s", m.Name, f.Path)
			}

			paths[f.Path] = true
		}
	}
}

func TestLookupModelUnknown(t *testing.T) {
	_, err := pockettts.LookupModel("klingon")
	if err == nil || !strings.Contains(err.Error(), "english_2026-01") {
		t.Fatalf("LookupModel(klingon) = %v; want an error listing the models", err)
	}
}

func TestCatalogJSONIsACopy(t *testing.T) {
	a := pockettts.CatalogJSON()
	a[0] = 'x'

	if pockettts.CatalogJSON()[0] == 'x' {
		t.Fatal("CatalogJSON exposes the embedded bytes")
	}
}
