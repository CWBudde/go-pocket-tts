// Command gencatalog regenerates catalog.json at the module root: the public
// model catalog built by internal/catalogsrc, with every file size taken
// from the Hugging Face tree API at the file's pinned revision.
//
// Run it via `go generate .` at the module root.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/catalogsrc"
)

// maxListing bounds one tree API response.
const maxListing = 16 << 20

type treeEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
	LFS  *struct {
		Size int64 `json:"size"`
	} `json:"lfs"`
}

func main() {
	out := flag.String("out", "catalog.json", "output file")

	flag.Parse()

	err := run(*out)
	if err != nil {
		log.Fatal(err)
	}
}

func run(out string) error {
	client := &http.Client{Timeout: 60 * time.Second}
	listings := map[string]map[string]int64{}

	size := func(r catalogsrc.Ref) (int64, error) {
		dir := path.Dir(r.File)
		if dir == "." {
			dir = ""
		}

		key := r.Repo + "@" + r.Revision + ":" + dir

		sizes, ok := listings[key]
		if !ok {
			var err error

			sizes, err = listTree(client, r.Repo, r.Revision, dir)
			if err != nil {
				return 0, err
			}

			listings[key] = sizes
		}

		n, ok := sizes[r.File]
		if !ok || n <= 0 {
			return 0, fmt.Errorf("%s not listed in %s", r.File, key)
		}

		return n, nil
	}

	catalog, err := catalogsrc.Build(size)
	if err != nil {
		return err
	}

	var buf bytes.Buffer

	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")

	err = enc.Encode(catalog)
	if err != nil {
		return err
	}

	return os.WriteFile(out, buf.Bytes(), 0o600)
}

// listTree returns the sizes of the files directly in dir at rev, keyed by
// path. An LFS file reports its object size, not its pointer's.
func listTree(client *http.Client, repo, rev, dir string) (map[string]int64, error) {
	url := fmt.Sprintf("https://huggingface.co/api/models/%s/tree/%s/%s", repo, rev, dir)

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	// #nosec G704 -- URL is built from pinned model config references against huggingface.co.
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxListing+1))
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, errors.New(url + ": " + resp.Status)
	}

	if len(body) > maxListing {
		return nil, errors.New(url + ": listing too large")
	}

	var entries []treeEntry

	err = json.Unmarshal(body, &entries)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", url, err)
	}

	sizes := make(map[string]int64, len(entries))

	for _, e := range entries {
		if e.Type != "file" {
			continue
		}

		sizes[e.Path] = e.Size
		if e.LFS != nil && e.LFS.Size > 0 {
			sizes[e.Path] = e.LFS.Size
		}
	}

	return sizes, nil
}
