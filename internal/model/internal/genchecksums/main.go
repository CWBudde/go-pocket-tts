// Command genchecksums regenerates internal/model/language_checksums.json:
// the SHA256 of every ungated file a per-language download fetches (model,
// tokenizer and predefined voices), read from the Hugging Face tree API at
// the revisions the embedded model configs pin. Model and tokenizer are also
// recorded at the voices revision (modelcfg.VoicesRevision).
//
// Run it via `go generate ./internal/model/`.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"strings"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/model"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
)

type treeEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	LFS  *struct {
		OID string `json:"oid"`
	} `json:"lfs"`
}

func main() {
	out := flag.String("out", "language_checksums.json", "output file")

	flag.Parse()

	err := run(*out)
	if err != nil {
		log.Fatal(err)
	}
}

func run(out string) error {
	client := &http.Client{Timeout: 60 * time.Second}
	sums := map[string]string{}

	for _, language := range modelcfg.Languages() {
		if language == model.FlatLayoutLanguage {
			continue
		}

		cfg, err := modelcfg.Lookup(language)
		if err != nil {
			return err
		}

		for _, ref := range []string{cfg.WeightsPathWithoutVoiceCloning, cfg.FlowLM.LookupTable.TokenizerPath} {
			repo, file, rev, err := model.ParseHFRef(ref)
			if err != nil {
				return fmt.Errorf("%s: %w", language, err)
			}

			// Also record the file at the voices revision, so a test can check
			// that the config pins and that revision serve the same blobs.
			for _, r := range []string{rev, cfg.VoicesRevision} {
				err = addDir(client, sums, repo, r, path.Dir(file), file)
				if err != nil {
					return err
				}
			}
		}

		err = addDir(client, sums, model.VoiceRepo, cfg.VoicesRevision,
			path.Join("languages", language, "embeddings"), "")
		if err != nil {
			return err
		}
	}

	b, err := json.MarshalIndent(sums, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(out, append(b, '\n'), 0o600)
}

// addDir lists dir at rev and records the LFS SHA256 of only (when set) or of
// every file in it.
func addDir(client *http.Client, sums map[string]string, repo, rev, dir, only string) error {
	entries, err := listTree(client, repo, rev, dir)
	if err != nil {
		return err
	}

	found := false

	for _, e := range entries {
		if e.Type != "file" || (only != "" && e.Path != only) {
			continue
		}

		if e.LFS == nil || e.LFS.OID == "" {
			if only != "" {
				return fmt.Errorf("%s/%s@%s is not an LFS file", repo, e.Path, rev)
			}

			continue
		}

		sums[model.ChecksumKey(repo, e.Path, rev)] = strings.ToLower(e.LFS.OID)
		found = true
	}

	if !found {
		return fmt.Errorf("no LFS files for %s/%s@%s (only=%q)", repo, dir, rev, only)
	}

	return nil
}

func listTree(client *http.Client, repo, rev, dir string) ([]treeEntry, error) {
	url := fmt.Sprintf("https://huggingface.co/api/models/%s/tree/%s/%s", repo, rev, dir)

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	// #nosec G704 -- URL is built from pinned model config references against huggingface.co.
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, errors.New(url + ": " + resp.Status + ": " + string(body))
	}

	var entries []treeEntry

	err = json.NewDecoder(resp.Body).Decode(&entries)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", url, err)
	}

	return entries, nil
}
