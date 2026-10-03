// Command genchecksums regenerates internal/model/language_checksums.json:
// the SHA256 of every ungated file a per-language download fetches (model,
// tokenizer and predefined voices) at the revisions the embedded model
// configs pin. LFS files take it from the Hugging Face tree API; small files
// stored in git (tokenizer.json) have none there, so they are downloaded and
// hashed, after checking them against their git blob id. Model and tokenizer
// are also recorded at the voices revision (modelcfg.VoicesRevision).
//
// Run it via `go generate ./internal/model/`.
package main

import (
	"crypto/sha1" // #nosec G505 -- git blob ids are SHA-1; only used to check a download.
	"crypto/sha256"
	"encoding/hex"
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

// maxGitFileSize bounds a downloaded non-LFS file (tokenizer.json is ~250 KB).
const maxGitFileSize = 16 << 20

type treeEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	// OID is the git blob id (SHA-1) of the file.
	OID string `json:"oid"`
	LFS *struct {
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

// addDir lists dir at rev and records the SHA256 of only (when set) or of
// every LFS file in it. A non-LFS only is downloaded and hashed.
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

		var sum string

		switch {
		case e.LFS != nil && e.LFS.OID != "":
			sum = strings.ToLower(e.LFS.OID)
		case only != "":
			sum, err = hashGitFile(client, repo, rev, e)
			if err != nil {
				return err
			}
		default:
			continue
		}

		sums[model.ChecksumKey(repo, e.Path, rev)] = sum
		found = true
	}

	if !found {
		return fmt.Errorf("no LFS files for %s/%s@%s (only=%q)", repo, dir, rev, only)
	}

	return nil
}

// hashGitFile downloads a file stored in git (not LFS), checks it against its
// git blob id and returns its SHA256.
func hashGitFile(client *http.Client, repo, rev string, e treeEntry) (string, error) {
	data, err := get(client, fmt.Sprintf("https://huggingface.co/%s/resolve/%s/%s", repo, rev, e.Path), maxGitFileSize)
	if err != nil {
		return "", err
	}

	blob := sha1.New() // #nosec G401 -- git blob id, not a security check.
	_, _ = fmt.Fprintf(blob, "blob %d\x00", len(data))
	_, _ = blob.Write(data)

	if got := hex.EncodeToString(blob.Sum(nil)); got != strings.ToLower(e.OID) {
		return "", fmt.Errorf("%s/%s@%s: git blob id %s, tree says %s", repo, e.Path, rev, got, e.OID)
	}

	sum := sha256.Sum256(data)

	return hex.EncodeToString(sum[:]), nil
}

func listTree(client *http.Client, repo, rev, dir string) ([]treeEntry, error) {
	url := fmt.Sprintf("https://huggingface.co/api/models/%s/tree/%s/%s", repo, rev, dir)

	body, err := get(client, url, maxGitFileSize)
	if err != nil {
		return nil, err
	}

	var entries []treeEntry

	err = json.Unmarshal(body, &entries)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", url, err)
	}

	return entries, nil
}

// get fetches url and returns at most limit bytes of its body; a longer body
// is an error.
func get(client *http.Client, url string, limit int64) ([]byte, error) {
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

	body, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", url, err)
	}

	if int64(len(body)) > limit {
		return nil, fmt.Errorf("%s: larger than %d bytes", url, limit)
	}

	return body, nil
}
