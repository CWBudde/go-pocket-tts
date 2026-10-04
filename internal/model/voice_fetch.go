package model

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// FetchVoiceOptions configures FetchVoice.
type FetchVoiceOptions struct {
	Client   *http.Client // nil means http.DefaultClient
	CacheDir string       // directory the voice is downloaded into
	Token    string       // Hugging Face token, sent for hf:// references only
	Stdout   io.Writer    // progress output; nil discards it
}

// IsRemoteVoice reports whether ref names a voice to download (a URL or an
// hf:// reference) rather than a voice ID or local file.
func IsRemoteVoice(ref string) bool {
	return strings.Contains(ref, "://")
}

// FetchVoice downloads the voice at ref (https:// or a pinned
// hf://<org>/<repo>/<path>@<revision>) into opts.CacheDir once and returns
// the cached file. A cached file is reused without a request. Remote voices
// have no pinned checksum, so the caller must load the file to validate it.
func FetchVoice(ref string, opts FetchVoiceOptions) (string, error) {
	rawURL, isHF, err := voiceURL(ref)
	if err != nil {
		return "", err
	}

	if opts.CacheDir == "" {
		return "", errors.New("voice cache dir is required")
	}

	sum := sha256.Sum256([]byte(ref))
	cachePath := filepath.Join(opts.CacheDir, hex.EncodeToString(sum[:8])+"-"+voiceFileName(rawURL))

	_, err = os.Stat(cachePath)
	if err == nil {
		return cachePath, nil
	}

	if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("stat cached voice: %w", err)
	}

	err = os.MkdirAll(opts.CacheDir, 0o750)
	if err != nil {
		return "", fmt.Errorf("create voice cache dir: %w", err)
	}

	client := httpsOnlyClient(opts.Client)

	stdout := opts.Stdout
	if stdout == nil {
		stdout = io.Discard
	}

	token, deniedHint := "", ""
	if isHF {
		token, deniedHint = opts.Token, hfTokenHint
	}

	_, _ = fmt.Fprintf(stdout, "download voice %s -> %s\n", ref, cachePath)

	_, err = downloadURL(client, rawURL, ref, ref, token, deniedHint, cachePath, stdout)
	if err != nil {
		return "", fmt.Errorf("download voice %s: %w", ref, err)
	}

	return cachePath, nil
}

// maxVoiceRedirects matches the redirect limit of net/http's default policy.
const maxVoiceRedirects = 10

// httpsOnlyClient returns a copy of client (nil means http.DefaultClient)
// that refuses redirects to anything but https://, so a redirect cannot
// undo voiceURL's scheme check. The client's own redirect policy still runs.
func httpsOnlyClient(client *http.Client) *http.Client {
	if client == nil {
		client = http.DefaultClient
	}

	c := *client
	next := client.CheckRedirect

	c.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if req.URL.Scheme != "https" {
			return fmt.Errorf("voice download redirected to %s; only https:// is allowed", req.URL.Redacted())
		}

		if next != nil {
			return next(req, via)
		}

		if len(via) >= maxVoiceRedirects {
			return fmt.Errorf("stopped after %d redirects", maxVoiceRedirects)
		}

		return nil
	}

	return &c
}

// VoiceRefExt returns the lower-case extension of the file a voice reference
// names: the file path of an hf:// reference (without @revision), the URL
// path of an https:// voice (without query or fragment), else the local path.
func VoiceRefExt(ref string) string {
	name, ext := ref, filepath.Ext

	switch {
	case strings.HasPrefix(ref, "hf://"):
		_, file, _, err := ParseHFRef(ref)
		if err == nil {
			name, ext = file, path.Ext
		}
	case IsRemoteVoice(ref):
		u, err := url.Parse(ref)
		if err == nil {
			name, ext = u.Path, path.Ext
		}
	}

	return strings.ToLower(ext(name))
}

// voiceURL maps ref to the URL to download and reports whether it is an
// hf:// reference. Only https:// and pinned hf:// references are accepted.
func voiceURL(ref string) (string, bool, error) {
	if strings.HasPrefix(ref, "hf://") {
		repo, file, revision, err := ParseHFRef(ref)
		if err != nil {
			return "", false, err
		}

		return resolveURL(repo, ModelFile{Filename: file, Revision: revision}), true, nil
	}

	u, err := url.Parse(ref)
	if err != nil {
		return "", false, fmt.Errorf("voice URL %q: %w", ref, err)
	}

	if u.Scheme != "https" {
		return "", false, fmt.Errorf("voice URL %q: only https:// and hf:// voices are supported", ref)
	}

	if u.Host == "" {
		return "", false, fmt.Errorf("voice URL %q has no host", ref)
	}

	return ref, false, nil
}

// voiceFileName returns the base name of rawURL's path with a .safetensors
// extension, so the cached file is recognised as a voice file.
func voiceFileName(rawURL string) string {
	name := "voice"

	u, err := url.Parse(rawURL)
	if err == nil {
		if base := path.Base(u.Path); base != "." && base != "/" {
			name = base
		}
	}

	if !strings.HasSuffix(name, ".safetensors") {
		name += ".safetensors"
	}

	return name
}
