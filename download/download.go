// Package download fetches catalog models into a model root, the layout
// pockettts.LoadDir and pockettts.LoadVoiceDir read. Every file is checked
// against its pinned size and SHA-256 before it is renamed into place, so an
// interrupted or corrupted download never replaces a good file.
package download

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	pockettts "github.com/cwbudde/go-pocket-tts"
)

// Progress reports the bytes of one file; Done counts bytes received for
// File so far, Total is its catalog size.
type Progress struct {
	File        string
	Done, Total int64
}

// Options configures Model.
type Options struct {
	// Client performs the requests; nil uses http.DefaultClient.
	Client *http.Client
	// Progress, when set, is called while files are received.
	Progress func(Progress)
}

// Files returns the files of m that Model fetches for voices: the weights,
// the tokenizer and each named voice (the default voice when voices is
// empty).
func Files(m pockettts.Model, voices []string) ([]pockettts.File, error) {
	if len(voices) == 0 {
		voices = []string{m.DefaultVoice}
	}

	files := []pockettts.File{m.Weights, m.Tokenizer}

	for _, id := range voices {
		v, ok := m.Voice(id)
		if !ok {
			return nil, fmt.Errorf("download: model %s has no voice %q", m.Name, id)
		}

		files = append(files, v.File)
	}

	return files, nil
}

// Model fetches the files of m (see Files) below root. Files that are
// already present with the pinned checksum are kept.
func Model(ctx context.Context, root string, m pockettts.Model, voices []string, opts Options) error {
	files, err := Files(m, voices)
	if err != nil {
		return err
	}

	for _, f := range files {
		err = File(ctx, root, f, opts)
		if err != nil {
			return err
		}
	}

	return nil
}

// File fetches f below root unless it is already present with the pinned
// checksum.
func File(ctx context.Context, root string, f pockettts.File, opts Options) error {
	dst := filepath.Join(root, filepath.FromSlash(f.Path))

	ok, err := present(dst, f)
	if err != nil || ok {
		return err
	}

	err = os.MkdirAll(filepath.Dir(dst), 0o755)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}

	part := dst + ".part"

	err = fetch(ctx, part, f, opts)
	if err != nil {
		_ = os.Remove(part)
		return err
	}

	err = os.Rename(part, dst)
	if err != nil {
		_ = os.Remove(part)
		return fmt.Errorf("download: %w", err)
	}

	return nil
}

// present reports whether dst holds f's bytes.
func present(dst string, f pockettts.File) (bool, error) {
	info, err := os.Stat(dst)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}

	if err != nil {
		return false, fmt.Errorf("download: %w", err)
	}

	if info.Size() != f.Size {
		return false, nil
	}

	file, err := os.Open(dst) // #nosec G304 -- dst is a catalog path below the caller's model root.
	if err != nil {
		return false, fmt.Errorf("download: %w", err)
	}
	defer file.Close()

	sum := sha256.New()

	_, err = io.Copy(sum, file)
	if err != nil {
		return false, fmt.Errorf("download: %s: %w", dst, err)
	}

	return hex.EncodeToString(sum.Sum(nil)) == f.SHA256, nil
}

func fetch(ctx context.Context, part string, f pockettts.File, opts Options) error {
	client := opts.Client
	if client == nil {
		client = http.DefaultClient
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, f.URL, nil)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}

	// #nosec G107 G704 -- f.URL is a revision-pinned catalog URL; the body is checked against its pinned SHA-256.
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("download: %s: %w", f.Path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download: %s: %s", f.URL, resp.Status)
	}

	out, err := os.OpenFile(part, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644) // #nosec G302 G304 -- model files are not secret; part is below the caller's model root.
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}

	sum := sha256.New()
	w := &progressWriter{file: f.Path, total: f.Size, report: opts.Progress}

	// One byte more than pinned is enough to tell a wrong file.
	n, err := io.Copy(io.MultiWriter(out, sum, w), io.LimitReader(resp.Body, f.Size+1))

	closeErr := out.Close()

	switch {
	case err != nil:
		return fmt.Errorf("download: %s: %w", f.Path, err)
	case closeErr != nil:
		return fmt.Errorf("download: %s: %w", f.Path, closeErr)
	case n != f.Size:
		return fmt.Errorf("download: %s: got %d bytes, want %d", f.Path, n, f.Size)
	case hex.EncodeToString(sum.Sum(nil)) != f.SHA256:
		return fmt.Errorf("download: %s: checksum mismatch", f.Path)
	}

	return nil
}

type progressWriter struct {
	file   string
	done   int64
	total  int64
	report func(Progress)
}

func (w *progressWriter) Write(p []byte) (int, error) {
	w.done += int64(len(p))
	if w.report != nil {
		w.report(Progress{File: w.file, Done: w.done, Total: w.total})
	}

	return len(p), nil
}
