package model

import (
	"errors"
	"fmt"
	"io"
	"path/filepath"

	"github.com/cwbudde/go-pocket-tts/internal/config"
)

// VoiceTarget says where a voice download goes.
type VoiceTarget struct {
	Language   string // --language whose predefined voices to fetch
	Dir        string // directory for the voice files
	Manifest   string // voice manifest listing them
	WriteIndex bool   // false only for the tracked flat voices/manifest.json
}

// ResolveVoiceTarget targets the configured voice manifest
// (paths.voice_manifest, which follows --language unless set) and its
// directory, or dir/manifest.json when outDirSet.
func ResolveVoiceTarget(cfg config.Config, outDir string, outDirSet bool) (VoiceTarget, error) {
	if cfg.TTS.ModelConfigPath != "" {
		return VoiceTarget{}, errors.New("custom model configs (--model-config) have no predefined voices to download; use --language")
	}

	manifest := filepath.FromSlash(cfg.Paths.VoiceManifest)
	if outDirSet {
		manifest = filepath.Join(outDir, "manifest.json")
	}

	flatDefault := filepath.FromSlash(config.PathsForLanguage(config.DefaultLanguage).VoiceManifest)

	return VoiceTarget{
		Language:   cfg.TTS.Language,
		Dir:        filepath.Dir(manifest),
		Manifest:   manifest,
		WriteIndex: cfg.TTS.Language != FlatLayoutLanguage || filepath.Clean(manifest) != flatDefault,
	}, nil
}

// DownloadVoices fetches the predefined voices of target.Language (every one
// when voices is empty) into target.Dir, records them in its lock file and,
// with target.WriteIndex, adds them to the voice manifest.
func DownloadVoices(target VoiceTarget, voices []string, stdout io.Writer) error {
	manifest, err := VoiceManifestForLanguage(target.Language, voices)
	if err != nil {
		return err
	}

	err = DownloadManifest(DownloadOptions{OutDir: target.Dir, Stdout: stdout}, manifest)
	if err != nil {
		return err
	}

	if !target.WriteIndex {
		return nil
	}

	ids := make([]string, 0, len(manifest.Files))
	for _, f := range manifest.Files {
		ids = append(ids, voiceID(f))
	}

	err = WriteVoiceIndex(target.Manifest, ids)
	if err != nil {
		return err
	}

	if stdout != nil {
		_, _ = fmt.Fprintf(stdout, "wrote voice manifest: %s\n", target.Manifest)
	}

	return nil
}
