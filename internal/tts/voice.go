package tts

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

type Voice struct {
	ID      string `json:"id"`
	Path    string `json:"path"`
	License string `json:"license"`
}

type voiceManifest struct {
	Voices []Voice `json:"voices"`
}

type VoiceManager struct {
	manifestPath string
	baseDir      string
	voices       []Voice
	byID         map[string]Voice
}

func NewVoiceManager(manifestPath string) (*VoiceManager, error) {
	if manifestPath == "" {
		return nil, errors.New("manifest path is required")
	}

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read voice manifest: %w", err)
	}

	var manifest voiceManifest

	err = json.Unmarshal(data, &manifest)
	if err != nil {
		return nil, fmt.Errorf("decode voice manifest: %w", err)
	}

	mgr := &VoiceManager{
		manifestPath: manifestPath,
		baseDir:      filepath.Dir(manifestPath),
		voices:       append([]Voice(nil), manifest.Voices...),
		byID:         make(map[string]Voice, len(manifest.Voices)),
	}

	for _, v := range manifest.Voices {
		if v.ID == "" {
			return nil, errors.New("voice manifest contains empty id")
		}

		if v.Path == "" {
			return nil, fmt.Errorf("voice %q has empty path", v.ID)
		}

		if _, exists := mgr.byID[v.ID]; exists {
			return nil, fmt.Errorf("duplicate voice id %q", v.ID)
		}

		mgr.byID[v.ID] = v
	}

	return mgr, nil
}

func (m *VoiceManager) ListVoices() []Voice {
	return append([]Voice(nil), m.voices...)
}

func (m *VoiceManager) ResolvePath(id string) (string, error) {
	v, ok := m.byID[id]
	if !ok {
		return "", fmt.Errorf("unknown voice id %q", id)
	}

	resolved := v.Path
	if !filepath.IsAbs(resolved) {
		resolved = filepath.Join(m.baseDir, resolved)
	}

	resolved = filepath.Clean(resolved)

	_, err := os.Stat(resolved)
	if err != nil {
		return "", fmt.Errorf("voice file for %q: %w", id, err)
	}

	return resolved, nil
}
