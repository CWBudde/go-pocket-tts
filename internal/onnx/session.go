package onnx

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

type NodeInfo struct {
	Name  string `json:"name"`
	DType string `json:"dtype"`
	Shape []any  `json:"shape"`
}

type Session struct {
	Name string
	Path string

	Inputs  []NodeInfo
	Outputs []NodeInfo
}

type SessionManager struct {
	mu       sync.RWMutex
	sessions map[string]Session
	order    []string
}

var (
	sessionMgrOnce sync.Once
	sessionMgr     *SessionManager
	errSessionMgr  error
)

type onnxManifest struct {
	Graphs []onnxGraph `json:"graphs"`
}

type onnxGraph struct {
	Name     string     `json:"name"`
	Filename string     `json:"filename"`
	Inputs   []NodeInfo `json:"inputs"`
	Outputs  []NodeInfo `json:"outputs"`
}

func NewSessionManager(manifestPath string) (*SessionManager, error) {
	if manifestPath == "" {
		return nil, errors.New("manifest path is required")
	}

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read ONNX manifest: %w", err)
	}

	var manifest onnxManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("decode ONNX manifest: %w", err)
	}

	if len(manifest.Graphs) == 0 {
		return nil, errors.New("ONNX manifest has no graphs")
	}

	baseDir := filepath.Dir(manifestPath)
	sm := &SessionManager{
		sessions: make(map[string]Session, len(manifest.Graphs)),
		order:    make([]string, 0, len(manifest.Graphs)),
	}

	for _, g := range manifest.Graphs {
		if g.Name == "" {
			return nil, errors.New("manifest graph has empty name")
		}

		if g.Filename == "" {
			return nil, fmt.Errorf("manifest graph %q has empty filename", g.Name)
		}

		if _, exists := sm.sessions[g.Name]; exists {
			return nil, fmt.Errorf("duplicate session name %q in manifest", g.Name)
		}

		sessionPath := g.Filename
		if !filepath.IsAbs(sessionPath) {
			sessionPath = filepath.Join(baseDir, g.Filename)
		}

		sessionPath = filepath.Clean(sessionPath)
		if _, err := os.Stat(sessionPath); err != nil {
			return nil, fmt.Errorf("session file for %q: %w", g.Name, err)
		}

		session := Session{
			Name:    g.Name,
			Path:    sessionPath,
			Inputs:  append([]NodeInfo(nil), g.Inputs...),
			Outputs: append([]NodeInfo(nil), g.Outputs...),
		}
		sm.sessions[g.Name] = session
		sm.order = append(sm.order, g.Name)

		slog.Info(
			"loaded ONNX session",
			"name", g.Name,
			"path", sessionPath,
			"inputs", nodeNames(g.Inputs),
			"outputs", nodeNames(g.Outputs),
		)
	}

	return sm, nil
}

// LoadSessionsOnce loads the ONNX manifest exactly once per process.
// Reloading is intentionally not supported in MVP; restart the process to reload.
func LoadSessionsOnce(manifestPath string) (*SessionManager, error) {
	sessionMgrOnce.Do(func() {
		sessionMgr, errSessionMgr = NewSessionManager(manifestPath)
	})

	if errSessionMgr != nil {
		return nil, errSessionMgr
	}

	return sessionMgr, nil
}

func (m *SessionManager) Session(name string) (Session, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	s, ok := m.sessions[name]

	return s, ok
}

func (m *SessionManager) Sessions() []Session {
	m.mu.RLock()
	defer m.mu.RUnlock()

	out := make([]Session, 0, len(m.order))
	for _, name := range m.order {
		s := m.sessions[name]
		s.Inputs = append([]NodeInfo(nil), s.Inputs...)
		s.Outputs = append([]NodeInfo(nil), s.Outputs...)
		out = append(out, s)
	}

	return out
}

func nodeNames(nodes []NodeInfo) string {
	if len(nodes) == 0 {
		return ""
	}

	names := make([]string, 0, len(nodes))
	for _, n := range nodes {
		names = append(names, n.Name)
	}

	return strings.Join(names, ",")
}
