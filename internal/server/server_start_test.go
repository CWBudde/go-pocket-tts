package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/example/go-pocket-tts/internal/config"
)

func TestStart_CLIBackend_LifecycleHealthAndShutdown(t *testing.T) {
	// Find an available port.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	addr := ln.Addr().String()
	ln.Close() // free it for the server

	cfg := config.DefaultConfig()
	cfg.TTS.Backend = "cli"
	cfg.Server.ListenAddr = addr

	s := New(cfg, nil).WithShutdownTimeout(2 * time.Second)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errCh := make(chan error, 1)

	go func() {
		errCh <- s.Start(ctx)
	}()

	// Wait for the server to be ready.
	client := &http.Client{Timeout: 2 * time.Second}

	var resp *http.Response

	for i := range 50 {
		_ = i

		resp, err = client.Get(fmt.Sprintf("http://%s/health", addr))
		if err == nil {
			break
		}

		time.Sleep(20 * time.Millisecond)
	}

	if err != nil {
		t.Fatalf("server never became ready: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/health status = %d; want 200", resp.StatusCode)
	}

	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode /health: %v", err)
	}

	if body["status"] != "ok" {
		t.Errorf("status = %q; want ok", body["status"])
	}

	// Graceful shutdown.
	cancel()

	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("Start() returned error on shutdown: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Start() did not return within 5s of context cancel")
	}
}

func TestStart_ChooseWorkerLimit(t *testing.T) {
	tests := []struct {
		name    string
		backend string
		workers int
		conc    int
		want    int
	}{
		{"non-cli returns 0", config.BackendNative, 4, 2, 0},
		{"cli uses server workers", "cli", 4, 2, 4},
		{"cli falls back to concurrency", "cli", 0, 3, 3},
		{"cli both zero", "cli", 0, 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := config.DefaultConfig()
			cfg.Server.Workers = tt.workers
			cfg.TTS.Concurrency = tt.conc

			got := chooseWorkerLimit(cfg, tt.backend)
			if got != tt.want {
				t.Fatalf("chooseWorkerLimit = %d; want %d", got, tt.want)
			}
		})
	}
}
