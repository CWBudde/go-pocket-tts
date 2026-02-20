package server

import (
	"testing"

	"github.com/example/go-pocket-tts/internal/config"
)

func TestChooseWorkerLimit(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Server.Workers = 3
	cfg.TTS.Concurrency = 7

	if got := chooseWorkerLimit(cfg, "native"); got != 0 {
		t.Fatalf("native backend should disable worker pool, got %d", got)
	}
	if got := chooseWorkerLimit(cfg, "cli"); got != 3 {
		t.Fatalf("cli backend should use server workers first, got %d", got)
	}

	cfg.Server.Workers = 0
	if got := chooseWorkerLimit(cfg, "cli"); got != 7 {
		t.Fatalf("cli backend should fall back to tts concurrency, got %d", got)
	}
}
