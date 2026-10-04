package main

import (
	"os"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
	"github.com/cwbudde/go-pocket-tts/internal/tokenizer"
)

func TestNewEngine_UnknownConfig(t *testing.T) {
	_, err := newEngine([]byte("model"), []byte("tokenizer"), "klingon", nil)
	if err == nil || !strings.Contains(err.Error(), "klingon") {
		t.Fatalf("err = %v, want an unknown config error naming klingon", err)
	}
}

func TestNewEngine_RejectsBadTokenizerBeforeModel(t *testing.T) {
	// "" selects the default config, whose tokenizer check runs before the
	// model is opened.
	_, err := newEngine([]byte("model"), []byte("not a tokenizer"), "", nil)
	if err == nil || !strings.Contains(err.Error(), "load tokenizer") {
		t.Fatalf("err = %v, want a tokenizer error", err)
	}
}

// TestNewEngine_German builds the engine from the local german download
// (pockettts model download --language german) and checks that the page's
// config choice reaches the engine.
func TestNewEngine_German(t *testing.T) {
	const dir = "../../models/german/"

	modelBytes, err := os.ReadFile(dir + "model.safetensors")
	if err != nil {
		t.Skipf("german model not downloaded: %v", err)
	}

	tokBytes, err := os.ReadFile(dir + "tokenizer.json")
	if err != nil {
		t.Skipf("german tokenizer not downloaded: %v", err)
	}

	var stages []string

	e, err := newEngine(modelBytes, tokBytes, "german", func(stage string, _, _ int, _ string) {
		stages = append(stages, stage)
	})
	if err != nil {
		t.Fatalf("newEngine: %v", err)
	}
	defer e.runtime.Close()

	if e.name != "german" || !strings.Contains(e.model.FlowLM.LookupTable.TokenizerPath, "/german/") {
		t.Fatalf("engine config = %q (tokenizer %s), want german", e.name, e.model.FlowLM.LookupTable.TokenizerPath)
	}

	if !slices.Contains(stages, "tokenizer") || !slices.Contains(stages, "load") {
		t.Errorf("progress stages = %v", stages)
	}

	cfg, err := modelcfg.Lookup("german")
	if err != nil {
		t.Fatal(err)
	}

	want, err := tokenizer.Load(dir+"tokenizer.json", cfg.FlowLM.LookupTable.NBins)
	if err != nil {
		t.Fatal(err)
	}

	wantIDs, err := want.Encode(cfg.DefaultText)
	if err != nil {
		t.Fatal(err)
	}

	gotIDs, err := e.tokenizer.Encode(cfg.DefaultText)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(gotIDs, wantIDs) {
		t.Errorf("tokens = %v, want %v", gotIDs, wantIDs)
	}
}

// TestNewEngine_ReleasesCheckpointBytes checks that the engine drops the
// checkpoint bytes once the weights are decoded, so the 24-layer models fit in
// 4 GB of WASM memory.
func TestNewEngine_ReleasesCheckpointBytes(t *testing.T) {
	const dir = "../../models/german/"

	modelBytes, err := os.ReadFile(dir + "model.safetensors")
	if err != nil {
		t.Skipf("german model not downloaded: %v", err)
	}

	tokBytes, err := os.ReadFile(dir + "tokenizer.json")
	if err != nil {
		t.Skipf("german tokenizer not downloaded: %v", err)
	}

	released := make(chan struct{})
	runtime.AddCleanup(&modelBytes[0], func(ch chan struct{}) { close(ch) }, released)

	e, err := newEngine(modelBytes, tokBytes, "german", nil)
	if err != nil {
		t.Fatalf("newEngine: %v", err)
	}
	defer e.runtime.Close()

	modelBytes = nil //nolint:wastedassign // drop the test's own reference

	for range 20 {
		runtime.GC()

		select {
		case <-released:
			return
		case <-time.After(50 * time.Millisecond):
		}
	}

	t.Fatal("the engine keeps the checkpoint bytes alive")
}
