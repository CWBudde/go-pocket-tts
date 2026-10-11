package pockettts

import (
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
)

// TestLoadDefaultWorkers: algo-tensor defaults to GOMAXPROCS workers, but
// pockettts keeps its documented default of 1 unless SetWorkers chose one.
func TestLoadDefaultWorkers(t *testing.T) {
	prev, prevSet := tensor.Workers(), workersSet.Load()

	t.Cleanup(func() {
		tensor.SetWorkers(prev)
		workersSet.Store(prevSet)
	})

	workersSet.Store(false)
	tensor.SetWorkers(7)

	_, err := Load("no-such-model", nil, nil)
	if err == nil {
		t.Fatal("Load of an unknown model succeeded")
	}

	if got := tensor.Workers(); got != 1 {
		t.Errorf("workers after the first Load = %d, want the default 1", got)
	}

	SetWorkers(3)

	_, err = LoadDir(t.TempDir(), "no-such-model")
	if err == nil {
		t.Fatal("LoadDir of an unknown model succeeded")
	}

	if got := tensor.Workers(); got != 3 {
		t.Errorf("workers after SetWorkers(3) and LoadDir = %d, want 3", got)
	}
}
