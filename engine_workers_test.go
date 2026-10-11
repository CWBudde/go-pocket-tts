package pockettts

import (
	"sync"
	"testing"

	"github.com/cwbudde/algo-tensor/ops"
	"github.com/cwbudde/algo-tensor/tensor"
)

// resetWorkers forgets the chosen worker count and sets the tensor count to
// n, restoring both counts and the flag when the test ends. ops has no getter
// for the conv count; 0, its start value, disables conv parallelism as 1 does.
func resetWorkers(t *testing.T, n int) {
	t.Helper()

	workersMu.Lock()
	prev, prevSet := tensor.Workers(), workersSet
	workersSet = false
	workersMu.Unlock()

	tensor.SetWorkers(n)

	t.Cleanup(func() {
		workersMu.Lock()
		defer workersMu.Unlock()

		ops.SetConvWorkers(0)
		tensor.SetWorkers(prev)

		workersSet = prevSet
	})
}

// TestLoadDefaultWorkers: algo-tensor defaults to GOMAXPROCS workers, but
// pockettts keeps its documented default of 1 unless SetWorkers chose one.
func TestLoadDefaultWorkers(t *testing.T) {
	resetWorkers(t, 7)

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

// TestLoadDefaultWorkersConcurrent: every first Load returns only after the
// default is set, not just the one that set it.
func TestLoadDefaultWorkersConcurrent(t *testing.T) {
	resetWorkers(t, 7)

	var wg sync.WaitGroup

	got := make([]int, 8)
	for i := range got {
		wg.Go(func() {
			_, _ = Load("no-such-model", nil, nil)
			got[i] = tensor.Workers()
		})
	}

	wg.Wait()

	for i, n := range got {
		if n != 1 {
			t.Errorf("Load %d returned with %d workers, want 1", i, n)
		}
	}
}
