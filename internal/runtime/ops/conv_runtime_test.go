package ops

import "testing"

func TestSetConvWorkersClamp(t *testing.T) {
	SetConvWorkers(-5)

	if got := getConvWorkers(); got != 0 {
		t.Fatalf("getConvWorkers() = %d, want 0", got)
	}

	const maxInt32 = int(^uint32(0) >> 1)
	SetConvWorkers(maxInt32 + 123)

	if got := getConvWorkers(); got != maxInt32 {
		t.Fatalf("getConvWorkers() = %d, want %d", got, maxInt32)
	}

	SetConvWorkers(1)
}
