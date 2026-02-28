package tensor

import (
	"sync"
	"sync/atomic"
)

// workers controls goroutine parallelism for tensor math kernels such as
// Linear and MatMul. Values <= 1 disable parallel execution.
var workers atomic.Int32

func init() {
	workers.Store(1)
}

// SetWorkers sets the maximum number of goroutines used by tensor kernels.
// n <= 1 disables kernel parallelism.
func SetWorkers(n int) {
	const maxInt32 = int(^uint32(0) >> 1)

	if n < 1 {
		n = 1
	}

	if n > maxInt32 {
		n = maxInt32
	}

	workers.Store(int32(n))
}

func getWorkers() int {
	n := int(workers.Load())
	if n < 1 {
		return 1
	}

	return n
}

func parallelFor(n, maxWorkers int, fn func(lo, hi int)) {
	if n <= 0 {
		return
	}

	if maxWorkers <= 1 || n == 1 {
		fn(0, n)
		return
	}

	if maxWorkers > n {
		maxWorkers = n
	}

	chunk := (n + maxWorkers - 1) / maxWorkers
	var wg sync.WaitGroup

	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)

		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}

	wg.Wait()
}
