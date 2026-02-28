package ops

import (
	"sync"
	"sync/atomic"
)

// convWorkers controls the number of goroutines used by the parallel Conv1D
// and ConvTranspose1D fast paths.  A value of 0 or 1 means sequential
// (default).  Values >= 2 enable parallel execution.
//
// Set via SetConvWorkers, typically wired to --conv-workers.
var convWorkers atomic.Int32

// SetConvWorkers sets the maximum number of goroutines used for parallel
// Conv1D / ConvTranspose1D execution.  n <= 1 disables parallelism.
func SetConvWorkers(n int) {
	const maxInt32 = int(^uint32(0) >> 1)

	if n < 0 {
		n = 0
	}

	if n > maxInt32 {
		n = maxInt32
	}

	convWorkers.Store(int32(n))
}

// getConvWorkers returns the current worker count (0 or 1 -> sequential).
func getConvWorkers() int { return int(convWorkers.Load()) }

// parallelFor splits the range [0, n) into chunks and runs fn(lo, hi)
// concurrently.  When workers <= 1 the call is sequential (no goroutines).
func parallelFor(n, workers int, fn func(lo, hi int)) {
	if workers <= 1 || n <= 1 {
		fn(0, n)
		return
	}

	if workers > n {
		workers = n
	}
	var wg sync.WaitGroup

	chunk := (n + workers - 1) / workers
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

// scratchPool is a size-class pool for reusable []float32 scratch buffers.
// It avoids the multi-MB per-call allocations in the im2col and kernel-repack
// paths (conv1DFastGroups1, convTranspose1DGroups1).
//
// Size classes are powers of two from 2^10 (1 Ki) to 2^26 (64 Mi floats ~= 256 MB).
// A request for n floats rounds up to the next power-of-two class.
var scratchPools [17]sync.Pool // indices 10..26 -> pools[0..16]

// getScratch returns a zeroed []float32 of exactly n elements from the pool.
// The caller MUST call putScratch when done.
func getScratch(n int) []float32 {
	cls := scratchClass(n)
	sz := 1 << (cls + 10)
	// If the rounded-up class size is smaller than n (overflow past maxPoolClass),
	// fall back to a plain allocation - it will not be pooled on return.
	if sz < n {
		return make([]float32, n)
	}

	if v := scratchPools[cls].Get(); v != nil {
		buf, ok := v.([]float32)
		if !ok {
			return make([]float32, n)
		}
		// The backing array may be larger than n; re-slice and zero.
		buf = buf[:n]
		for i := range buf {
			buf[i] = 0
		}

		return buf
	}
	// Allocate at the class size so the buffer is reusable for smaller requests
	// in the same class.
	buf := make([]float32, sz)

	return buf[:n]
}

// putScratch returns a buffer obtained from getScratch back to the pool.
// Oversized buffers (that bypassed the pool in getScratch) are silently dropped.
func putScratch(buf []float32) {
	c := cap(buf)

	cls := scratchClass(c)
	if 1<<(cls+10) < c {
		return // oversized - let GC reclaim it
	}
	// Restore full backing-array capacity so the next getScratch can re-slice.
	buf = buf[:c]
	scratchPools[cls].Put(buf)
}

// scratchClass returns the pool index for a buffer of n elements.
func scratchClass(n int) int {
	if n <= 1<<10 {
		return 0
	}
	// Bit length of (n-1) gives the exponent for the next power of two.
	bits := 0

	v := n - 1
	for v > 0 {
		v >>= 1
		bits++
	}

	cls := min(max(bits-10, 0), 16)

	return cls
}
