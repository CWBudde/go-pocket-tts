#include "textflag.h"

// func dotF32AVX2(a, b *float32, n int) float32
//
// Computes the dot product of n float32 values starting at a[0] and b[0]
// using AVX2 + FMA. n must be >= 1; caller guarantees this and that both
// pointers are valid for at least n elements.
//
// ABI0 stack layout (amd64, all args on stack via FP):
//   a   +0(FP)   *float32   8 bytes
//   b   +8(FP)   *float32   8 bytes
//   n  +16(FP)   int        8 bytes
//   ret+24(FP)   float32    4 bytes
//   ───────────────────────────────
//   total argsize: 28 bytes
//
// Register allocation:
//   SI   pointer into a (advances 32 B per 8 floats)
//   DI   pointer into b (advances 32 B per 8 floats)
//   CX   remaining element count
//   Y0–Y3  four float32×8 accumulators (main 4-wide loop)
//   Y4–Y7  load temporaries
//
// Design notes:
//   The inner loop is unrolled 4× (32 floats/iter) so the CPU can hide the
//   ~5-cycle FMA latency (Alder Lake P-core) by keeping four independent
//   dependency chains in flight.  After the main loop the four accumulators
//   are folded into Y0, then an 8-wide single-step loop drains the remainder
//   down to < 8 elements, which are handled scalar.
//   A final horizontal reduction turns Y0 (8 partial sums) into one float32.
//
// Gotcha: FP refs MUST use named form (a+0(FP), not 0(FP)) or go vet fails.

TEXT ·dotF32AVX2(SB), NOSPLIT, $0-28
    MOVQ a+0(FP),  SI       // SI = &a[0]
    MOVQ b+8(FP),  DI       // DI = &b[0]
    MOVQ n+16(FP), CX       // CX = n (element count)

    // Zero all four YMM accumulators.
    VPXOR Y0, Y0, Y0
    VPXOR Y1, Y1, Y1
    VPXOR Y2, Y2, Y2
    VPXOR Y3, Y3, Y3

    // ── Main loop: 32 floats (4×8) per iteration ──────────────────────────
    // VFMADD231PS mem, yreg, acc  →  acc += yreg * mem  (AT&T operand order)
    CMPQ CX, $32
    JL   tail8

loop32:
    VMOVUPS   0(SI), Y4
    VFMADD231PS   0(DI), Y4, Y0
    VMOVUPS  32(SI), Y5
    VFMADD231PS  32(DI), Y5, Y1
    VMOVUPS  64(SI), Y6
    VFMADD231PS  64(DI), Y6, Y2
    VMOVUPS  96(SI), Y7
    VFMADD231PS  96(DI), Y7, Y3
    ADDQ $128, SI
    ADDQ $128, DI
    SUBQ  $32, CX
    CMPQ  CX, $32
    JGE  loop32

    // Fold four accumulators into Y0.
    VADDPS Y1, Y0, Y0
    VADDPS Y3, Y2, Y2
    VADDPS Y2, Y0, Y0

    // ── Single-step loop: 8 floats per iteration ──────────────────────────
tail8:
    CMPQ CX, $8
    JL   tail1

loop8:
    VMOVUPS 0(SI), Y4
    VFMADD231PS 0(DI), Y4, Y0
    ADDQ $32, SI
    ADDQ $32, DI
    SUBQ  $8, CX
    CMPQ  CX, $8
    JGE  loop8

    // ── Scalar tail: 0–7 remaining elements ───────────────────────────────
    // Accumulate directly into X0 (low 32 bits of Y0); other lanes stay put.
tail1:
    TESTQ CX, CX
    JZ    reduce

loop1:
    VMOVSS      0(SI), X4
    VFMADD231SS 0(DI), X4, X0   // X0[31:0] += X4[31:0] * mem[DI]
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  loop1

    // ── Horizontal reduce: Y0 (8 float32) → scalar ────────────────────────
    //
    // After the loops Y0 holds 8 partial sums.  Scalar tail elements landed
    // in X0[31:0] (the lowest lane of Y0); the remaining lanes retain their
    // SIMD partials — all are summed correctly by the reduction below.
reduce:
    // Step 1: fold upper 128 bits into lower 128 bits.
    VEXTRACTF128 $1, Y0, X1    // X1 = Y0[255:128] (upper 4 floats)
    VADDPS X1, X0, X0          // X0 = X0[127:0] + X1 (element-wise, 4 sums)

    // Step 2: horizontal sum the 4 floats in X0.
    // VHADDPS X0, X0, X0 → X0[0]=X0[1]+X0[0], X0[1]=X0[3]+X0[2], ...
    VHADDPS X0, X0, X0
    // Now X0[0]=a+b, X0[1]=c+d; second VHADDPS finishes the sum.
    VHADDPS X0, X0, X0         // X0[0] = a+b+c+d = final result

    VZEROUPPER                  // clear upper YMM bits before returning to SSE
    VMOVSS X0, ret+24(FP)
    RET
