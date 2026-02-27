#include "textflag.h"

// func dotF32NEON(a, b *float32, n int) float32
//
// Computes the dot product of n float32 values starting at a[0] and b[0]
// using ARM64 NEON instructions. n must be >= 1; caller guarantees this
// and that both pointers are valid for at least n elements.
//
// ABI0 stack layout (arm64, all args on stack via FP):
//   a    +0(FP)   *float32   8 bytes
//   b    +8(FP)   *float32   8 bytes
//   n   +16(FP)   int        8 bytes
//   ret +24(FP)   float32    4 bytes
//   ─────────────────────────────
//   total argsize: 28 bytes
//
// Register allocation:
//   R0   pointer into a (advanced manually)
//   R1   pointer into b (advanced manually)
//   R2   remaining element count
//   V0–V3  four float32×4 accumulators (main 4-wide loop)
//   V4–V7  load temporaries
//   V8/F8  scalar tail accumulator (separate from V0 to avoid zeroing lanes)
//
// Design notes:
//   The inner loop is unrolled 4× (16 floats/iter) so the CPU can hide the
//   ~4-cycle FMLA latency by keeping four independent dependency chains in
//   flight. After the main loop the four accumulators are folded into V0
//   using vector float add, then a 4-wide single-step loop drains the
//   remainder down to < 4 elements, which are handled scalar using F8 as
//   accumulator (not F0, since writing F0/S0 would zero V0[127:32]).
//   A final pairwise reduction turns V0 into one float32.
//
// Instruction encoding notes:
//   Go's arm64 assembler lacks FADD (vector float) and FADDP mnemonics.
//   We use WORD directives with raw ARM64 encodings:
//     FADD Vd.4S, Vn.4S, Vm.4S: 0 1 0 01110 00 1 Rm 110101 Rn Rd
//     FADDP Vd.4S, Vn.4S, Vm.4S: 0 1 1 01110 00 1 Rm 110101 Rn Rd
//     FADDP Sd, Vn.2S:           01 1 11110 00 11000 011010 Rn Rd

TEXT ·dotF32NEON(SB), NOSPLIT, $0-28
	MOVD a+0(FP),  R0       // R0 = &a[0]
	MOVD b+8(FP),  R1       // R1 = &b[0]
	MOVD n+16(FP), R2       // R2 = n (element count)

	// Zero all four NEON accumulators (V0–V3, each 4×float32).
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	// ── Main loop: 16 floats (4×4) per iteration ────────────────────────
	CMP  $16, R2
	BLT  tail4

loop16:
	VLD1 (R0), [V4.S4]
	VLD1 (R1), [V5.S4]
	VFMLA V5.S4, V4.S4, V0.S4       // V0 += V4 * V5
	ADD  $16, R0
	ADD  $16, R1

	VLD1 (R0), [V4.S4]
	VLD1 (R1), [V5.S4]
	VFMLA V5.S4, V4.S4, V1.S4       // V1 += V4 * V5
	ADD  $16, R0
	ADD  $16, R1

	VLD1 (R0), [V6.S4]
	VLD1 (R1), [V7.S4]
	VFMLA V7.S4, V6.S4, V2.S4       // V2 += V6 * V7
	ADD  $16, R0
	ADD  $16, R1

	VLD1 (R0), [V6.S4]
	VLD1 (R1), [V7.S4]
	VFMLA V7.S4, V6.S4, V3.S4       // V3 += V6 * V7
	ADD  $16, R0
	ADD  $16, R1

	SUB  $16, R2
	CMP  $16, R2
	BGE  loop16

	// Fold four accumulators into V0 using vector FLOAT add.
	// FADD V0.4S, V0.4S, V1.4S  (V0 = V0 + V1)
	WORD $0x4E21D400
	// FADD V2.4S, V2.4S, V3.4S  (V2 = V2 + V3)
	WORD $0x4E23D442
	// FADD V0.4S, V0.4S, V2.4S  (V0 = V0 + V2)
	WORD $0x4E22D400

	// ── Single-step loop: 4 floats per iteration ────────────────────────
tail4:
	CMP  $4, R2
	BLT  tail1

loop4:
	VLD1 (R0), [V4.S4]
	VLD1 (R1), [V5.S4]
	VFMLA V5.S4, V4.S4, V0.S4
	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R2
	CMP  $4, R2
	BGE  loop4

	// ── Scalar tail: 0–3 remaining elements ─────────────────────────────
	// We accumulate into F8 (not F0) because writing to F0/S0 would zero
	// V0[127:32], destroying the SIMD partial sums in the upper lanes.
tail1:
	CBZ  R2, reduce
	VEOR V8.B16, V8.B16, V8.B16     // zero scalar accumulator

loop1:
	FMOVS (R0), F4
	FMOVS (R1), F5
	FMULS F5, F4, F4           // F4 = a[i] * b[i]
	FADDS F4, F8, F8           // F8 += F4
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, loop1

	// Merge scalar tail sum (V8) into V0 with vector float add.
	// V8 = [scalar_sum, 0, 0, 0]; V0 = [simd0, simd1, simd2, simd3]
	// FADD V0.4S, V0.4S, V8.4S  (V0 = V0 + V8)
	WORD $0x4E28D400

	// ── Horizontal reduce: V0 (4 float32) → scalar F0 ──────────────────
reduce:
	// Pairwise add: V0.S[0]+V0.S[1] and V0.S[2]+V0.S[3]
	// FADDP V0.4S, V0.4S, V0.4S
	WORD $0x6E20D400
	// Now V0.S[0] = sum01, V0.S[1] = sum23; scalar pairwise add finishes.
	// FADDP S0, V0.2S
	WORD $0x7E30D800
	// F0 (= V0.S[0]) is the final scalar result.

	FMOVS F0, ret+24(FP)
	RET
