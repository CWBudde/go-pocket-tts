#include "textflag.h"

// Register-blocked dot products for MatMulTransB (see gemm_arm64.go).
//
// Each loaded vector of a or b feeds four FMLAs (one per row of the other
// operand), where a lone dot product feeds one FMLA per two loads. Each
// output keeps one float32×4 accumulator; a pairwise FADDP tree reduces four
// accumulators to the four sums of one output row.
//
// Go's arm64 assembler lacks the vector FADD and FADDP mnemonics, so they
// are WORD-encoded like in dot_arm64.s:
//   FADD  Vd.4S, Vn.4S, Vm.4S: 0x4E20D400 | Rm<<16 | Rn<<5 | Rd
//   FADDP Vd.4S, Vn.4S, Vm.4S: 0x6E20D400 | Rm<<16 | Rn<<5 | Rd
// FADDP Vd, Vn, Vm = [n0+n1, n2+n3, m0+m1, m2+m3], so FADDP(FADDP(x0, x1),
// FADDP(x2, x3)) = [Σx0, Σx1, Σx2, Σx3].

// func gemm4x4NEON(a, b *float32, k4, stride int, out *[16]float32)
//
// out[r*4+s] = Σ_{p<k4} a[r*stride+p] * b[s*stride+p] for r, s < 4.
// k4 is a positive multiple of 4.
//
// Registers:
//   R0–R3    a rows 0–3, R4–R7 b rows 0–3 (post-incremented)
//   R8       out, R9 remaining k, R10 row stride in bytes
//   V0–V15   accumulators, V(r*4+s) for output (r, s)
//   V16–V19  a row vectors, V20–V23 b row vectors
TEXT ·gemm4x4NEON(SB), NOSPLIT, $0-40
	MOVD a+0(FP), R0
	MOVD b+8(FP), R4
	MOVD k4+16(FP), R9
	MOVD stride+24(FP), R10
	MOVD out+32(FP), R8

	LSL  $2, R10, R10
	ADD  R10, R0, R1
	ADD  R10, R1, R2
	ADD  R10, R2, R3
	ADD  R10, R4, R5
	ADD  R10, R5, R6
	ADD  R10, R6, R7

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

loop4x4:
	VLD1.P 16(R0), [V16.S4]
	VLD1.P 16(R1), [V17.S4]
	VLD1.P 16(R2), [V18.S4]
	VLD1.P 16(R3), [V19.S4]
	VLD1.P 16(R4), [V20.S4]
	VLD1.P 16(R5), [V21.S4]
	VLD1.P 16(R6), [V22.S4]
	VLD1.P 16(R7), [V23.S4]

	VFMLA V20.S4, V16.S4, V0.S4  // (0,0) += a0 * b0
	VFMLA V21.S4, V16.S4, V1.S4  // (0,1)
	VFMLA V22.S4, V16.S4, V2.S4  // (0,2)
	VFMLA V23.S4, V16.S4, V3.S4  // (0,3)
	VFMLA V20.S4, V17.S4, V4.S4  // (1,0)
	VFMLA V21.S4, V17.S4, V5.S4
	VFMLA V22.S4, V17.S4, V6.S4
	VFMLA V23.S4, V17.S4, V7.S4
	VFMLA V20.S4, V18.S4, V8.S4  // (2,0)
	VFMLA V21.S4, V18.S4, V9.S4
	VFMLA V22.S4, V18.S4, V10.S4
	VFMLA V23.S4, V18.S4, V11.S4
	VFMLA V20.S4, V19.S4, V12.S4 // (3,0)
	VFMLA V21.S4, V19.S4, V13.S4
	VFMLA V22.S4, V19.S4, V14.S4
	VFMLA V23.S4, V19.S4, V15.S4

	SUBS $4, R9, R9
	BNE  loop4x4

	// Row 0: V0 = [Σ(0,0), Σ(0,1), Σ(0,2), Σ(0,3)]
	WORD $0x6E21D400 // FADDP V0.4S, V0.4S, V1.4S
	WORD $0x6E23D442 // FADDP V2.4S, V2.4S, V3.4S
	WORD $0x6E22D400 // FADDP V0.4S, V0.4S, V2.4S
	// Row 1 into V4.
	WORD $0x6E25D484 // FADDP V4.4S, V4.4S, V5.4S
	WORD $0x6E27D4C6 // FADDP V6.4S, V6.4S, V7.4S
	WORD $0x6E26D484 // FADDP V4.4S, V4.4S, V6.4S
	// Row 2 into V8.
	WORD $0x6E29D508 // FADDP V8.4S, V8.4S, V9.4S
	WORD $0x6E2BD54A // FADDP V10.4S, V10.4S, V11.4S
	WORD $0x6E2AD508 // FADDP V8.4S, V8.4S, V10.4S
	// Row 3 into V12.
	WORD $0x6E2DD58C // FADDP V12.4S, V12.4S, V13.4S
	WORD $0x6E2FD5CE // FADDP V14.4S, V14.4S, V15.4S
	WORD $0x6E2ED58C // FADDP V12.4S, V12.4S, V14.4S

	VST1.P [V0.S4], 16(R8)
	VST1.P [V4.S4], 16(R8)
	VST1.P [V8.S4], 16(R8)
	VST1   [V12.S4], (R8)
	RET

// func gemm1x4NEON(a, b *float32, k4, stride int, out *[4]float32)
//
// out[s] = Σ_{p<k4} a[p] * b[s*stride+p] for s < 4. k4 is a positive
// multiple of 4. The main loop takes 8 floats per row and step with two
// accumulator sets (V0–V3, V4–V7), so eight FMLA chains hide the latency.
//
// Registers:
//   R0       a, R4–R7 b rows 0–3 (post-incremented)
//   R8       out, R9 remaining k, R10 row stride in bytes
//   V0–V7    accumulators, V16/V17 a, V20–V27 b
TEXT ·gemm1x4NEON(SB), NOSPLIT, $0-40
	MOVD a+0(FP), R0
	MOVD b+8(FP), R4
	MOVD k4+16(FP), R9
	MOVD stride+24(FP), R10
	MOVD out+32(FP), R8

	LSL  $2, R10, R10
	ADD  R10, R4, R5
	ADD  R10, R5, R6
	ADD  R10, R6, R7

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16

	CMP  $8, R9
	BLT  step4

loop1x8:
	VLD1.P 32(R0), [V16.S4, V17.S4]
	VLD1.P 32(R4), [V20.S4, V21.S4]
	VLD1.P 32(R5), [V22.S4, V23.S4]
	VLD1.P 32(R6), [V24.S4, V25.S4]
	VLD1.P 32(R7), [V26.S4, V27.S4]

	VFMLA V20.S4, V16.S4, V0.S4
	VFMLA V22.S4, V16.S4, V1.S4
	VFMLA V24.S4, V16.S4, V2.S4
	VFMLA V26.S4, V16.S4, V3.S4
	VFMLA V21.S4, V17.S4, V4.S4
	VFMLA V23.S4, V17.S4, V5.S4
	VFMLA V25.S4, V17.S4, V6.S4
	VFMLA V27.S4, V17.S4, V7.S4

	SUB  $8, R9, R9
	CMP  $8, R9
	BGE  loop1x8

step4:
	CBZ  R9, reduce1x4

	VLD1 (R0), [V16.S4]
	VLD1 (R4), [V20.S4]
	VLD1 (R5), [V22.S4]
	VLD1 (R6), [V24.S4]
	VLD1 (R7), [V26.S4]

	VFMLA V20.S4, V16.S4, V0.S4
	VFMLA V22.S4, V16.S4, V1.S4
	VFMLA V24.S4, V16.S4, V2.S4
	VFMLA V26.S4, V16.S4, V3.S4

reduce1x4:
	WORD $0x4E24D400 // FADD V0.4S, V0.4S, V4.4S
	WORD $0x4E25D421 // FADD V1.4S, V1.4S, V5.4S
	WORD $0x4E26D442 // FADD V2.4S, V2.4S, V6.4S
	WORD $0x4E27D463 // FADD V3.4S, V3.4S, V7.4S
	WORD $0x6E21D400 // FADDP V0.4S, V0.4S, V1.4S
	WORD $0x6E23D442 // FADDP V2.4S, V2.4S, V3.4S
	WORD $0x6E22D400 // FADDP V0.4S, V0.4S, V2.4S

	VST1 [V0.S4], (R8)
	RET
