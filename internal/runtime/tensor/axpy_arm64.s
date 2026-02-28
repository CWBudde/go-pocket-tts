#include "textflag.h"

// func axpyF32NEON(dst, src *float32, alpha float32, n int)
//
// Computes dst[i] += alpha * src[i] for i in [0, n).
// Preconditions:
//   - n >= 4
//   - n is a multiple of 4
//   - dst and src are valid for at least n float32 elements
//
// ABI0 stack layout:
//   dst   +0(FP)   *float32
//   src   +8(FP)   *float32
//   alpha+16(FP)   float32
//   n    +24(FP)   int

TEXT ·axpyF32NEON(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+24(FP), R2
	FMOVS alpha+16(FP), F0
	VDUP V0.S[0], V1.S4

loop4:
	VLD1 (R1), [V2.S4]
	VLD1 (R0), [V3.S4]
	VFMLA V2.S4, V1.S4, V3.S4
	VST1 [V3.S4], (R0)
	ADD $16, R1
	ADD $16, R0
	SUB $4, R2
	CBNZ R2, loop4

	RET

