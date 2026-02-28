#include "textflag.h"

// func axpyF32AVX2(dst, src *float32, alpha float32, n int)
//
// Computes dst[i] += alpha * src[i] for i in [0, n).
// Preconditions:
//   - n >= 8
//   - n is a multiple of 8
//   - dst and src are valid for at least n float32 elements
//
// ABI0 stack layout:
//   dst   +0(FP)   *float32
//   src   +8(FP)   *float32
//   alpha+16(FP)   float32
//   n    +24(FP)   int

TEXT ·axpyF32AVX2(SB), NOSPLIT, $0-32
    MOVQ dst+0(FP), DI
    MOVQ src+8(FP), SI
    VMOVSS alpha+16(FP), X0
    MOVQ n+24(FP), CX

    VBROADCASTSS X0, Y1

loop8:
    VMOVUPS 0(SI), Y2
    VMOVUPS 0(DI), Y3
    VMULPS Y1, Y2, Y2
    VADDPS Y2, Y3, Y3
    VMOVUPS Y3, 0(DI)
    ADDQ $32, SI
    ADDQ $32, DI
    SUBQ $8, CX
    JNZ loop8

    VZEROUPPER
    RET

