package audio

import (
	"math"
	"slices"
)

// scipy.signal.resample_poly defaults: window ('kaiser', 5.0) and a filter
// half length of 10 taps per unit of max(up, down).
const (
	resamplePolyKaiserBeta  = 5.0
	resamplePolyHalfLenRate = 10
)

// ResamplePoly resamples samples by the rational factor up/down, a port of
// scipy.signal.resample_poly with its default arguments as upstream's
// convert_audio calls it on float32 audio: a Kaiser-windowed (β = 5) low-pass
// FIR with cutoff 1/max(up, down), zero padding, the group delay removed and
// ceil(len·up/down) output samples. Like scipy on float32 input, the taps are
// rounded to float32; the accumulation runs in float64.
//
// The input slice is not modified. up and down must be positive.
func ResamplePoly(samples []float32, up, down int) []float32 {
	g := gcd(up, down)
	up /= g
	down /= g

	if up == down {
		return slices.Clone(samples)
	}

	nIn := len(samples)
	nOut := (nIn*up + down - 1) / down

	maxRate := max(up, down)
	halfLen := resamplePolyHalfLenRate * maxRate
	h := resamplePolyFilter(2*halfLen+1, 1/float64(maxRate), up)

	// scipy prepends nPrePad zeros to h so that output samples land on the
	// filter centre, then drops the first nPreRemove outputs of upfirdn.
	nPrePad := down - halfLen%down
	nPreRemove := (halfLen + nPrePad) / down

	out := make([]float32, nOut)

	for o := range out {
		// Position in the zero-stuffed (×up) input, shifted by the pre-pad.
		pos := (o+nPreRemove)*down - nPrePad

		// Input i contributes tap h[pos-i·up] when 0 ≤ pos-i·up < len(h).
		iHi := min(pos/up, nIn-1)

		iLo := 0
		if lo := pos - len(h) + 1; lo > 0 {
			iLo = (lo + up - 1) / up
		}

		var acc float64
		for i := iLo; i <= iHi; i++ {
			acc += float64(h[pos-i*up]) * float64(samples[i])
		}

		out[o] = float32(acc)
	}

	return out
}

// resamplePolyFilter returns scipy's firwin(numtaps, cutoff, window=('kaiser',
// 5.0)) rounded to float32 and scaled by up in float32, as resample_poly does
// for float32 input.
func resamplePolyFilter(numtaps int, cutoff float64, up int) []float32 {
	alpha := 0.5 * float64(numtaps-1)
	i0Beta := besselI0(resamplePolyKaiserBeta)

	h := make([]float64, numtaps)

	var sum float64

	for n := range h {
		m := float64(n) - alpha
		r := m / alpha
		window := besselI0(resamplePolyKaiserBeta*math.Sqrt(1-r*r)) / i0Beta
		h[n] = cutoff * sinc(cutoff*m) * window
		sum += h[n]
	}

	taps := make([]float32, numtaps)
	for n, v := range h {
		taps[n] = float32(v/sum) * float32(up)
	}

	return taps
}

// sinc is numpy's normalized sinc, sin(πx)/(πx).
func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}

	return math.Sin(math.Pi*x) / (math.Pi * x)
}

// besselI0 is the modified Bessel function of the first kind, order 0, by its
// power series Σ ((x/2)^k / k!)², which converges quickly for the small
// arguments of a Kaiser window.
func besselI0(x float64) float64 {
	q := x * x / 4
	term, sum := 1.0, 1.0

	for k := 1.0; term > sum*1e-17; k++ {
		term *= q / (k * k)
		sum += term
	}

	return sum
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}

	return a
}
