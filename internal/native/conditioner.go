package native

import (
	"fmt"

	"github.com/example/go-pocket-tts/internal/runtime/tensor"
)

// LUTConditioner mirrors xn's embedding-based text conditioner.
type LUTConditioner struct {
	embed *tensor.Tensor // [n_bins+1, d_model]
	dim   int64
}

func loadLUTConditioner(vb *VarBuilder) (*LUTConditioner, error) {
	embed, err := vb.Tensor("conditioner.embed.weight")
	if err != nil {
		return nil, err
	}
	shape := embed.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("native: conditioner embed must be rank-2, got %v", shape)
	}
	return &LUTConditioner{embed: embed, dim: shape[1]}, nil
}

// EmbedTokens returns [1, T, D] text embeddings.
func (c *LUTConditioner) EmbedTokens(tokenIDs []int64) (*tensor.Tensor, error) {
	if c == nil || c.embed == nil {
		return nil, fmt.Errorf("native: conditioner is not initialized")
	}
	if len(tokenIDs) == 0 {
		return tensor.Zeros([]int64{1, 0, c.dim})
	}

	nBins := c.embed.Shape()[0]
	for i, id := range tokenIDs {
		if id < 0 || id >= nBins {
			return nil, fmt.Errorf("native: token id %d (%d) out of range [0,%d)", i, id, nBins)
		}
	}

	g, err := c.embed.Gather(0, tokenIDs)
	if err != nil {
		return nil, err
	}
	return g.Reshape([]int64{1, int64(len(tokenIDs)), c.dim})
}
