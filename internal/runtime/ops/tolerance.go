package ops

import "fmt"

// Tolerance defines acceptable numeric drift versus ONNX reference outputs.
type Tolerance struct {
	Abs float64
	Rel float64
}

// KernelTolerances defines per-kernel parity targets used for native runtime
// validation against ONNX snapshots.
var KernelTolerances = map[string]Tolerance{
	"matmul":          {Abs: 1e-4, Rel: 1e-4},
	"linear":          {Abs: 1e-4, Rel: 1e-4},
	"softmax":         {Abs: 1e-4, Rel: 1e-4},
	"layer_norm":      {Abs: 1e-4, Rel: 1e-4},
	"causal_mask":     {Abs: 0, Rel: 0},
	"rope":            {Abs: 2e-4, Rel: 2e-4},
	"attention":       {Abs: 2e-4, Rel: 2e-4},
	"mlp":             {Abs: 2e-4, Rel: 2e-4},
	"conv1d":          {Abs: 2e-4, Rel: 2e-4},
	"convtranspose1d": {Abs: 2e-4, Rel: 2e-4},
}

func KernelTolerance(name string) (Tolerance, error) {
	t, ok := KernelTolerances[name]
	if !ok {
		return Tolerance{}, fmt.Errorf("ops: no tolerance configured for kernel %q", name)
	}

	return t, nil
}
