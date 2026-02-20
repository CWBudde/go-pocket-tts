package model

import "fmt"

type Manifest struct {
	Repo  string      `json:"repo"`
	Files []ModelFile `json:"files"`
}

type ModelFile struct {
	Filename string `json:"filename"`
	Revision string `json:"revision"`
	SHA256   string `json:"sha256"`
}

func PinnedManifest(repo string) (Manifest, error) {
	switch repo {
	case "kyutai/pocket-tts":
		return Manifest{
			Repo: repo,
			Files: []ModelFile{
				{
					Filename: "tts_b6369a24.safetensors",
					Revision: "427e3d61b276ed69fdd03de0d185fa8a8d97fc5b",
					// The gated repo checksum is resolved from HF metadata at runtime
					// and then persisted into a local lock manifest.
					SHA256: "",
				},
			},
		}, nil
	case "kyutai/pocket-tts-without-voice-cloning":
		return Manifest{
			Repo: repo,
			Files: []ModelFile{
				{
					Filename: "tts_b6369a24.safetensors",
					Revision: "d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3",
					SHA256:   "58aa704a88faad35f22c34ea1cb55c4c5629de8b8e035c6e4936e2673dc07617",
				},
				{
					Filename: "tokenizer.model",
					Revision: "d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3",
					SHA256:   "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6",
				},
			},
		}, nil
	default:
		return Manifest{}, fmt.Errorf("no pinned manifest for repo %q", repo)
	}
}
