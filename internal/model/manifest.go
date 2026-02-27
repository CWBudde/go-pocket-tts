package model

import "fmt"

type Manifest struct {
	Repo  string      `json:"repo"`
	Files []ModelFile `json:"files"`
}

type ModelFile struct {
	Filename  string `json:"filename"`
	Revision  string `json:"revision"`
	SHA256    string `json:"sha256"`
	LocalPath string `json:"local_path,omitempty"` // Override local save path (defaults to Filename).
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

const (
	voiceRepo     = "kyutai/pocket-tts-without-voice-cloning"
	voiceRevision = "d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
)

func VoiceManifest() Manifest {
	voices := []struct {
		name   string
		sha256 string
	}{
		{"alba", "ad234695323e4030336b6afc8a050c97e3110603e11ecd8226d9562488300a50"},
		{"azelma", "ef33fad34437cb187d2702f0a946d8ba7a01efdb8efbc8088c770d49c181ba73"},
		{"cosette", "ca8926c4f234afa9d722173967e7bebdc6269538ca5910d65f41c3c1317717d3"},
		{"eponine", "bb31940f62da665391de139da2e57d740757df26b73d7ec24152c78a3b8ac0c5"},
		{"fantine", "b6918a2ece002d2d9037ff53c4ea38730175e8798786658b0958443edf49d355"},
		{"javert", "2e857904ee76657e083b0e92664d21bd133e37df320af6eb04f752e679422d91"},
		{"jean", "329530f87ce503061acefca8669300963420ff97e43647a326aa46bd987b983c"},
		{"marius", "33f75e45fac0005630671f4b1bb632d51b6a083b18417de94855bbd7596a0630"},
	}

	files := make([]ModelFile, len(voices))
	for i, v := range voices {
		files[i] = ModelFile{
			Filename:  "embeddings/" + v.name + ".safetensors",
			Revision:  voiceRevision,
			SHA256:    v.sha256,
			LocalPath: v.name + ".safetensors",
		}
	}

	return Manifest{Repo: voiceRepo, Files: files}
}
