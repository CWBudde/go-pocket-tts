# Voices and Licensing

This project expects voice entries in `voices/manifest.json`.

Voice files may be either upstream PocketTTS model-state `.safetensors` files
or legacy Go `audio_prompt` `.safetensors` files. The current default voices
are upstream model-state files for the `english_2026-01` model family.

Format compatibility:

| Format | Contents | Supported by |
| ------ | -------- | ------------ |
| Upstream model state | Prompted FlowLM transformer state with tensors such as `<module>/cache` and `<module>/offset` | Native Go backend and browser/WASM kernel |
| Legacy `audio_prompt` | Speaker audio prompt embedding from earlier Go tooling | Native Go backend, browser/WASM kernel, and native ONNX voice-injection path |

The native backend accepts both formats so existing local manifests continue to
work. The ONNX backend still expects legacy `audio_prompt` voice embeddings and
rejects full upstream model-state voice files.

Each entry must define:

- `id`: stable voice identifier used by CLI and APIs
- `path`: path to `.safetensors` voice file (relative to manifest location or absolute)
- `license`: source license for the voice asset

License guidance:

- Some voice assets are non-commercial (for example, CC-BY-NC variants).
- Do not use non-commercial voices in commercial deployments.
- Always verify and retain attribution/terms from the original model or dataset provider.
