# Voices and Licensing

This project expects voice entries in `voices/manifest.json`.

Each entry must define:

- `id`: stable voice identifier used by CLI and APIs
- `path`: path to `.safetensors` voice file (relative to manifest location or absolute)
- `license`: source license for the voice asset

License guidance:

- Some voice assets are non-commercial (for example, CC-BY-NC variants).
- Do not use non-commercial voices in commercial deployments.
- Always verify and retain attribution/terms from the original model or dataset provider.
