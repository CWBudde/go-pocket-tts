# Stage Profiler

Canonical location for the stage profiler entrypoint:

- `go run -tags asm ./bench/stageprof`
- `go run ./bench/stageprof`

## Recommended usage

Use `just` recipes so outputs are stored under `bench/results/`:

- `just bench-stageprof-asm`
- `just bench-stageprof-noavx`
- `just bench-wasm-decode`

All files in `bench/results/` are gitignored (folder skeleton only is tracked).
