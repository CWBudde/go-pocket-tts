set shell := ["bash", "-uc"]

# Default recipe - show available commands
default:
    @just --list

# Format all code using treefmt
fmt:
    treefmt --allow-missing-formatter

# Check if code is formatted correctly
check-formatted:
    treefmt --allow-missing-formatter --fail-on-change

# Run linters
lint:
    GOCACHE="${GOCACHE:-/tmp/gocache}" GOMODCACHE="${GOMODCACHE:-/tmp/gomodcache}" GOLANGCI_LINT_CACHE="${GOLANGCI_LINT_CACHE:-/tmp/golangci-lint-cache}" golangci-lint run --timeout=2m ./...

# Run linters with auto-fix
lint-fix:
    GOCACHE="${GOCACHE:-/tmp/gocache}" GOMODCACHE="${GOMODCACHE:-/tmp/gomodcache}" GOLANGCI_LINT_CACHE="${GOLANGCI_LINT_CACHE:-/tmp/golangci-lint-cache}" golangci-lint run --fix --timeout=2m ./...

# Ensure go.mod is tidy
check-tidy:
    go mod tidy
    git diff --exit-code go.mod go.sum

# Run all tests
test:
    go test -v ./...

# Cross-compile and run tensor tests on ARM64 via QEMU
test-arm64:
    GOARCH=arm64 go test -exec "qemu-aarch64-static" -v ./internal/runtime/tensor/

# Run tests with race detector
test-race:
    go test -race ./...

# Run dedicated js/wasm decode benchmarks (mimi_only + decode_stage).
# Usage:
#   just bench-wasm-decode
#   POCKETTTS_BENCH_DECODE_FRAMES=4,8,16 just bench-wasm-decode 1x 1 bench/results/wasm/wasm-decode.txt
bench-wasm-decode benchtime="3x" count="3" out="bench/results/wasm/wasm-decode-bench.txt":
    mkdir -p "$(dirname {{out}})"
    env -i \
        PATH="/usr/bin:/bin:$(go env GOROOT)/bin" \
        HOME="$HOME" \
        TMPDIR="${TMPDIR:-/tmp}" \
        GOPATH="$(go env GOPATH)" \
        GOMODCACHE="$(go env GOMODCACHE)" \
        GOCACHE="${GOCACHE:-/tmp/go-build}" \
        GOOS=js \
        GOARCH=wasm \
        go test \
            -exec "$(go env GOROOT)/lib/wasm/go_js_wasm_exec" \
            ./internal/native \
            -run '^$$' \
            -bench '^BenchmarkWASMDecode$$' \
            -benchmem \
            -benchtime={{benchtime}} \
            -count={{count}} | tee {{out}}

# Run stage profiler with AVX2/FMA (asm tag).
# Usage:
#   just bench-stageprof-asm
#   just bench-stageprof-asm 10 2 "Hello from PocketTTS in the browser." bench/results/stageprof/custom_asm.txt
bench-stageprof-asm runs="5" warmup="1" text="Hello from PocketTTS in the browser." out="":
    mkdir -p bench/results/stageprof
    out_file="{{out}}"; if [ -z "$out_file" ]; then out_file="bench/results/stageprof/stageprof_$(date -u +%Y%m%d-%H%M%S)_asm.txt"; fi; GOCACHE="${GOCACHE:-/tmp/go-build}" go run -tags asm ./bench/stageprof --text "{{text}}" --warmup {{warmup}} --runs {{runs}} | tee "$out_file"; echo "saved: $out_file"

# Run stage profiler with AVX2/FMA disabled.
# Usage:
#   just bench-stageprof-noavx
#   just bench-stageprof-noavx 10 2 "Hello from PocketTTS in the browser." bench/results/stageprof/custom_noavx.txt
bench-stageprof-noavx runs="5" warmup="1" text="Hello from PocketTTS in the browser." out="":
    mkdir -p bench/results/stageprof
    out_file="{{out}}"; if [ -z "$out_file" ]; then out_file="bench/results/stageprof/stageprof_$(date -u +%Y%m%d-%H%M%S)_noavx.txt"; fi; GOCACHE="${GOCACHE:-/tmp/go-build}" GODEBUG=cpu.avx2=off,cpu.fma=off go run ./bench/stageprof --text "{{text}}" --warmup {{warmup}} --runs {{runs}} | tee "$out_file"; echo "saved: $out_file"

# Run tests with coverage
test-coverage:
    go test -v -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out -o coverage.html

# Run all checks (formatting, linting, tests, tidiness)
ci: check-formatted test lint check-tidy

# Build all command line tools
build:
    go build -o pockettts ./cmd/pockettts
    go build -o pockettts-tools ./cmd/pockettts-tools
    go build -o pockettts-wasm ./cmd/pockettts-wasm

# Clean build artifacts
clean:
    rm -f coverage.out coverage.html pockettts pockettts-tools pockettts-wasm

# Generate a "Hello, world!" WAV file using the safetensors-native backend.
# Builds the binary, downloads models/voices if missing, then synthesizes.
generate:
    @echo "==> Building pockettts..."
    go build -o pockettts ./cmd/pockettts
    @echo "==> Downloading models (skipped if already present)..."
    ./pockettts model download
    @echo "==> Ensuring tokenizer.model is available..."
    @if [ ! -f "models/tokenizer.model" ]; then \
        echo "    tokenizer.model missing; downloading from ungated repo..."; \
        ./pockettts model download --hf-repo kyutai/pocket-tts-without-voice-cloning --out-dir models; \
    else \
        echo "    tokenizer.model already present; skipping extra download."; \
    fi
    @echo "==> Downloading voice embeddings (skipped if already present)..."
    @for voice in alba marius javert jean fantine cosette eponine azelma; do \
        if [ ! -f "voices/$voice.safetensors" ]; then \
            echo "    Downloading $voice..."; \
            hf download kyutai/pocket-tts-without-voice-cloning "embeddings/$voice.safetensors" --local-dir /tmp/pockettts-voices/ && \
            cp "/tmp/pockettts-voices/embeddings/$voice.safetensors" "voices/$voice.safetensors"; \
        fi; \
    done
    @echo "==> Synthesizing 'Hello, world!'..."
    @echo "==> Removing old output (if any)..."
    rm -f hello-world.wav
    ./pockettts synth \
        --backend native \
        --voice alba \
        --text "Hello, world!" \
        --out hello-world.wav
    @echo "==> Done: hello-world.wav"
