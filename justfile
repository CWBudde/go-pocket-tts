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

# Run tests with race detector
test-race:
    go test -race ./...

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

# Generate a "Hello, world!" WAV file using the native backend.
# Builds the binary, downloads models/voices if missing, then synthesizes.
# Set ORT_LIB to override the ONNX Runtime library path.
generate ORT_LIB="/usr/local/lib/libonnxruntime.so.1.23.0":
    @echo "==> Building pockettts..."
    go build -o pockettts ./cmd/pockettts
    @echo "==> Downloading models (skipped if already present)..."
    ./pockettts model download
    @echo "==> Downloading voice embeddings (skipped if already present)..."
    @for voice in alba marius javert jean fantine cosette eponine azelma; do \
        if [ ! -f "voices/$voice.safetensors" ]; then \
            echo "    Downloading $voice..."; \
            hf download kyutai/pocket-tts-without-voice-cloning "embeddings/$voice.safetensors" --local-dir /tmp/pockettts-voices/ && \
            cp "/tmp/pockettts-voices/embeddings/$voice.safetensors" "voices/$voice.safetensors"; \
        fi; \
    done
    @echo "==> Synthesizing 'Hello, world!'..."
    ORT_LIBRARY_PATH="{{ ORT_LIB }}" ./pockettts synth \
        --backend native \
        --voice alba \
        --text "Hello, world!" \
        --out hello-world.wav
    @echo "==> Done: hello-world.wav"
