# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Go CLI and HTTP server for [PocketTTS](https://github.com/kyutai-labs/pocket-tts) text-to-speech. Currently uses the `pocket-tts` Python executable as a subprocess for synthesis; a native ONNX Runtime backend is in progress (Phase 11 in PLAN.md).

**Module:** `github.com/example/go-pocket-tts`
**Go version:** 1.25+

## Build & Development Commands

```bash
# Build
go build -o pockettts ./cmd/pockettts

# Run tests
just test              # or: go test -v ./...
just test-race         # with race detector

# Run a single test
go test -v -run TestFunctionName ./internal/package/...

# Lint
just lint              # or: golangci-lint run --timeout=2m ./...
just lint-fix          # auto-fix

# Format
just fmt               # treefmt

# Full CI check (format + test + lint + tidy)
just ci
```

## Architecture

### CLI Structure (Cobra)

Entry point: `cmd/pockettts/main.go`. Commands are registered in `root.go`.

```
pockettts
├── synth          Text → WAV via pocket-tts subprocess
├── serve          HTTP server (/healthz, /voices, /tts)
├── doctor         Preflight dependency checks
├── health         Server health probe
├── export-voice   Extract voice embedding from WAV
└── model
    ├── download   Fetch from Hugging Face
    ├── export     PyTorch → ONNX conversion
    └── verify     Smoke-test ONNX graphs
```

### Internal Packages

| Package | Purpose |
|---------|---------|
| `config` | Unified config struct loaded via Viper (flags → env → file → defaults). Env prefix: `POCKETTTS_` |
| `audio` | WAV encode/decode using `cwbudde/wav`, DSP chain (normalize, DC-block, fade) using `cwbudde/algo-dsp` |
| `text` | Text normalization and sentence-based chunking for synthesis |
| `tts` | Service orchestration, voice manifest management (VoiceManager) |
| `onnx` | ONNX Runtime bootstrap (`sync.Once` singleton), session management, tensor utilities |
| `model` | Model download (HF with checksum), ONNX export (Python script), verification |
| `server` | HTTP handlers with functional options pattern, semaphore-based worker pool, graceful shutdown |
| `doctor` | System health checks with dependency-injected validators |

### Key Patterns

- **Configuration:** Single `config.Config` struct loaded in root `PersistentPreRunE`, passed to services
- **HTTP server options:** Functional options (`server.Option`) for handler configuration
- **ONNX singleton:** `onnx.Bootstrap()` with `sync.Once`; `onnx.Shutdown()` deferred in `main()`
- **Testing:** Interfaces + mocks for unit tests; integration tests use build tag `integration` and skip gracefully when deps unavailable
- **Processing pipeline:** Text → Normalize → Chunk → Synthesize per chunk → Concatenate PCM → DSP → Encode WAV

### Linting

golangci-lint with: govet, staticcheck, errcheck, ineffassign, unused, gofmt. Config in `.golangci.yml`.

## Development Plan

See `PLAN.md` for the phased implementation roadmap. Phases 0-8 are complete. Current work is in Phases 9-11 (benchmarking, packaging, native ONNX backend).
