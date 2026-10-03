# PLAN

> Open work items for go-pocket-tts. The core TTS pipeline (native-safetensors backend) is complete and default.
> The current focus is **resyncing with upstream pocket-tts v3.3.0** (multi-language models, German first) and
> getting a clean baseline on a fresh machine. Older hardening and performance items are kept at the end.

## Upstream Sync Status

|             | Commit                                               | Version | Date       |
| ----------- | ---------------------------------------------------- | ------- | ---------- |
| Last synced | `2dff8a2d1b3b21bf44ecf0084cc8ce79ab6d6bba`           | 2.1.0   | 2026-05-05 |
| Target      | `41cbc84` (re-check `git ls-remote` before starting) | 3.3.0   | 2026-09/10 |

Delta: 98 commits. Ignore `training/` (new in #244), README community-model listings and CI-only changes.
The relevant upstream files are `pocket_tts/{models/tts_model.py, models/flow_lm.py, modules/{transformer,attention,mlp,text_conditioner}.py, text_chunking.py, data/audio_utils.py, utils/{config,utils}.py, default_parameters.py, config/*.yaml, main.py}`.

Rules of thumb for this sync:

- The Go port is currently wired to `tts_b6369a24` (= upstream `english_2026-01`). Keep that model working
  until the new config layer can express its differences (`pad_with_spaces`, Mimi `inner_dim: 512`,
  no BOS-before-voice).
- Every numerics change (GELU, EOS, fade-in) invalidates existing parity fixtures, so regenerate them after each slice.
- Run `go test ./...` after every slice.

---

## Phase 0 — New-Machine Baseline

Fresh macOS arm64 machine (Go 1.27.1, uv, gh, golangci-lint, just, treefmt present). `go build`, `go vet`
and `go test -short ./...` pass. 12 tests skip (10 because ORT is missing, 2 because `POCKETTTS_NATIVE_PY_FIXTURE` is unset).
The local `models/tts_b6369a24.safetensors` (checksum OK), `models/tokenizer.model` and 8 English voices are present.

- [x] **Fix `doctor` / `model verify` false failure.** `internal/safetensors/reader.go` `requiredModelKeys`
      lists keys that don't exist in real checkpoints (`text_emb.weight`, `flow_transformer.*.q_proj`,
      `lsd_decode.*`, `mimi_decode.*`). Replace them with real names, e.g.
      `flow_lm.conditioner.embed.weight`, `flow_lm.transformer.layers.0.self_attn.in_proj.weight`,
      a `flow_lm.flow_net.*` key, `mimi.decoder.model.0.conv.weight`. Verify them against the real header.
      Add a test that reads the header of the real local model when present (skip otherwise). This currently breaks
      step 3 of README "Get Started" on every machine.
      (2026-10-03) — `requiredModelKeys` now checks 5 keys shared by every upstream config (verified against the
      `english_2026-01` and `german` headers); `TestValidateModelKeys_RealModel` validates the local checkpoint;
      `doctor` and `model verify` pass.
- [x] **Setup docs** (`README.md` "Get Started", `docs/INSTALL.md`):
  - [x] Install the HF CLI (`uv tool install huggingface_hub` → `hf`), which `just generate` needs for voices
  - [x] Install `prettier` (`npm i -g prettier` or `brew install prettier`): `treefmt` / `just fmt` / `just ci` fail without it
  - [x] Download voices as an explicit step
  - [x] Optional ORT on macOS: `brew install onnxruntime` + `POCKETTTS_ORT_LIB=/opt/homebrew/lib/libonnxruntime.dylib`
  - [x] Note that the ungated repo `kyutai/pocket-tts-without-voice-cloning` is enough for precomputed voices;
        `HF_TOKEN` and accepting the terms are only needed for the gated `kyutai/pocket-tts`
        (voice-cloning weights)
  - [x] Parity setup: clone upstream at the pinned commit into `original/pockettts`, then `uv sync`. Mention
        `exclude-newer = "7 days"` and that the dev group is now heavy (torchaudio, transformers, utmos)
  - (2026-10-03) — README "Get Started" voice step + "Development tools", gated/ungated note in Quickstart;
    `docs/INSTALL.md` Homebrew ORT and "Parity setup (Python reference)" sections
- [ ] Refresh the `original/pockettts` snapshot to the target commit, and `original/xn` if useful (the xn Rust
      port moved to `gradium-ai/xn-ptts`, #330). (2026-10-03) — deferred to Phase 8: a 3.x snapshot produces
      parity fixtures with tanh GELU etc. that the Go numerics don't match yet. Read upstream files at the
      target commit via `gh api repos/kyutai-labs/pocket-tts/contents/<path>?ref=41cbc84…` until then.
- [ ] Bump the commit/version pins in `README.md` (says 2.1.0 / `2dff8a2`) and in this file once Phase 8 is green
- [x] Pin `pocket-tts` in `.github/workflows/test-integration.yml` and `model-export.yml`; both were unpinned
      and would silently pick up new defaults. (2026-10-03) — pinned to `2.1.0` (the synced version;
      `scripts/export_onnx.py` does not work with 3.x yet). Bump to 3.3.0 together with Phase 8.
- [x] Lint baseline: golangci-lint 2.13 reported 452 issues on `main`. (2026-10-03) — `exhaustruct_v5` disabled
      like its deprecated predecessor, the remaining 113 findings fixed; `just ci` is green.
- [x] Replace the deprecated `gomodguard` linter with `gomodguard_v2` in `.golangci.yml` (golangci-lint 2.12+ warns)
      (2026-10-03) — `gomodguard` disabled; `gomodguard_v2` stays on via `default: all`; no deprecation warning left.

## Phase 1 — Model Config Layer (prerequisite for everything multi-language)

Upstream selects a model with `--language <name>` → `pocket_tts/config/<name>.yaml`. Go has no equivalent;
constants and paths are hard-coded around `tts_b6369a24`.

- [x] New package `internal/modelcfg` (or extend `internal/model`) with a `ModelConfig` struct that mirrors the
      upstream YAML fields the Go port needs:
  - `weights_path` and `weights_path_without_voice_cloning`, `tokenizer_path`, `tokenizer` kind (`sentencepiece` | `tokenizers`)
  - `flow_lm.transformer.num_layers` (informational; layers are auto-detected), `flow_lm.flow.{type,depth,dim}`
  - `mimi.inner_dim`, `flow_lm.insert_bos_before_voice`
  - `default_temperature` (0.3 for every shipped config), `model_recommended_frames_after_eos`
  - Text options: `pad_with_spaces_for_short_inputs`, `remove_semicolons`, `replace_characters` (map),
    `capitalize_first_letter`, `append_terminal_punctuation`
  - `default_voice`, `voices_revision`
  - (2026-10-03) — `internal/modelcfg`: `ModelConfig` mirrors the full upstream `utils/config.py` schema at
    41cbc84 (strict decode like `extra="forbid"`, required-key check, upstream defaults pre-filled, enum
    validation), plus `NumTimeConds()` and `MimiInnerDim()` (falls back to `seanet.dimension` like upstream).
    `default_voice` / `voices_revision` are Go-only fields (`yaml:"-"`): upstream keeps them outside the YAML
    (`DEFAULT_VOICE_FOR_LANGUAGE`), so the item-2 registry fills them. Tested against three upstream configs in
    `testdata/`; all 19 upstream configs parse.
- [ ] Embed the configs we support (`//go:embed configs/*.yaml`), transcribed from upstream `pocket_tts/config/`:
      `english_2026-01` (current), `english_2026-09` (new upstream default `english`), `english_2026-09_24l`,
      `german`, `german_24l`; later `french`, `italian`, `spanish`, `portuguese`, `dutch` and `*_24l`.
      Rewrite the `hf://…/tokenizer.json` paths to the `.model` sibling until the Phase 5 JSON loader exists.
      Also allow `--model-config <path>` for custom ones.
- [ ] `internal/config/config.go`: add `tts.language` (default `english_2026-01` until Phase 6 switches
      it; `POCKETTTS_TTS_LANGUAGE`, `--language` persistent flag). Model, tokenizer and voice paths
      come from the language unless set explicitly.
- [ ] Change the default `Temperature` from 0.7 to "use `default_temperature` from the model config" (0.3). Keep
      the explicit `--temperature` override.
- [ ] Rename `LSDDecodeSteps` → `SamplerDecodeSteps` (keep the old flag as a hidden deprecated alias, like upstream)
- [ ] `internal/model/manifest.go` + `models/download-manifest.lock.json`: make them per language. Use the upstream layout
      `languages/<lang>/{model.safetensors, tokenizer.model, tokenizer.json, embeddings/*.safetensors}`
      at revision `1e08e6a23401048648a9fdcfde2f89348215c2a7` (ungated repo; gated `kyutai/pocket-tts@3e82814…`
      is optional for cloning). Local layout:
      `models/<lang>/model.safetensors`, `models/<lang>/tokenizer.model`, `voices/<lang>/*.safetensors` + `voices/<lang>/manifest.json`. Keep the current flat layout working as `english_2026-01`.
- [ ] Remove the remaining hard-coded `b6369a24` defaults: `cmd/pockettts-tools/model_download_onnx.go`,
      `model_export.go`, `internal/model/export.go`, `internal/model/onnx_bundle.go`,
      `internal/onnx/voice_encode.go`
- [ ] Remove the remaining hard-coded `voices/manifest.json`: `cmd/pockettts/synth.go`, `cmd/pockettts/doctor.go`,
      `internal/server/server.go`, `web/main.js`

## Phase 2 — Checkpoint / Numerics Parity

- [ ] **tanh GELU (#278).** Upstream now uses `F.gelu(x, approximate="tanh")` in the shared
      `StreamingTransformerLayer` (`modules/transformer.py`), so both the FlowLM backbone and the Mimi
      encoder/decoder transformers use it. Add `geluTanh` (`0.5·x·(1+tanh(√(2/π)·(x+0.044715x³)))`) next to the
      erf versions in `internal/native/tensor_util.go` and switch the call sites in `native/flow_transformer.go`
      and `native/mimi.go`. Confirm the flow-net MLP (`modules/mlp.py`) is unchanged. This also applies to
      `english_2026-01` (upstream runs it with tanh GELU now). Benchmark it: tanh is cheaper than erf.
- [ ] **Mimi `inner_dim: 32`.** Every config except `english_2026-01` uses it. Verify the decode path
      (`latent_to_mimi` / quantizer output projection / upsample) loads and runs with the new shapes. The
      header check suggests only the encoder's downsample changes (512 → 32), but confirm it on a real German
      checkpoint (`flow_lm.speaker_proj_weight` is `[1024,32]`).
- [ ] **`insert_bos_before_voice`.** If set, prepend `flow_lm.bos_before_voice` (`[1,1,1024]`) before the
      voice-prompt embedding (`tts_model.py` ~L1035). This only affects the audio-embedding voice path
      (`VoiceEmbedding` concat in `internal/tts/runtime_native_safetensors.go`); precomputed model states
      already include it.
- [ ] **Sampler head generalisation (#329).** `flow.type ∈ {lsd, flow_matching, drifting}`, with
      `num_time_conds` = 2 / 1 / 0. In `native/flow_net.go`, make `time_embed.N` variable-length (detect it from the
      keys present), and only average the time embeddings when there are more than 0. Add `DriftingDecode`
      (`v_t(x_0)`, one step, no time input) and optionally `OTDecode` (`for i: cur += v_t(i/n, cur)/n`). Dispatch on
      `FlowType` in `native/flow_lm.go`. Target model: `english_drifting_26-09` (lower priority than German).
- [ ] **24-layer variants** need no new modules; layer count auto-detection (`native/flow_transformer.go`) should
      already work. Add a smoke test with a `*_24l` checkpoint (~670 MB, so run it manually or in integration only).
- [ ] **Voice state `pad` key.** Newly exported states carry `transformer.layers.N.self_attn/pad` (int64 `[B]`).
      The Go loader ignores unknown keys; add a test with a real `voices/german/juergen.safetensors`.

## Phase 3 — Generation Loop Parity

All in `internal/tts/runtime_native_safetensors.go`, mirrored in `internal/onnx/generate.go`.

- [ ] **Minimum frames before EOS (#319):** only accept EOS when `step >= 6` (`_MIN_FRAMES_BEFORE_EOS`).
- [ ] **Off-by-one after EOS:** upstream breaks _before_ queueing when `step >= eos_step + frames_after_eos`
      and emits exactly F frames after EOS. Go appends first and emits F+1. Fix it and pin it with a test.
- [ ] **frames_after_eos default** from `model_recommended_frames_after_eos`, falling back to the
      upstream heuristic.
- [ ] **Per-chunk fade-in (#332→#335):** multiply the first 120 samples (`sample_rate/200`, 5 ms) of each
      chunk's first decoded frame by `linspace(0, 1, 120)` (inclusive endpoints). Do it after `MimiDecode` per
      chunk, not as the optional CLI DSP step. Fix `audio/dsp.go` fade so its ramp reaches 1.0, or keep the two separate.
- [ ] Cancellation (#228) is already covered: `ctx` from `r.Context()` is checked every step. Add a test that
      at most one extra step runs after cancel, to match `test_streaming_cancellation.py`.

## Phase 4 — Text Preparation Parity

`internal/text/prepare.go` + `internal/text/chunk.go`; callers are `internal/tts/service.go`, `internal/tts/parity.go` and
`internal/bench/stageprof/stageprof.go`.

- [ ] Introduce `text.Options` built from `ModelConfig`:
  - [ ] `PadShortInputs`: today it's always on; it should only be on for `english_2026-01` (8-space pad for fewer than 5 words)
  - [ ] `CapitalizeFirst` (#307): today it's always on
  - [ ] `RemoveSemicolons`: `;` → `,` (french, german)
  - [ ] `ReplaceCharacters` (#325): translate/delete, collapse whitespace, then
        `([.!?…])\s*[,;:]` → `$1`. If the text ends up empty, return an error (upstream raises `ValueError`).
        German/French/… set: delete `" “ ” „ « » ( ) [ ]`, map `’ ‘` → `'`. Spanish also deletes `¡ ¿`. French also maps `:` → `,`.
  - [ ] `AppendTerminalPunctuation` (#296, #288): terminal set `.!?…`. A trailing weak mark `, ; : - – —` is
        replaced by `.`. Closers `" ' ” ’ ) ] »` stay after the inserted period. Port the 14-case table from
        upstream `tests/test_split_sentences.py`.
- [ ] **Decimal points do not split sentences (#217):** no boundary when the prefix ends with digit + `.` and
      the suffix starts with a digit (`Version 2.0 is out. Pi is 3.14.` → 1 chunk).
- [ ] Re-check chunking against upstream `text_chunking.py`: upstream splits on token boundaries
      (`.!…?`, then `,;:` sub-splits for long sentences), while Go splits on characters. Decide whether to port the
      token-based splitter exactly or document the difference. Prefer porting it for fixture parity.
- [ ] German numbers: upstream does no number expansion, and the German vocab has no digits, so they become byte
      tokens. Keep parity first, then consider optional German number/abbreviation normalisation as a
      Go-only extra behind a flag.

## Phase 5 — Tokenizer

`internal/tokenizer/sentencepiece.go` (uses `vikesh-raj/go-sentencepiece-encoder`).

- [ ] **Byte fallback.** Every shipped tokenizer is Unigram with `byte_fallback: true` (256 `<0xXX>` pieces).
      Today unknown characters become `<unk>` (id 0). For German this hits all digits, `Ä Ö Ü`, `€` and `„ “`.
      Implement it by encoding each unknown character as its UTF-8 bytes → `<0xXX>` ids.
- [ ] **No NFKC.** The released tokenizers use an `identity` normalizer; the Go encoder always applies NFKC.
      Disable it or work around it, and test with `café`, `14½-13½`, `"  hello  "`, `""`, `" "` (upstream
      `test_tokenizer_backends.py`).
- [ ] Probably replace the third-party encoder with our own Unigram Viterbi (vocab + scores + byte fallback,
      ~200 LOC). That gives full control and drops the NFKC dependency.
- [ ] **`tokenizer.json` loader (#317).** Parse `model.vocab` (`[piece, score]` pairs), `unk_id` and
      `byte_fallback`, and reuse the same Viterbi. Pick the loader by file extension in `internal/tts/service.go` and
      `cmd/pockettts-wasm/main_wasm.go`. This is needed for models that may be JSON-only (dutch, re-tokenized
      french `@8843db7`; verify on HF). German still ships `tokenizer.model`, which is identical to the json.

## Phase 6 — German Support (end-to-end)

Upstream config `pocket_tts/config/german.yaml`: `german` (6L, 219 MB, sha256 `9fe42605…7621`) and `german_24l`
(24L, 672 MB, sha256 `78d0155b…9312`), repo `kyutai/pocket-tts-without-voice-cloning@1e08e6a`, path
`languages/german/`. Own tokenizer (`tokenizer.model` sha256 `b3d6fb75…e34b`, 4000 pieces; config pins
`tokenizer.json@4e1e0a3`). 27 precomputed voices; the default is `juergen` (sha256 `65104a25…26d1`). The English-named
voices in that folder are German-model states.

Minimal path (precomputed voices only; needs Phases 1, 3, 4 and Phase 5 byte fallback; tanh GELU from Phase 2):

- [ ] `pockettts model download --language german`: model, `tokenizer.model` and voice embeddings → lock file entries
- [ ] `voices/german/manifest.json` with `juergen` as default (generate it from the HF tree listing)
- [ ] `pockettts synth --language german --voice juergen --text "…"` produces intelligible German
      (listening check plus a Python-reference comparison from Phase 8)
- [ ] German demo text from upstream `default_parameters.py` used as a CLI smoke example
- [ ] `serve --language german`: one language per process (same as upstream `serve`); add `--default-voice`
      (#271: name | local wav/safetensors | URL, resolved at startup, fail fast)
- [ ] `doctor` validates the selected language's files
- [ ] Docs: README section "Languages" with the table of supported configs and a German example

Follow-ups:

- [ ] Multi-language server: `language` field in `ttsRequest`, `/voices?language=`, and a lazy-loaded
      language → `tts.Service` registry with an LRU cap (each model is ~220 MB of weights)
- [ ] Web/WASM: language picker. `web/main.js` hard-codes the English model and tokenizer URLs.
- [ ] `german_24l` support (verify quality and speed; ~3× the weights)
- [ ] Other upstream languages (french, italian, spanish, portuguese, dutch): only config + manifest work
      once German works
- [ ] Switch the default English model to `english_2026-09` (upstream's `english`) once Phases 2–5 have
      parity, and keep `english_2026-01` selectable

## Phase 7 — Voice Cloning (audio → voice state) Updates

- [ ] **`end_on_pause` (#334)** in `internal/audio`: 20 ms RMS frames, keep everything up to the last frame
      within 35 dB of the peak, 20 ms fade-out, append 80 ms of zeros. If the input is all silence, return it unchanged.
      Call it before encoding in `internal/onnx/voice_encode.go`. Port `tests/test_end_on_pause.py`.
- [ ] WAV input: accept 24/32-bit int and float WAVs (`internal/audio/decode.go` rejects anything that isn't 16-bit).
- [ ] Encoder latent dim from config (`mimiEncoderLatentDim = 512` hard-coded in
      `internal/onnx/voice_encode.go`; new models use 32) + `speaker_proj_weight [1024, inner_dim]`
- [ ] Cloning for new models requires the gated `kyutai/pocket-tts` weights (`weights_path`); the ungated
      `weights_path_without_voice_cloning` is enough for precomputed voices only. Make this explicit in `doctor`.
- [ ] `export-voice --language german` (ONNX encoder must be re-exported per language; see Phase 9).
      The native Mimi encoder is still unimplemented (`internal/native/mimi.go`).

## Phase 8 — Parity Fixtures & Tests

- [ ] Fix `scripts/dump_python_parity.py` and `scripts/export_onnx.py` for the new upstream layout:
      `pocket_tts.conditioners.base.TokenizedText` is gone; use `modules/text_conditioner.LUTConditioner`
      (`prepare(str)` → tokens, `forward(tokens)`). Remove the dead beartype shim.
- [ ] Add `--language` to `dump_python_parity.py`. Generate fixtures for `english_2026-01` and `german`:
      tokenizer ids, text embeddings, voice model state, FlowLM prefill/step, latent → mimi → PCM.
      Store small ones in `testdata/` and keep large ones gitignored behind `POCKETTTS_NATIVE_PY_FIXTURE`.
- [ ] Port the upstream test tables as Go table tests:
  - `test_split_sentences.py`: decimals, 14 terminal-punctuation cases, `capitalize_first_letter=False`,
    `replace_characters`, Hindi passthrough
  - `test_generation_regressions.py`: `"hi"` → `"        Hi."` for 2026-01, fade-in only on first frame
  - `test_tokenizer_backends.py`: JSON vs SentencePiece id equality on the edge-case strings
  - `test_end_on_pause.py`, `test_audio.py` (WAV header frame count), `test_streaming_cancellation.py`
  - `test_utils.py`: every embedded config has `default_temperature == 0.3`
- [ ] Attention-mask parity tests with offset/context edge cases (carried over)
- [ ] Bump the `pocket-tts==2.1.0` pin in `test-integration.yml` / `model-export.yml` to the synced version once
      the scripts above work with it
- [ ] Known caveat: ONNX-backed native parity tests can panic inside `onnxruntime-purego` with
      `runtime.AddCleanup`; `go test ./... -skip 'TestParity_.*_VsONNX'` is the workaround. Re-check this on Go 1.27.

## Phase 9 — ONNX Backend

- [ ] The rebuilt `english_2026-01` stateful bundle (`/tmp/pockettts-onnx-english_2026-01-stateful.tar.gz`,
      SHA256 `8d5124e3…70f6`) is **lost** on the new machine; `bundles/onnx-bundles.lock.json` is still empty.
      The local `models/onnx/` export exists. Either re-package from it or re-export after Phase 2 (tanh GELU
      changes the graphs anyway).
- [ ] Re-export per language (`scripts/export_onnx.py --language german`), then publish and update the lock file
- [ ] Known issue: garbled audio at the beginning of longer inputs
- [ ] **Decide:** keep ONNX only as the voice-cloning encoder until the native Mimi encoder exists, or deprecate it.
      Upstream changes now have to be applied twice (Go + re-export), which is a strong argument for
      minimising it.

---

## Carried-Over Items

### Safetensors Hardening

- [ ] Memory-map large files (mmap for files > 64 MiB) with safe cleanup on `Close()`. More relevant now
      with 24L models (~670 MB) and several languages.

### Streaming Audio Generation

True frame-level streaming: decode and flush latent frames as they are generated instead of waiting for the
whole AR loop.

- [ ] Run the AR loop (FlowLM) in a producer goroutine and emit latent frames to a channel
- [ ] Run the Mimi decoder in a consumer goroutine and emit PCM chunks (upstream's decoder thread drains every queued
      latent per call, #300)
- [ ] Requires a stateful Mimi decoder rewrite (it's currently stateless and processes all frames at once)

Note: chunk-level streaming (`/tts/stream`) is already implemented.

### Performance

- [ ] Memory budgeting for model weights, KV cache and per-request buffers (multi-language registry, 24L)
- [ ] Im2col tiling for cache-friendliness on large convolutions (res3: 38400×192 im2col = 30 MB, overflows L3)

---

## Reference Architecture

Shared by all shipped configs (upstream v3.3.0):

- `d_model = 1024`, `num_heads = 16`, `hidden_scale = 4`, `max_period = 10000`
- `num_layers = 6` (`*_24l`: 24); flow head `dim = 512`, `depth = 6`
- `ldim = 32`, `sample_rate = 24000`, `frame_rate = 12.5` (1920 samples/frame)
- `n_bins = 4000` (Unigram, byte fallback, identity normalizer)
- `temperature = 0.3` (was 0.7), `eos_threshold = -4.0`, `sampler_decode_steps = 1` (was `lsd_decode_steps`)
- FFN activation: tanh-approximate GELU (was erf)
- EOS ignored for the first 6 frames; 5 ms fade-in at each chunk start

| Config                                           | Layers | Flow head | Mimi inner_dim | BOS before voice | Pad short inputs | Text options                                      |
| ------------------------------------------------ | ------ | --------- | -------------- | ---------------- | ---------------- | ------------------------------------------------- |
| english_2026-01 (`b6369a24`, current Go default) | 6      | lsd       | 512            | no               | yes              | —                                                 |
| english_2026-04 / `_24l`                         | 6 / 24 | lsd       | 32             | yes              | no               | —                                                 |
| english_2026-09 (= upstream `english`) / `_24l`  | 6 / 24 | lsd       | 32             | yes              | no               | —                                                 |
| english_drifting_26-09                           | 6      | drifting  | 32             | yes              | no               | —                                                 |
| german / german_24l                              | 6 / 24 | lsd       | 32             | yes              | no               | remove_semicolons, replace_characters             |
| french / french_24l                              | 6 / 24 | lsd       | 32             | yes              | no               | remove_semicolons, replace_characters (+ `:`→`,`) |
| italian, portuguese (+`_24l`)                    | 6 / 24 | lsd       | 32             | yes              | no               | replace_characters                                |
| spanish (+`_24l`)                                | 6 / 24 | lsd       | 32             | yes              | no               | replace_characters (+ delete `¡ ¿`)               |
| dutch (+`_24l`)                                  | 6 / 24 | lsd       | 32             | yes              | no               | replace_characters; default voice `daan`          |

ONNX graphs (6 total): `text_conditioner`, `flow_lm_main`, `flow_lm_flow`, `latent_to_mimi`, `mimi_decoder`, `mimi_encoder`
