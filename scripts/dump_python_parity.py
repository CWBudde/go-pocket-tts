#!/usr/bin/env python3
"""Dump upstream PocketTTS tensors for native Go runtime parity tests.

Run this from the Go repo root after installing the upstream checkout at the
sync target (see docs/INSTALL.md, "Parity setup"):

    original/pockettts/.venv/bin/python scripts/dump_python_parity.py \
      --language german \
      --weights models/german/model.safetensors \
      --tokenizer models/german/tokenizer.json \
      --voice voices/german/juergen.safetensors \
      --output internal/native/testdata/python_parity/german.json
    prettier -w internal/native/testdata/python_parity/german.json

The committed fixtures in internal/native/testdata/python_parity/ run with
`go test ./internal/native`; POCKETTTS_NATIVE_PY_FIXTURE adds one more file.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

# Rows of the text embeddings stored in the fixture; the lookup table is
# trivially right once the token ids match, so the full matrix is not needed.
TEXT_EMBEDDING_ROWS = 3

# Scale of the deterministic latents in the generated-range Mimi cases: values
# span about ±1.65 with std ≈ 1, like normalized flow latents.
GENERATED_LATENT_SCALE = 0.15

# Audio samples kept from the end of the long Mimi case (two 12.5 Hz frames at
# 24 kHz).
MIMI_TAIL_SAMPLES = 3840


def main() -> int:
    args = parse_args()
    upstream = args.upstream.resolve()
    if not (upstream / "pocket_tts").is_dir():
        print(f"upstream checkout not found at {upstream}", file=sys.stderr)
        return 2

    sys.path.insert(0, upstream.as_posix())

    try:
        import torch
        import yaml
        from pocket_tts.default_parameters import get_default_text_for_language
        from pocket_tts.models.model_state import _import_model_state
        from pocket_tts.models.text_chunking import prepare_text_prompt
        from pocket_tts.models.tts_model import TTSModel
        from pocket_tts.modules.stateful_module import increment_steps, init_states
    except ModuleNotFoundError as exc:
        print(
            f"missing Python dependency {exc.name!r}; run `uv sync --no-dev` in {upstream}",
            file=sys.stderr,
        )
        return 2

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    if args.config is not None and (args.weights is None or args.tokenizer is None):
        # The Go test resolves an embedded config's files by language name; a
        # custom config has none, so the fixture must name them.
        print("--config needs --weights and --tokenizer", file=sys.stderr)
        return 2

    if args.config is not None:
        config_path = Path(args.config)
        source_config = args.config
    else:
        config_path = upstream / "pocket_tts" / "config" / f"{args.language}.yaml"
        source_config = args.language

    with tempfile.TemporaryDirectory() as tmp:
        model = TTSModel.load_model(
            config=local_config(yaml, config_path, args.weights, args.tokenizer, Path(tmp))
        )
    model.eval()

    raw_text = args.text if args.text is not None else get_default_text_for_language(args.language)
    prepared_text, _ = prepare_text_prompt(
        raw_text,
        model.pad_with_spaces_for_short_inputs,
        model.remove_semicolons,
        model.append_terminal_punctuation,
        model.capitalize_first_letter,
        model.replace_characters,
    )

    fixture: dict[str, Any] = {
        "source": {
            "upstream": upstream_revision(upstream),
            "config": source_config,
            "seed": args.seed,
        }
    }
    if args.config is not None:
        files = {"weights": args.weights.as_posix(), "tokenizer": args.tokenizer.as_posix()}
        if args.voice is not None:
            files["voices"] = args.voice.parent.as_posix()
        fixture["source"]["files"] = files
    fixture["text"] = dump_text(torch, model, raw_text, prepared_text)
    fixture["flow_lm_prefill_step"] = dump_flow_lm_prefill_step(
        torch,
        init_states,
        increment_steps,
        model,
        parse_ints(args.flow_tokens),
        args.flow_cache_length,
    )
    if args.voice is not None:
        fixture["source"]["voice"] = args.voice.stem
        fixture["voice_prefill_step"] = dump_voice_prefill_step(
            torch,
            _import_model_state,
            init_states,
            increment_steps,
            model,
            args.voice,
            fixture["text"]["tokens"],
        )
    fixture["mimi"] = [
        dump_mimi_case(torch, init_states, model, frames, args.mimi_cache_length)
        for frames in parse_ints(args.mimi_frames)
    ]
    # Generated latents are roughly unit-scale; the small cases above barely
    # excite the decoder transformer's MLP.
    fixture["mimi"].append(
        dump_mimi_case(
            torch,
            init_states,
            model,
            args.mimi_generated_frames,
            args.mimi_cache_length,
            scale=GENERATED_LATENT_SCALE,
            name=f"{args.mimi_generated_frames}_frames_generated_range",
        )
    )
    # Long enough that the decoder transformer runs past its attention
    # context; only the audio tail is stored to keep the fixture small.
    fixture["mimi"].append(
        dump_mimi_case(
            torch,
            init_states,
            model,
            args.mimi_long_frames,
            args.mimi_cache_length,
            scale=GENERATED_LATENT_SCALE,
            name=f"{args.mimi_long_frames}_frames_past_context",
            tail_samples=MIMI_TAIL_SAMPLES,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(args.output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("original/pockettts"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--language", default="english_2026-01")
    source.add_argument(
        "--config",
        help="custom upstream YAML config instead of --language (needs --weights and --tokenizer)",
    )
    parser.add_argument(
        "--weights", type=Path, help="local checkpoint instead of the config's pinned one"
    )
    parser.add_argument(
        "--tokenizer", type=Path, help="local tokenizer.json instead of the config's pinned one"
    )
    parser.add_argument(
        "--voice", type=Path, help="model-state voice (.safetensors) for voice_prefill_step"
    )
    parser.add_argument("--text", help="text for the text and voice cases (default: demo text)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--flow-tokens", default="10,20,30")
    parser.add_argument("--flow-cache-length", type=int, default=64)
    parser.add_argument("--mimi-frames", default="1,2,4")
    parser.add_argument("--mimi-cache-length", type=int, default=64)
    parser.add_argument(
        "--mimi-generated-frames",
        type=int,
        default=4,
        help="frames of the Mimi case with latents in the generated range",
    )
    parser.add_argument(
        "--mimi-long-frames",
        type=int,
        default=130,
        help="frames of the Mimi case that runs past the decoder's attention context",
    )
    return parser.parse_args()


def local_config(
    yaml: Any, config_path: Path, weights: Path | None, tokenizer: Path | None, tmp: Path
) -> Path:
    """Returns config_path, or a copy pointing at local weights and tokenizer, so
    Python and Go read the same files."""
    if weights is None and tokenizer is None:
        return config_path

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if weights is not None:
        config["weights_path"] = weights.resolve().as_posix()
        config.pop("weights_path_without_voice_cloning", None)
    if tokenizer is not None:
        config["flow_lm"]["lookup_table"]["tokenizer_path"] = tokenizer.resolve().as_posix()

    path = tmp / config_path.name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def upstream_revision(upstream: Path) -> str:
    head = upstream / ".git" / "HEAD"
    try:
        return head.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def parse_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def dump_text(torch: Any, model: Any, raw_text: str, prepared_text: str) -> dict[str, Any]:
    flow = model.flow_lm
    tokens = flow.conditioner.prepare(prepared_text)
    with torch.no_grad():
        embeddings = flow.conditioner(tokens)

    return {
        "raw": raw_text,
        "prepared": prepared_text,
        "tokens": [int(t) for t in tokens.reshape(-1).tolist()],
        "embeddings_head": tensor_to_json(embeddings[:, :TEXT_EMBEDDING_ROWS]),
    }


def dump_flow_lm_prefill_step(
    torch: Any,
    init_states: Any,
    increment_steps: Any,
    model: Any,
    tokens: list[int],
    cache_length: int,
) -> dict[str, Any]:
    flow = model.flow_lm
    text_tokens = torch.tensor([tokens], dtype=torch.int64, device=flow.device)
    state = init_states(flow, batch_size=1, sequence_length=cache_length)

    with torch.no_grad():
        text_embeddings = flow.conditioner(text_tokens)
        _ = flow.transformer(text_embeddings, state)
        increment_steps(flow, state, increment=text_embeddings.shape[1])
        prompt_offsets = state_offsets(state)

        step_latent = deterministic_tensor(torch, (1, 1, flow.ldim), scale=0.05)
        last_hidden, eos_logits = flow_step(torch, flow, state, step_latent)
        increment_steps(flow, state, increment=1)
        step_offsets = state_offsets(state)

    return {
        "tokens": tokens,
        "step_latent": tensor_to_json(step_latent),
        "prompt_layer_offsets": prompt_offsets,
        "step_layer_offsets": step_offsets,
        "step_last_hidden": tensor_to_json(last_hidden),
        "step_eos_logits": tensor_to_json(eos_logits),
    }


def dump_voice_prefill_step(
    torch: Any,
    import_model_state: Any,
    init_states: Any,
    increment_steps: Any,
    model: Any,
    voice: Path,
    tokens: list[int],
) -> dict[str, Any]:
    """Mirrors TTSModel._generate up to the first frame: get the voice state,
    grow its KV cache, prompt the text, then run one backbone step (with a fixed
    latent instead of the sampled BOS frame's successor).

    A model-state voice is imported as upstream does. A legacy audio_prompt
    voice (the flat english_2026-01 voices, from upstream 2.x) is the encoded
    prompt of get_state_for_audio_prompt, so it is prompted the way that
    function does after encoding."""
    flow = model.flow_lm
    extra = len(tokens) + 8
    prompt = load_audio_prompt(torch, voice)
    if prompt is None:
        voice_format = "model_state"
        state = import_model_state(voice, flow.device)
        voice_offsets = state_offsets(state)
        model._expand_kv_cache(state, sequence_length=voice_offsets[0] + extra)
    else:
        voice_format = "audio_prompt"
        if flow.insert_bos_before_voice:
            prompt = torch.cat([flow.bos_before_voice, prompt], dim=1)
        state = init_states(flow, batch_size=1, sequence_length=prompt.shape[1] + extra)
        with torch.no_grad():
            _ = flow.transformer(prompt, state)
        increment_steps(flow, state, increment=prompt.shape[1])
        voice_offsets = state_offsets(state)

    text_tokens = torch.tensor([tokens], dtype=torch.int64, device=flow.device)
    with torch.no_grad():
        text_embeddings = flow.conditioner(text_tokens)
        _ = flow.transformer(text_embeddings, state)
        increment_steps(flow, state, increment=len(tokens))
        prompt_offsets = state_offsets(state)

        step_latent = deterministic_tensor(torch, (1, 1, flow.ldim), scale=0.05)
        last_hidden, eos_logits = flow_step(torch, flow, state, step_latent)
        increment_steps(flow, state, increment=1)
        step_offsets = state_offsets(state)

    return {
        "voice_format": voice_format,
        "voice_layer_offsets": voice_offsets,
        "prompt_layer_offsets": prompt_offsets,
        "step_layer_offsets": step_offsets,
        "step_latent": tensor_to_json(step_latent),
        "step_last_hidden": tensor_to_json(last_hidden),
        "step_eos_logits": tensor_to_json(eos_logits),
    }


def load_audio_prompt(torch: Any, voice: Path) -> Any:
    """Returns the audio_prompt tensor of a legacy voice file, or None for a
    model-state voice."""
    import safetensors

    with safetensors.safe_open(voice, framework="pt") as f:
        if list(f.keys()) != ["audio_prompt"]:
            return None
        return f.get_tensor("audio_prompt").to(torch.float32)


def flow_step(torch: Any, flow: Any, state: Any, step_latent: Any) -> tuple[Any, Any]:
    """FlowLM.forward up to the EOS logits, without sampling a latent."""
    sequence = torch.where(torch.isnan(step_latent), flow.bos_emb, step_latent)
    empty_text = torch.empty((1, 0, flow.dim), dtype=flow.dtype, device=flow.device)
    out = flow.backbone(flow.input_linear(sequence), empty_text, sequence, model_state=state)
    last_hidden = out.to(torch.float32)[:, -1]
    return last_hidden, flow.out_eos(last_hidden)


def dump_mimi_case(
    torch: Any,
    init_states: Any,
    model: Any,
    frames: int,
    cache_length: int,
    scale: float = 0.03,
    name: str | None = None,
    tail_samples: int | None = None,
) -> dict[str, Any]:
    flow = model.flow_lm
    mimi = model.mimi
    latent = deterministic_tensor(torch, (1, frames, flow.ldim), scale=scale)
    with torch.no_grad():
        mimi_input = latent * flow.emb_std + flow.emb_mean
        quantized = mimi.quantizer(mimi_input.transpose(-1, -2))

        mimi_steps_per_latent = int(mimi.encoder_frame_rate / mimi.frame_rate)
        sequence_length = max(cache_length, frames * mimi_steps_per_latent)
        mimi_state = init_states(mimi, batch_size=1, sequence_length=sequence_length)
        # decode_from_latent applies the quantizer itself (upstream 3.x).
        audio = mimi.decode_from_latent(mimi_input, mimi_state)

    case = {
        "name": name if name is not None else f"{frames}_frames",
        "latent": tensor_to_json(latent),
    }
    if tail_samples is None:
        case["latent_to_mimi"] = tensor_to_json(quantized)
        case["mimi_decode"] = tensor_to_json(audio)
    else:
        # The short cases already cover latent_to_mimi; the full long tensors
        # would dominate the fixture size.
        case["mimi_decode_tail"] = tensor_to_json(audio[..., -tail_samples:])
    return case


def deterministic_tensor(torch: Any, shape: tuple[int, ...], scale: float) -> Any:
    count = 1
    for dim in shape:
        count *= dim
    values = torch.arange(count, dtype=torch.float32)
    values = ((values % 23) - 11) * scale
    return values.reshape(shape)


def state_offsets(state: dict[str, dict[str, Any]]) -> list[int]:
    offsets: list[tuple[str, int]] = []
    for name, module_state in state.items():
        offset = module_state.get("offset")
        if offset is not None:
            offsets.append((name, int(offset.reshape(-1)[0].item())))
    # Sort by layer index, not by name: "layers.10" sorts before "layers.2".
    offsets.sort(key=lambda item: natural_key(item[0]))
    return [offset for _, offset in offsets]


def natural_key(name: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in name.replace(".", " . ").split()]


def tensor_to_json(tensor: Any) -> dict[str, Any]:
    cpu = tensor.detach().float().cpu().contiguous()
    # 9 significant digits round-trip every float32 exactly.
    return {
        "shape": list(cpu.shape),
        "data": [float(f"{x:.9g}") for x in cpu.reshape(-1).tolist()],
    }


if __name__ == "__main__":
    raise SystemExit(main())
