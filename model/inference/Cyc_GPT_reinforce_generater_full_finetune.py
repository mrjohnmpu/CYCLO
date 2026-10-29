from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.shared.generation_utils import (  # noqa: E402
    build_generation_records,
    generate_conditioned_outputs,
)
from model.shared.pipeline_utils import (  # noqa: E402
    FULL_FINETUNE_STATE_FILENAME,
    load_tokenizer,
    parse_args_with_config,
    seed_everything,
)


def resolve_repo_path(path_str: str | None, *, allow_model_hub: bool = False) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    project_candidate = PROJECT_ROOT / path
    model_candidate = PROJECT_ROOT / "model" / path

    if str(path_str).startswith("../"):
        return str(model_candidate.resolve())
    if str(path_str).startswith(("./", "data/", "output/", "configs/", "evaluation/", "model/", "scripts/")):
        return str(project_candidate.resolve())
    if project_candidate.exists():
        return str(project_candidate.resolve())
    if allow_model_hub:
        return path_str
    return str(project_candidate.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate peptides from a full-finetune RL checkpoint.")
    parser.add_argument("--base_model_path", type=str, default="output/best_model_with_mask_trainer/checkpoint-177198")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Path to the full-finetune checkpoint directory.")
    parser.add_argument("--checkpoint_mode", type=str, default="full_finetune", choices=["auto", "full_finetune"])
    parser.add_argument("--condition_type", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='runs/full_finetune_generated')
    parser.add_argument("--output_filename", type=str, default="generated.csv")
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_tag", type=str, default=None)
    parser.add_argument("--objective_tag", type=str, default=None)
    parser.add_argument("--scorer_tag", type=str, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_args_with_config(parser)

    missing = [name for name in ("base_model_path", "tokenizer_path", "lora_adapter_path", "condition_type") if not getattr(args, name)]
    if missing:
        parser.error(
            "Missing required arguments after config resolution: " + ", ".join(f"--{name}" for name in missing)
        )
    if args.num_samples < 1:
        parser.error("--num_samples must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.max_length < 2:
        parser.error("--max_length must be >= 2")

    args.base_model_path = resolve_repo_path(args.base_model_path)
    args.tokenizer_path = resolve_repo_path(args.tokenizer_path, allow_model_hub=True)
    args.lora_adapter_path = resolve_repo_path(args.lora_adapter_path)
    args.output_dir = resolve_repo_path(args.output_dir)
    return args


def load_full_finetune_model(args, tokenizer):
    checkpoint_dir = Path(args.lora_adapter_path)
    print(f"Loading Tokenizer from: {args.tokenizer_path}")
    print(f"Loading Base Model from: {args.base_model_path}")
    print(f"Loading Checkpoint from: {args.lora_adapter_path}")
    print("Checkpoint mode: full_finetune")

    if (checkpoint_dir / "config.json").exists() and ((checkpoint_dir / "model.safetensors").exists() or (checkpoint_dir / "pytorch_model.bin").exists()):
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir))
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model_path)
        model.resize_token_embeddings(len(tokenizer))
        state_path = checkpoint_dir / FULL_FINETUNE_STATE_FILENAME
        if not state_path.exists():
            raise FileNotFoundError(f"Full-finetune state file not found: {state_path}")
        state_dict = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def append_rows(rows: list[dict[str, object]], output_path: Path, *, write_header: bool) -> int:
    df = pd.DataFrame(rows)
    df.to_csv(output_path, mode="a", header=write_header, index=False)
    return int(df["Valid"].sum())


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_source = args.lora_adapter_path if os.path.exists(os.path.join(args.lora_adapter_path, "tokenizer.json")) else args.tokenizer_path
    tokenizer = load_tokenizer(tokenizer_source, model_max_length=args.max_length)
    model = load_full_finetune_model(args, tokenizer)

    output_path = Path(args.output_dir) / args.output_filename
    metadata_path = output_path.with_suffix(".metadata.json")
    progress_path = output_path.with_suffix(".progress.json")

    max_supported = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
    effective_max_length = args.max_length
    if max_supported is not None and effective_max_length > max_supported:
        print(f"Requested max_length {effective_max_length} exceeds model limit {max_supported}; clipping to {max_supported}.")
        effective_max_length = int(max_supported)

    if output_path.exists():
        output_path.unlink()

    metadata = {
        "base_model_path": args.base_model_path,
        "tokenizer_path": tokenizer_source,
        "checkpoint_path": args.lora_adapter_path,
        "checkpoint_mode": "full_finetune",
        "condition_type": args.condition_type,
        "output_path": str(output_path),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "requested_max_length": args.max_length,
        "effective_max_length": effective_max_length,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "device": args.device,
        "csv_columns": ["SMILES", "Valid", "Cyclization", "CheckpointTag", "ObjectiveTag", "ScorerTag"],
    }
    write_json(metadata_path, metadata)

    total_generated = 0
    valid_count = 0
    write_header = True
    pbar = tqdm(total=args.num_samples, desc=f"Generating {args.condition_type}")

    while total_generated < args.num_samples:
        current_batch_size = min(args.batch_size, args.num_samples - total_generated)
        _, _, decoded_sequences = generate_conditioned_outputs(
            model,
            tokenizer,
            args.condition_type,
            batch_size=current_batch_size,
            max_length=effective_max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            skip_special_tokens=True,
        )
        rows, _ = build_generation_records(
            decoded_sequences,
            condition_type=args.condition_type,
            tokenizer=tokenizer,
        )
        if args.checkpoint_tag:
            for row in rows:
                row["CheckpointTag"] = args.checkpoint_tag
        if args.objective_tag:
            for row in rows:
                row["ObjectiveTag"] = args.objective_tag
        if args.scorer_tag:
            for row in rows:
                row["ScorerTag"] = args.scorer_tag

        valid_count += append_rows(rows, output_path, write_header=write_header)
        write_header = False
        total_generated += len(rows)

        write_json(
            progress_path,
            {
                "generated": total_generated,
                "num_samples": args.num_samples,
                "valid_count": valid_count,
                "valid_rate": (valid_count / total_generated) if total_generated else 0.0,
                "output_path": str(output_path),
                "condition_type": args.condition_type,
                "checkpoint_path": args.lora_adapter_path,
            },
        )
        pbar.update(len(rows))

    pbar.close()
    print(f"Saved {total_generated} samples to {output_path}")
    print(f"Valid SMILES: {valid_count} ({valid_count / total_generated * 100:.2f}%)")


if __name__ == "__main__":
    main()
