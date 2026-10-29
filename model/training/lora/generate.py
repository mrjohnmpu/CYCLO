from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.shared.pipeline_utils import (
    is_valid_smiles,
    load_peft_model,
    load_tokenizer,
    parse_args_with_config,
    seed_everything,
)
from model.shared.generation_utils import clean_generated_sequence
from model.training.lora.evaluate import PeptideEvaluator


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
    parser = argparse.ArgumentParser(description="Generate sequences from a LoRA adapter with conditional type tokens.")
    parser.add_argument("--model_path", type=str, default="output/best_model_with_mask_trainer/checkpoint-177198")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--condition_type", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='runs/lora_type_adapters_all/generated')
    parser.add_argument("--output_filename", type=str, default="generated.csv")
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=576)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--do_evaluate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:1")

    # Backward-compatible aliases.
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--adapter_base_dir", type=str, default=None)
    parser.add_argument("--peptide_type", type=str, default=None)
    parser.add_argument("--num_sequences", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_args_with_config(parser)

    if args.base_model_path and not args.model_path:
        args.model_path = args.base_model_path
    if args.peptide_type and not args.condition_type:
        args.condition_type = args.peptide_type
    if args.num_sequences is not None:
        args.num_samples = args.num_sequences
    if args.output_path:
        output_path = Path(args.output_path)
        args.output_dir = str(output_path.parent)
        args.output_filename = output_path.name
    if args.adapter_base_dir and args.condition_type and not args.adapter_path:
        args.adapter_path = str(Path(args.adapter_base_dir) / f"adapter_{args.condition_type}")
    if args.tokenizer_path is None and args.adapter_base_dir:
        args.tokenizer_path = str(Path(args.adapter_base_dir) / "tokenizer")

    missing = [
        name
        for name in ("model_path", "tokenizer_path", "adapter_path", "condition_type")
        if not getattr(args, name)
    ]
    if missing:
        parser.error(
            "Missing required arguments after config/alias resolution: "
            + ", ".join(f"--{name}" for name in missing)
        )

    if args.num_samples < 1:
        parser.error("--num_samples must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.max_length < 2:
        parser.error("--max_length must be >= 2")

    args.model_path = resolve_repo_path(args.model_path)
    args.tokenizer_path = resolve_repo_path(args.tokenizer_path, allow_model_hub=True)
    args.adapter_path = resolve_repo_path(args.adapter_path)
    args.output_dir = resolve_repo_path(args.output_dir)
    return args


def generate_conditioned_batch(
    model,
    tokenizer,
    *,
    condition_type: str,
    batch_size: int,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device: str,
):
    condition_token = f"<{condition_type}>"
    token_id = tokenizer.convert_tokens_to_ids(condition_token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Condition token {condition_token!r} not found in tokenizer vocabulary.")

    input_ids = torch.full(
        (batch_size, 1),
        token_id,
        dtype=torch.long,
        device=device,
    )

    generation_kwargs = {
        "max_length": max_length,
        "do_sample": True,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, **generation_kwargs)
    finally:
        if was_training:
            model.train()

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def build_rows(decoded_sequences, *, condition_type: str, model_path: str, adapter_path: str, tokenizer):
    rows = []
    valid_count = 0
    condition_token = f"<{condition_type}>"
    for seq in decoded_sequences:
        smiles = clean_generated_sequence(seq, condition_token=condition_token, tokenizer=tokenizer)
        is_valid = is_valid_smiles(smiles)
        valid_count += int(is_valid)
        rows.append(
            {
                "SMILES": smiles,
                "Valid": is_valid,
                "model_path": model_path,
                "adapter_path": adapter_path,
                "condition_type": condition_type,
            }
        )
    return rows, valid_count


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

    tokenizer = load_tokenizer(args.tokenizer_path, model_max_length=args.max_length)
    model = load_peft_model(
        args.model_path,
        args.adapter_path,
        tokenizer,
        device=args.device,
        trainable=False,
    )
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
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "adapter_path": args.adapter_path,
        "condition_type": args.condition_type,
        "output_path": str(output_path),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "requested_max_length": args.max_length,
        "effective_max_length": effective_max_length,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "device": args.device,
        "csv_columns": ["SMILES", "Valid", "model_path", "adapter_path", "condition_type"],
        "cleaning_rule": "strip lowercase prefix before the first uppercase token",
    }
    write_json(metadata_path, metadata)

    total_generated = 0
    valid_count = 0
    write_header = True
    pbar = tqdm(total=args.num_samples, desc=f"Generating {args.condition_type}")
    while total_generated < args.num_samples:
        current_batch_size = min(args.batch_size, args.num_samples - total_generated)
        decoded_sequences = generate_conditioned_batch(
            model,
            tokenizer,
            condition_type=args.condition_type,
            batch_size=current_batch_size,
            max_length=effective_max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
        )
        rows, batch_valid_count = build_rows(
            decoded_sequences,
            condition_type=args.condition_type,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            tokenizer=tokenizer,
        )
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
            },
        )
        pbar.update(len(rows))
    pbar.close()

    print(f"Saved {total_generated} samples to {output_path}")
    print(f"Valid SMILES: {valid_count} ({valid_count / total_generated * 100:.2f}%)")

    if args.do_evaluate:
        df = pd.read_csv(output_path)
        evaluator = PeptideEvaluator()
        metrics = evaluator.evaluate(df["SMILES"].astype(str).tolist(), args.condition_type)
        metrics_path = output_path.with_suffix(".metrics.json")
        write_json(metrics_path, metrics)
        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
