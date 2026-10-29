import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.shared.pipeline_utils import (
    is_valid_smiles,
    load_tokenizer,
    parse_args_with_config,
    seed_everything,
)


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sequences from a pretrained base GPT2 checkpoint.")
    parser.add_argument(
        "--model_path",
        type=str,
        default='./output/best_model_with_mask_trainer/checkpoint-177198',
        help="Path to the pretrained base GPT2 checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/generated/base_gpt2",
        help="Directory for generated CSV output.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="base_gpt2_generated_30000.csv",
        help="CSV filename for generated results.",
    )
    parser.add_argument("--num_samples", type=int, default=30000, help="Number of sequences to generate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size used during generation.")
    parser.add_argument("--max_length", type=int, default=576, help="Maximum generation length.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Torch device used for generation.")
    args = parse_args_with_config(parser)

    if args.num_samples < 1:
        parser.error("--num_samples must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.max_length < 2:
        parser.error("--max_length must be >= 2")

    args.model_path = resolve_repo_path(args.model_path)
    args.output_dir = resolve_repo_path(args.output_dir)
    return args


def clean_smiles(text: str) -> str:
    return text.replace(" ", "").strip()


def generate_unconditional_outputs(
    model,
    tokenizer,
    *,
    batch_size: int,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str,
):
    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer must define bos_token_id for unconditional generation.")

    input_ids = torch.full(
        (batch_size, 1),
        tokenizer.bos_token_id,
        dtype=torch.long,
        device=device,
    )

    generation_kwargs = {
        "max_length": max_length,
        "do_sample": True,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
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


def build_generation_records(decoded_sequences, *, model_path: str):
    records = []
    for seq in decoded_sequences:
        smiles = clean_smiles(seq)
        records.append(
            {
                "SMILES": smiles,
                "Valid": is_valid_smiles(smiles),
                "model_path": model_path,
            }
        )
    return records


def append_records_to_csv(records, output_path: Path, *, write_header: bool) -> int:
    df = pd.DataFrame(records)
    df.to_csv(output_path, mode="a", header=write_header, index=False)
    return int(df["Valid"].sum())


def write_run_metadata(metadata_path: Path, payload: dict) -> None:
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = load_tokenizer(args.model_path, model_max_length=args.max_length)
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(args.device)
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
        "csv_columns": ["SMILES", "Valid", "model_path"],
    }
    write_run_metadata(metadata_path, metadata)

    total_generated = 0
    valid_count = 0
    write_header = True
    pbar = tqdm(total=args.num_samples, desc="Generating")
    while total_generated < args.num_samples:
        current_batch_size = min(args.batch_size, args.num_samples - total_generated)
        decoded_sequences = generate_unconditional_outputs(
            model,
            tokenizer,
            batch_size=current_batch_size,
            max_length=effective_max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )
        records = build_generation_records(decoded_sequences, model_path=args.model_path)
        valid_count += append_records_to_csv(records, output_path, write_header=write_header)
        write_header = False
        total_generated += len(records)
        progress_path.write_text(
            json.dumps(
                {
                    "generated": total_generated,
                    "num_samples": args.num_samples,
                    "valid_count": valid_count,
                    "valid_rate": (valid_count / total_generated) if total_generated else 0.0,
                    "output_path": str(output_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        pbar.update(len(records))
    pbar.close()

    print(f"Saved {total_generated} samples to {output_path}")
    print(f"Valid SMILES: {valid_count} ({valid_count / total_generated * 100:.2f}%)")


if __name__ == "__main__":
    main()
