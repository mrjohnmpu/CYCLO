from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from rdkit import rdBase
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.shared.generation_utils import clean_generated_sequence
from model.shared.pipeline_utils import (
    build_condition_token,
    configure_condition_token,
    is_valid_smiles,
    load_peft_model,
    load_tokenizer,
    parse_args_with_config,
    seed_everything,
)
from model.training.lora.dataset import canonicalize_peptide_types


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate token-only ablation samples and report RDKit validity only."
    )
    parser.add_argument("--base_model_path", type=str, required=False)
    parser.add_argument("--trained_tokenizer_path", type=str, required=False)
    parser.add_argument("--adapter_path", type=str, required=False)
    parser.add_argument("--generation_output_dir", type=str, required=False)
    parser.add_argument("--peptide_types", nargs="*", default=None)
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--generation_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    args = parse_args_with_config(build_parser())
    missing = [
        key
        for key in (
            "base_model_path",
            "trained_tokenizer_path",
            "adapter_path",
            "generation_output_dir",
        )
        if not getattr(args, key, None)
    ]
    if missing:
        raise ValueError("Missing required settings: " + ", ".join(missing))
    args.peptide_types = canonicalize_peptide_types(args.peptide_types)
    return args


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def generate_type(model, tokenizer, peptide_type: str, args, output_path: Path) -> dict:
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output file already exists: {output_path}")
        output_path.unlink()

    seed_everything(args.generation_seed)
    condition_token = build_condition_token(peptide_type)
    start_token_id = configure_condition_token(tokenizer, peptide_type)
    model_limit = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", None
    )
    effective_max_length = min(args.max_length, int(model_limit or args.max_length))

    total = 0
    valid = 0
    write_header = True
    progress = tqdm(total=args.num_samples, desc=f"Generating {peptide_type}")
    while total < args.num_samples:
        current_batch = min(args.generation_batch_size, args.num_samples - total)
        input_ids = torch.full(
            (current_batch, 1),
            start_token_id,
            dtype=torch.long,
            device=args.device,
        )
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=effective_max_length,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        rows = []
        for sequence in decoded:
            smiles = clean_generated_sequence(
                sequence,
                condition_token=condition_token,
                tokenizer=tokenizer,
            )
            is_valid = is_valid_smiles(smiles)
            rows.append({"SMILES": smiles, "Valid": is_valid})
            valid += int(is_valid)
        pd.DataFrame(rows, columns=["SMILES", "Valid"]).to_csv(
            output_path,
            mode="a",
            header=write_header,
            index=False,
        )
        write_header = False
        total += len(rows)
        progress.update(len(rows))
    progress.close()
    return {
        "condition_type": peptide_type,
        "total_generated": total,
        "rdkit_valid": valid,
        "rdkit_invalid": total - valid,
        "rdkit_validity": valid / total if total else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.generation_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.trained_tokenizer_path, model_max_length=args.max_length)
    model = load_peft_model(
        args.base_model_path,
        args.adapter_path,
        tokenizer,
        device=args.device,
        trainable=False,
    )
    model.eval()

    summaries = []
    for peptide_type in args.peptide_types:
        summaries.append(
            generate_type(
                model,
                tokenizer,
                peptide_type,
                args,
                output_dir / f"{peptide_type}_n{args.num_samples}.csv",
            )
        )
    pd.DataFrame(
        summaries,
        columns=[
            "condition_type",
            "total_generated",
            "rdkit_valid",
            "rdkit_invalid",
            "rdkit_validity",
        ],
    ).to_csv(output_dir / "rdkit_validity_summary.csv", index=False)
    write_json(
        output_dir / "generation_metadata.json",
        {
            "base_model_path": args.base_model_path,
            "tokenizer_path": args.trained_tokenizer_path,
            "adapter_path": args.adapter_path,
            "peptide_types": args.peptide_types,
            "num_samples_per_type": args.num_samples,
            "batch_size": args.generation_batch_size,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "generation_seed": args.generation_seed,
            "rdkit_version": rdBase.rdkitVersion,
            "reported_metric": "RDKit validity only",
        },
    )


if __name__ == "__main__":
    main()
