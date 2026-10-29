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

from model.shared.pipeline_utils import (
    is_valid_smiles,
    load_peft_model,
    load_tokenizer,
    parse_args_with_config,
    seed_everything,
)
from model.training.lora.dataset import TYPE_PREFIX_TOKENS, canonicalize_peptide_types


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate from topology-specific LoRA adapters without condition tokens."
    )
    parser.add_argument("--base_model_path", type=str, required=False)
    parser.add_argument("--trained_tokenizer_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--generation_output_dir", type=str, required=False)
    parser.add_argument("--peptide_types", nargs="*", default=None)
    parser.add_argument("--expected_vocab_size", type=int, default=2140)
    parser.add_argument("--expected_bos_token_id", type=int, default=1)
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
            "output_dir",
            "generation_output_dir",
        )
        if not getattr(args, key, None)
    ]
    if missing:
        raise ValueError("Missing required settings: " + ", ".join(missing))
    args.peptide_types = canonicalize_peptide_types(args.peptide_types)
    return args


def audit_tokenizer(tokenizer, args) -> None:
    if len(tokenizer) != args.expected_vocab_size:
        raise RuntimeError(
            f"Tokenizer vocabulary changed: expected {args.expected_vocab_size}, received {len(tokenizer)}."
        )
    if tokenizer.bos_token_id != args.expected_bos_token_id:
        raise RuntimeError(
            f"Unexpected BOS token ID: expected {args.expected_bos_token_id}, "
            f"received {tokenizer.bos_token_id}."
        )
    present = [token for token in TYPE_PREFIX_TOKENS.values() if token in tokenizer.get_vocab()]
    if present:
        raise RuntimeError(f"Topology tokens must not be present: {present}")


def generate_type(model, tokenizer, peptide_type, args, output_path):
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output file already exists: {output_path}")
        output_path.unlink()

    seed_everything(args.generation_seed)
    total = 0
    valid = 0
    write_header = True
    progress = tqdm(total=args.num_samples, desc=f"Generating {peptide_type}")
    while total < args.num_samples:
        current_batch = min(args.generation_batch_size, args.num_samples - total)
        input_ids = torch.full(
            (current_batch, 1),
            tokenizer.bos_token_id,
            dtype=torch.long,
            device=args.device,
        )
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=args.max_length,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        rows = []
        for sequence in decoded:
            smiles = sequence.replace(" ", "").strip()
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


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.generation_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.trained_tokenizer_path, model_max_length=args.max_length)
    audit_tokenizer(tokenizer, args)

    summaries = []
    for peptide_type in args.peptide_types:
        adapter_path = Path(args.output_dir) / f"adapter_{peptide_type}"
        model = load_peft_model(
            args.base_model_path,
            str(adapter_path),
            tokenizer,
            device=args.device,
            trainable=False,
        )
        model.eval()
        summaries.append(
            generate_type(
                model,
                tokenizer,
                peptide_type,
                args,
                output_dir / f"{peptide_type}_n{args.num_samples}.csv",
            )
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            "adapter_base_dir": args.output_dir,
            "peptide_types": args.peptide_types,
            "num_samples_per_type": args.num_samples,
            "generation_seed": args.generation_seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_length": args.max_length,
            "rdkit_version": rdBase.rdkitVersion,
            "reported_metric": "RDKit validity only",
        },
    )


if __name__ == "__main__":
    main()
