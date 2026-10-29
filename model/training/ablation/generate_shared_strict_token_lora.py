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
from model.shared.pipeline_utils import build_condition_token, is_valid_smiles, load_peft_model, load_tokenizer, parse_args_with_config, seed_everything
from model.training.lora.dataset import canonicalize_peptide_types


def parser():
    value = argparse.ArgumentParser(description="Generate from one shared strict-token LoRA adapter.")
    value.add_argument("--base_model_path", required=False)
    value.add_argument("--trained_tokenizer_path", required=False)
    value.add_argument("--output_dir", required=False)
    value.add_argument("--generation_output_dir", required=False)
    value.add_argument("--peptide_types", nargs="*", default=None)
    value.add_argument("--expected_condition_token_ids", default=None)
    value.add_argument("--num_samples", type=int, default=30000)
    value.add_argument("--generation_batch_size", type=int, default=64)
    value.add_argument("--max_length", type=int, default=1024)
    value.add_argument("--temperature", type=float, default=0.8)
    value.add_argument("--top_k", type=int, default=50)
    value.add_argument("--top_p", type=float, default=0.95)
    value.add_argument("--repetition_penalty", type=float, default=1.0)
    value.add_argument("--generation_seed", type=int, default=42)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--overwrite", action="store_true")
    return value


def generate(model, tokenizer, peptide_type, token_id, args, output_path):
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}")
        output_path.unlink()
    seed_everything(args.generation_seed)
    condition_token = build_condition_token(peptide_type)
    total = valid = 0
    write_header = True
    progress = tqdm(total=args.num_samples, desc=f"Generating {peptide_type}")
    while total < args.num_samples:
        batch_size = min(args.generation_batch_size, args.num_samples - total)
        input_ids = torch.full((batch_size, 1), token_id, dtype=torch.long, device=args.device)
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
        rows = []
        for sequence in tokenizer.batch_decode(output_ids, skip_special_tokens=False):
            smiles = clean_generated_sequence(sequence, condition_token=condition_token, tokenizer=tokenizer)
            value = is_valid_smiles(smiles)
            rows.append({"SMILES": smiles, "Valid": value})
            valid += int(value)
        pd.DataFrame(rows, columns=["SMILES", "Valid"]).to_csv(
            output_path, mode="a", header=write_header, index=False
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
        "rdkit_validity": valid / total,
    }


def main():
    args = parse_args_with_config(parser())
    args.peptide_types = canonicalize_peptide_types(args.peptide_types)
    output_dir = Path(args.generation_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.trained_tokenizer_path, model_max_length=args.max_length)
    model = load_peft_model(
        args.base_model_path,
        str(Path(args.output_dir) / "adapter_shared"),
        tokenizer,
        device=args.device,
        trainable=False,
    )
    model.eval()
    summaries = []
    for peptide_type in args.peptide_types:
        token_id = tokenizer.convert_tokens_to_ids(build_condition_token(peptide_type))
        expected_id = int(args.expected_condition_token_ids[peptide_type])
        if token_id != expected_id:
            raise RuntimeError(f"Token ID mismatch for {peptide_type}: {token_id} != {expected_id}")
        summaries.append(
            generate(
                model, tokenizer, peptide_type, token_id, args,
                output_dir / f"{peptide_type}_n{args.num_samples}.csv",
            )
        )
    pd.DataFrame(summaries).to_csv(output_dir / "rdkit_validity_summary.csv", index=False)
    (output_dir / "generation_metadata.json").write_text(
        json.dumps(
            {
                "adapter_path": str(Path(args.output_dir) / "adapter_shared"),
                "tokenizer_path": args.trained_tokenizer_path,
                "condition_token_ids": args.expected_condition_token_ids,
                "num_samples_per_type": args.num_samples,
                "generation_seed": args.generation_seed,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "max_length": args.max_length,
                "rdkit_version": rdBase.rdkitVersion,
            }, indent=2, ensure_ascii=True
        ) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
