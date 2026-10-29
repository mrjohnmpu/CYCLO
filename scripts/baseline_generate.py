#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.moses_baselines import runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate peptides from a trained baseline run.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=runtime.DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=runtime.DEFAULT_SEED)
    parser.add_argument("--num_samples", type=int, default=runtime.DEFAULT_NUM_SAMPLES)
    parser.add_argument("--batch_size", type=int, default=runtime.DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_len", type=int, default=runtime.DEFAULT_MAX_LEN)
    parser.add_argument("--output_name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime.prepare_device(args.device)
    runtime.seed_everything(args.seed)

    run_dir = Path(args.run_dir).resolve()
    run_paths = runtime.ensure_run_dirs(run_dir, ("generated",))
    manifest, _, _, _, model = runtime.load_trained_run(run_dir, args.device)

    model_name = manifest["model"]
    cyclization_type = manifest["cyclization_type"]
    run_name = manifest["run_name"]
    output_name = args.output_name or runtime.default_output_name(model_name, cyclization_type, args.num_samples)
    if output_name.endswith(".csv"):
        output_name = output_name[:-4]

    records = []
    remaining = args.num_samples
    while remaining > 0:
        current_batch = min(args.batch_size, remaining)
        smiles = runtime.sample_batch(model_name, model, current_batch, args.max_len)
        records.extend(
            runtime.build_generation_records(
                smiles,
                model_name=model_name,
                run_name=run_name,
                run_dir=str(run_dir),
                cyclization_type=cyclization_type,
            )
        )
        remaining -= len(smiles)

    dataframe = pd.DataFrame(records, columns=["SMILES", "Valid", "Cyclization", "model_name", "run_name", "run_dir"])
    output_csv = run_paths["generated_dir"] / f"{output_name}.csv"
    metadata_json = output_csv.with_suffix(".metadata.json")
    dataframe.to_csv(output_csv, index=False)

    metadata = {
        "source_run": str(run_dir),
        "model": model_name,
        "run_name": run_name,
        "cyclization_type": cyclization_type,
        "checkpoint_paths": manifest["checkpoint_paths"],
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "seed": args.seed,
        "device": args.device,
        "generated_csv": str(output_csv),
        "num_valid": int(dataframe["Valid"].sum()),
        "validity": float(dataframe["Valid"].mean()) if len(dataframe) else 0.0,
    }
    runtime.save_json(metadata_json, metadata)

    print(f"Saved generated CSV: {output_csv}")
    print(f"Saved generation metadata: {metadata_json}")


if __name__ == "__main__":
    main()
