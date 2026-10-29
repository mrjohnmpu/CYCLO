#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.moses_baselines import runtime
from evaluation.metrics.core import evaluate_generated_csv, write_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one generated baseline CSV.")
    parser.add_argument("--generated_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--reference_csv", type=Path, required=True)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--cyclization_type", type=str, default=None)
    parser.add_argument("--smiles_col", type=str, default="SMILES")
    parser.add_argument("--type_col", type=str, default="Cyclization")
    parser.add_argument("--valid_col", type=str, default="Valid")
    parser.add_argument("--device", type=str, default=runtime.DEFAULT_DEVICE)
    parser.add_argument("--n_jobs", type=int, default=max(1, min(8, multiprocessing.cpu_count())))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_row = evaluate_generated_csv(
        generated_csv=args.generated_csv,
        train_csv=args.train_csv,
        test_csv=args.reference_csv,
        experiment_name=args.experiment_name,
        cyclization_type=args.cyclization_type,
        smiles_col=args.smiles_col,
        type_col=args.type_col,
        valid_col=args.valid_col,
        device=args.device,
        n_jobs=args.n_jobs,
    )
    csv_path, json_path, txt_path = write_reports(args.output_dir, "metrics", metrics_row)
    print(f"Saved CSV report: {csv_path}")
    print(f"Saved JSON report: {json_path}")
    print(f"Saved TXT report: {txt_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
