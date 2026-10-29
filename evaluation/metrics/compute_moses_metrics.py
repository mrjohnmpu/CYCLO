import argparse
import multiprocessing
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from evaluation.metrics.core import evaluate_generated_csv, write_reports


def parse_args():
    parser = argparse.ArgumentParser(description="Compute MOSES metrics for a generated peptide CSV.")
    parser.add_argument("--generated_csv", type=str, required=True, help="Generated CSV file containing a SMILES column.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store metric reports.")
    parser.add_argument("--experiment_name", type=str, default=None, help="Optional experiment name stored in reports.")
    parser.add_argument("--train_csv", type=str, required=True, help="Training CSV file.")
    parser.add_argument("--test_csv", type=str, required=True, help="Reference test CSV file.")
    parser.add_argument("--cyclization_type", type=str, default="head_to_tail", help="Cyclization type used to filter train and test data.")
    parser.add_argument("--smiles_col", type=str, default="SMILES", help="SMILES column name.")
    parser.add_argument("--type_col", type=str, default="Cyclization", help="Cyclization type column name.")
    parser.add_argument("--valid_col", type=str, default="Valid", help="Validity column in generated CSV.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_jobs", type=int, default=max(1, min(8, multiprocessing.cpu_count())))
    parser.add_argument("--report_prefix", type=str, default="metrics", help="Prefix for output report files.")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_row = evaluate_generated_csv(
        generated_csv=args.generated_csv,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        experiment_name=args.experiment_name,
        cyclization_type=args.cyclization_type,
        smiles_col=args.smiles_col,
        type_col=args.type_col,
        valid_col=args.valid_col,
        device=args.device,
        n_jobs=args.n_jobs,
    )
    csv_path, json_path, txt_path = write_reports(args.output_dir, args.report_prefix, metrics_row)
    print(f"Saved CSV report: {csv_path}")
    print(f"Saved JSON report: {json_path}")
    print(f"Saved TXT report: {txt_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
