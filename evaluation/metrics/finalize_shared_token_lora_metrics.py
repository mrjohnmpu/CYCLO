from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger, rdBase

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_COLUMNS = [
    "Method", "Cyclization", "IntDiv", "IntDiv2", "Novelty", "SNN",
    "Uniq@10k", "Uniq@20k", "Uniq@30k", "Validity", "ValidCount",
    "TotalCount", "GeneratedCSV",
]


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("validate", "summarize"), required=True)
    return parser.parse_args()


def load(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value):
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def load_config(path):
    config = load(resolve_path(path))
    for name in ("existing_fifteen_method_summary", "combined_output_dir"):
        config[name] = resolve_path(config[name])
    for entry in config["shared_entries"]:
        for name in ("generated_csv", "metrics_json"):
            entry[name] = resolve_path(entry[name])
    return config


def bool_value(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "t", "yes"}:
        return True
    if text in {"false", "0", "f", "no"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def validate(config):
    RDLogger.DisableLog("rdApp.*")
    rows = []
    for entry in config["shared_entries"]:
        path = Path(entry["generated_csv"])
        dataframe = pd.read_csv(path)
        if list(dataframe.columns) != ["SMILES", "Valid"] or len(dataframe) != config["expected_num_samples"]:
            raise RuntimeError(f"Unexpected generated CSV schema or row count: {path}")
        recorded = dataframe["Valid"].map(bool_value).tolist()
        recomputed = dataframe["SMILES"].fillna("").map(
            lambda value: bool(value) and Chem.MolFromSmiles(str(value).strip()) is not None
        ).tolist()
        mismatches = sum(left != right for left, right in zip(recorded, recomputed))
        if mismatches:
            raise RuntimeError(f"RDKit validity mismatch in {path}: {mismatches}")
        rows.append(
            {
                "Method": "Shared token+LoRA", "Cyclization": entry["cyclization"],
                "TotalCount": len(dataframe), "ValidCount": int(sum(recomputed)),
                "InvalidCount": len(dataframe) - int(sum(recomputed)),
                "Validity": sum(recomputed) / len(dataframe), "GeneratedCSV": str(path),
            }
        )
    root = Path(config["combined_output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "shared_generation_validation.csv", index=False)


def metric(metrics, key):
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite metric {key}: {value}")
    return value


def summarize(config):
    existing = pd.read_csv(config["existing_fifteen_method_summary"])
    if list(existing.columns) != SUMMARY_COLUMNS or len(existing) != 15:
        raise RuntimeError("Existing 15-row summary has an unexpected schema or row count.")
    additions = []
    for entry in config["shared_entries"]:
        metrics = load(entry["metrics_json"])
        row = {
            "Method": "Shared token+LoRA", "Cyclization": entry["cyclization"],
            "IntDiv": metric(metrics, "IntDiv"), "IntDiv2": metric(metrics, "IntDiv2"),
            "Novelty": metric(metrics, "Novelty"), "SNN": metric(metrics, "SNN/Test"),
            "Uniq@10k": metric(metrics, "unique@10000"),
            "Uniq@20k": metric(metrics, "unique@20000"),
            "Uniq@30k": metric(metrics, "unique@30000"),
            "Validity": metric(metrics, "validity"), "ValidCount": int(metrics["num_valid"]),
            "TotalCount": int(metrics["num_total"]), "GeneratedCSV": entry["generated_csv"],
        }
        for key in ("Novelty", "SNN", "Uniq@10k", "Uniq@20k", "Uniq@30k", "Validity"):
            if not 0 <= row[key] <= 1:
                raise ValueError(f"Metric {key} is outside [0,1]: {row[key]}")
        additions.append(row)
    combined = pd.concat([existing, pd.DataFrame(additions, columns=SUMMARY_COLUMNS)], ignore_index=True)
    if len(combined) != 18:
        raise RuntimeError(f"Expected 18 rows, received {len(combined)}")
    root = Path(config["combined_output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(root / "metrics_summary.csv", index=False)
    (root / "run_record.md").write_text(
        "\n".join(
            [
                "# Token/LoRA/shared ablation metrics", "",
                f"- Completed: {datetime.now().isoformat(timespec='seconds')}",
                f"- RDKit: {rdBase.rdkitVersion}",
                f"- Existing 15-row summary: `{config['existing_fifteen_method_summary']}`",
                "- Added method: Shared token+LoRA (r=64, alpha=128)",
                "- Reported metrics: IntDiv, IntDiv2, Novelty, SNN, Uniq@10k, Uniq@20k, Uniq@30k, Validity",
            ]
        ) + "\n", encoding="utf-8"
    )


def main():
    args = args_parser()
    config = load_config(args.config)
    validate(config) if args.mode == "validate" else summarize(config)


if __name__ == "__main__":
    main()
