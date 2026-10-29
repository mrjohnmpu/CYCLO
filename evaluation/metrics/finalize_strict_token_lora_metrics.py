from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger, rdBase

SUMMARY_COLUMNS = [
    "Method", "Cyclization", "IntDiv", "IntDiv2", "Novelty", "SNN",
    "Uniq@10k", "Uniq@20k", "Uniq@30k", "Validity", "ValidCount",
    "TotalCount", "GeneratedCSV",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("validate", "summarize"), required=True)
    return parser.parse_args()


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_bool(value):
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
    for entry in config["strict_entries"]:
        path = Path(entry["generated_csv"])
        dataframe = pd.read_csv(path)
        if list(dataframe.columns) != ["SMILES", "Valid"]:
            raise RuntimeError(f"Unexpected columns in {path}: {list(dataframe.columns)}")
        if len(dataframe) != config["expected_num_samples"]:
            raise RuntimeError(f"Unexpected row count in {path}: {len(dataframe)}")
        recorded = dataframe["Valid"].map(parse_bool).tolist()
        recomputed = dataframe["SMILES"].fillna("").map(
            lambda value: bool(value) and Chem.MolFromSmiles(str(value).strip()) is not None
        ).tolist()
        mismatches = sum(left != right for left, right in zip(recorded, recomputed))
        if mismatches:
            raise RuntimeError(f"RDKit validity mismatch in {path}: {mismatches}")
        rows.append(
            {
                "Method": "Strict token+LoRA",
                "Cyclization": entry["cyclization"],
                "TotalCount": len(dataframe),
                "ValidCount": int(sum(recomputed)),
                "InvalidCount": len(dataframe) - int(sum(recomputed)),
                "Validity": sum(recomputed) / len(dataframe),
                "GeneratedCSV": str(path),
            }
        )
    output_root = Path(config["factorial_output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_root / "strict_generation_validation.csv", index=False)


def number(metrics, key):
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite metric {key}: {value}")
    return value


def summarize(config):
    existing = pd.read_csv(config["existing_six_method_summary"])
    if list(existing.columns) != SUMMARY_COLUMNS or len(existing) != 6:
        raise RuntimeError("Existing Token-only/LoRA-only summary has an unexpected schema or row count.")
    strict_rows = []
    for entry in config["strict_entries"]:
        with Path(entry["metrics_json"]).open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        row = {
            "Method": "Strict token+LoRA",
            "Cyclization": entry["cyclization"],
            "IntDiv": number(metrics, "IntDiv"),
            "IntDiv2": number(metrics, "IntDiv2"),
            "Novelty": number(metrics, "Novelty"),
            "SNN": number(metrics, "SNN/Test"),
            "Uniq@10k": number(metrics, "unique@10000"),
            "Uniq@20k": number(metrics, "unique@20000"),
            "Uniq@30k": number(metrics, "unique@30000"),
            "Validity": number(metrics, "validity"),
            "ValidCount": int(metrics["num_valid"]),
            "TotalCount": int(metrics["num_total"]),
            "GeneratedCSV": entry["generated_csv"],
        }
        for key in ("Novelty", "SNN", "Uniq@10k", "Uniq@20k", "Uniq@30k", "Validity"):
            if not 0.0 <= row[key] <= 1.0:
                raise ValueError(f"Metric {key} is outside [0,1]: {row[key]}")
        strict_rows.append(row)
    combined = pd.concat([existing, pd.DataFrame(strict_rows, columns=SUMMARY_COLUMNS)], ignore_index=True)
    if len(combined) != 9:
        raise RuntimeError(f"Expected 9 summary rows, received {len(combined)}")
    output_root = Path(config["factorial_output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_root / "metrics_summary.csv", index=False)
    (output_root / "run_record.md").write_text(
        "\n".join(
            [
                "# Token/LoRA factorial metrics",
                "",
                f"- Completed: {datetime.now().isoformat(timespec='seconds')}",
                f"- RDKit: {rdBase.rdkitVersion}",
                f"- Existing six-row summary: `{config['existing_six_method_summary']}`",
                "- Added method: Strict token+LoRA",
                "- Reported metrics: IntDiv, IntDiv2, Novelty, SNN, Uniq@10k, Uniq@20k, Uniq@30k, Validity",
                "- SNN uses restored_validation.csv, which also supplied the topology adaptation split.",
            ]
        ) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    validate(config) if args.mode == "validate" else summarize(config)


if __name__ == "__main__":
    main()
