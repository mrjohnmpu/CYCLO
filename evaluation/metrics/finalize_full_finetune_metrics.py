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


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("validate", "summarize"), required=True)
    return parser.parse_args()


def config_from(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    for entry in config["full_finetune_entries"]:
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
                "Method": "Full fine-tune",
                "Cyclization": entry["cyclization"],
                "TotalCount": len(dataframe),
                "ValidCount": int(sum(recomputed)),
                "InvalidCount": len(dataframe) - int(sum(recomputed)),
                "Validity": sum(recomputed) / len(dataframe),
                "GeneratedCSV": str(path),
            }
        )
    root = Path(config["combined_output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "full_finetune_generation_validation.csv", index=False)


def metric(metrics, key):
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite metric {key}: {value}")
    return value


def metric_row(method, entry, metrics):
    row = {
        "Method": method,
        "Cyclization": entry["cyclization"],
        "IntDiv": metric(metrics, "IntDiv"),
        "IntDiv2": metric(metrics, "IntDiv2"),
        "Novelty": metric(metrics, "Novelty"),
        "SNN": metric(metrics, "SNN/Test"),
        "Uniq@10k": metric(metrics, "unique@10000"),
        "Uniq@20k": metric(metrics, "unique@20000"),
        "Uniq@30k": metric(metrics, "unique@30000"),
        "Validity": metric(metrics, "validity"),
        "ValidCount": int(metrics["num_valid"]),
        "TotalCount": int(metrics["num_total"]),
        "GeneratedCSV": entry["generated_csv"],
    }
    for key in ("Novelty", "SNN", "Uniq@10k", "Uniq@20k", "Uniq@30k", "Validity"):
        if not 0 <= row[key] <= 1:
            raise ValueError(f"Metric {key} is outside [0,1]: {row[key]}")
    return row


def summarize(config):
    existing = pd.read_csv(config["existing_nine_method_summary"])
    if list(existing.columns) != SUMMARY_COLUMNS or len(existing) != 9:
        raise RuntimeError("Existing nine-row factorial summary has an unexpected schema or row count.")
    additions = []
    for method, entries in (
        ("Paper-exact token+LoRA", config["paper_entries"]),
        ("Full fine-tune", config["full_finetune_entries"]),
    ):
        for entry in entries:
            with Path(entry["metrics_json"]).open("r", encoding="utf-8") as handle:
                additions.append(metric_row(method, entry, json.load(handle)))
    combined = pd.concat([existing, pd.DataFrame(additions, columns=SUMMARY_COLUMNS)], ignore_index=True)
    method_order = ["Token-only", "LoRA-only", "Paper-exact token+LoRA", "Strict token+LoRA", "Full fine-tune"]
    type_order = ["head_to_tail", "sc_to_tail", "disulfide_bridge"]
    combined["_method"] = pd.Categorical(combined["Method"], method_order, ordered=True)
    combined["_type"] = pd.Categorical(combined["Cyclization"], type_order, ordered=True)
    combined = combined.sort_values(["_method", "_type"]).drop(columns=["_method", "_type"])
    if len(combined) != 15:
        raise RuntimeError(f"Expected 15 rows, received {len(combined)}")
    root = Path(config["combined_output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(root / "metrics_summary.csv", index=False)
    (root / "run_record.md").write_text(
        "\n".join(
            [
                "# Token/LoRA/full-finetune ablation metrics",
                "",
                f"- Completed: {datetime.now().isoformat(timespec='seconds')}",
                f"- RDKit: {rdBase.rdkitVersion}",
                f"- Existing nine-row summary: `{config['existing_nine_method_summary']}`",
                "- Added methods: Paper-exact token+LoRA; Full fine-tune",
                "- Reported metrics: IntDiv, IntDiv2, Novelty, SNN, Uniq@10k, Uniq@20k, Uniq@30k, Validity",
                "- SNN uses restored_validation.csv, which also supplied the topology adaptation split.",
            ]
        ) + "\n",
        encoding="utf-8",
    )


def main():
    args = args_parser()
    config = config_from(args.config)
    validate(config) if args.mode == "validate" else summarize(config)


if __name__ == "__main__":
    main()
