from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger, rdBase


REQUIRED_GENERATED_COLUMNS = ["SMILES", "Valid"]
SUMMARY_COLUMNS = [
    "Method",
    "Cyclization",
    "IntDiv",
    "IntDiv2",
    "Novelty",
    "SNN",
    "Uniq@10k",
    "Uniq@20k",
    "Uniq@30k",
    "Validity",
    "ValidCount",
    "TotalCount",
    "GeneratedCSV",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ablation generation CSVs and summarize selected MOSES metrics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("validate", "summarize"), required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "t", "yes"}:
        return True
    if text in {"false", "0", "f", "no"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def rdkit_valid(smiles: str) -> bool:
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    return Chem.MolFromSmiles(smiles.strip()) is not None


def validate_generated(config: dict) -> None:
    RDLogger.DisableLog("rdApp.*")
    expected_rows = int(config["expected_num_samples"])
    validation_rows = []
    for entry in config["entries"]:
        path = Path(entry["generated_csv"])
        dataframe = pd.read_csv(path)
        if list(dataframe.columns) != REQUIRED_GENERATED_COLUMNS:
            raise RuntimeError(
                f"Unexpected columns in {path}: {list(dataframe.columns)}"
            )
        if len(dataframe) != expected_rows:
            raise RuntimeError(
                f"Unexpected row count in {path}: expected {expected_rows}, received {len(dataframe)}"
            )
        recorded = dataframe["Valid"].map(parse_bool).tolist()
        recomputed = dataframe["SMILES"].fillna("").map(rdkit_valid).tolist()
        mismatches = sum(left != right for left, right in zip(recorded, recomputed))
        if mismatches:
            raise RuntimeError(f"RDKit Valid mismatch in {path}: {mismatches} rows")
        valid_count = int(sum(recomputed))
        validation_rows.append(
            {
                "Method": entry["method"],
                "Cyclization": entry["cyclization"],
                "TotalCount": len(dataframe),
                "ValidCount": valid_count,
                "InvalidCount": len(dataframe) - valid_count,
                "Validity": valid_count / len(dataframe),
                "GeneratedCSV": str(path),
            }
        )

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).to_csv(
        output_root / "generation_validation.csv", index=False
    )
    (output_root / "generation_validation.json").write_text(
        json.dumps(validation_rows, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Validated {len(validation_rows)} generated CSV files.")


def require_number(metrics: dict, key: str) -> float:
    if key not in metrics:
        raise KeyError(f"Missing metric: {key}")
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"Metric {key} is not finite: {value}")
    return value


def summarize_metrics(config: dict) -> None:
    rows = []
    bounded_keys = (
        "Novelty",
        "SNN/Test",
        "unique@10000",
        "unique@20000",
        "unique@30000",
        "validity",
    )
    for entry in config["entries"]:
        metrics_path = Path(entry["metrics_json"])
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        for key in bounded_keys:
            value = require_number(metrics, key)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Metric {key} is outside [0, 1]: {value}")
        row = {
            "Method": entry["method"],
            "Cyclization": entry["cyclization"],
            "IntDiv": require_number(metrics, "IntDiv"),
            "IntDiv2": require_number(metrics, "IntDiv2"),
            "Novelty": require_number(metrics, "Novelty"),
            "SNN": require_number(metrics, "SNN/Test"),
            "Uniq@10k": require_number(metrics, "unique@10000"),
            "Uniq@20k": require_number(metrics, "unique@20000"),
            "Uniq@30k": require_number(metrics, "unique@30000"),
            "Validity": require_number(metrics, "validity"),
            "ValidCount": int(metrics["num_valid"]),
            "TotalCount": int(metrics["num_total"]),
            "GeneratedCSV": entry["generated_csv"],
        }
        if row["TotalCount"] != int(config["expected_num_samples"]):
            raise RuntimeError(f"Unexpected metric total count: {row}")
        rows.append(row)

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "metrics_summary.csv"
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)
    run_record = [
        "# Token-only vs LoRA-only generation metrics",
        "",
        f"- Completed: {datetime.now().isoformat(timespec='seconds')}",
        f"- RDKit: {rdBase.rdkitVersion}",
        f"- Training reference: `{config['train_csv']}`",
        f"- Test reference: `{config['test_csv']}`",
        "- Reported metrics: IntDiv, IntDiv2, Novelty, SNN, Uniq@10k, Uniq@20k, Uniq@30k, Validity",
        "- SNN uses the historical `SNN/Test` definition.",
        "- `restored_validation.csv` was also used as the topology adaptation corpus; SNN is not an independent external-test estimate.",
        "",
        f"Summary: `{summary_path}`",
    ]
    (output_root / "run_record.md").write_text(
        "\n".join(run_record) + "\n", encoding="utf-8"
    )
    print(f"Saved selected metric summary: {summary_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "validate":
        validate_generated(config)
    else:
        summarize_metrics(config)


if __name__ == "__main__":
    main()
