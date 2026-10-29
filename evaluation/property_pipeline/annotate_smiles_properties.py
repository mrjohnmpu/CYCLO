#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from property_pipeline.data_inventory import DatasetSpec
from property_pipeline.mol_cache import build_dataset_cache
from property_pipeline.peptiverse_batch import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_WEIGHT_ROOT,
    PeptiVerseBatchPredictor,
    run_peptiverse_predictions,
)
from property_pipeline.rdkit_properties import compute_basic_rdkit_property_table


SMILES_COLUMN_CANDIDATES = ("SMILES", "Smiles", "smiles")
RDKIT_OUTPUT_COLUMNS = [
    "MW",
    "LogP",
    "HBD",
    "HBA",
    "TPSA",
    "RotBonds",
]
PEPTIVERSE_OUTPUT_COLUMNS = [
    "hemolysis_score",
    "non_fouling_score",
    "toxicity_score",
    "half_life_value",
    "PAMPA_value",
    "Caco-2_value",
]
PEPTIVERSE_FAILURE_COLUMNS = {
    "hemolysis": "hemolysis_failed",
    "non_fouling": "non_fouling_failed",
    "toxicity": "toxicity_failed",
    "half_life": "half_life_failed",
    "PAMPA": "PAMPA_failed",
    "Caco-2": "Caco-2_failed",
}


@dataclass
class InputJob:
    csv_path: Path
    output_stem: str
    smiles_col_override: str | None = None
    drop_output_columns: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate generated SMILES CSVs with RDKit and PeptiVerse properties."
    )
    parser.add_argument("csv_paths", nargs="*", help="One or more CSV files to annotate.")
    parser.add_argument("--config", type=Path, default=None, help="JSON config with batch inputs.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for outputs.")
    parser.add_argument("--smiles-col", type=str, default=None, help="Global SMILES column override.")
    parser.add_argument("--device", type=str, default=None, help="Use cpu or cuda:0. Default is cuda:0.")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Optional PeptiVerse manifest override.")
    parser.add_argument("--weight-root", type=Path, default=None, help="Optional PeptiVerse weight root override.")
    return parser.parse_args()


def load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_jobs(args: argparse.Namespace, config: dict[str, Any]) -> list[InputJob]:
    if args.config and args.csv_paths:
        raise ValueError("Pass either csv_paths or --config, not both.")
    if not args.config and not args.csv_paths:
        raise ValueError("Provide at least one csv_path or a --config file.")

    if args.config:
        inputs = config.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("Config must contain a non-empty 'inputs' list.")
        jobs: list[InputJob] = []
        for item in inputs:
            csv_path = Path(str(item["csv_path"]))
            name = str(item.get("name") or csv_path.stem)
            smiles_col = item.get("smiles_col")
            drop_output_columns = item.get("drop_output_columns")
            jobs.append(
                InputJob(
                    csv_path=csv_path,
                    output_stem=name,
                    smiles_col_override=smiles_col,
                    drop_output_columns=None if drop_output_columns is None else list(drop_output_columns),
                )
            )
        return jobs

    return [InputJob(csv_path=Path(path), output_stem=Path(path).stem) for path in args.csv_paths]


def resolve_device(args: argparse.Namespace, config: dict[str, Any]) -> str:
    device = args.device or config.get("device") or "cuda:0"
    if device == "cpu":
        return device
    if device != "cuda:0":
        raise ValueError(f"Unsupported device '{device}'. Only cpu and cuda:0 are allowed.")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 was requested but no CUDA device is available.")
    return device


def resolve_manifest_path(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    return Path(args.manifest_path or config.get("manifest_path") or DEFAULT_MANIFEST_PATH)


def resolve_weight_root(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    return Path(args.weight_root or config.get("weight_root") or DEFAULT_WEIGHT_ROOT)


def detect_smiles_column(dataframe: pd.DataFrame, override: str | None, global_override: str | None) -> str:
    for candidate in (override, global_override):
        if candidate:
            if candidate not in dataframe.columns:
                raise ValueError(f"Missing SMILES column '{candidate}' in input columns {list(dataframe.columns)}")
            return candidate

    for candidate in SMILES_COLUMN_CANDIDATES:
        if candidate in dataframe.columns:
            return candidate
    raise ValueError(f"Unable to detect SMILES column from {list(dataframe.columns)}")


def resolve_output_path(
    job: InputJob,
    *,
    cli_output_dir: Path | None,
    config_output_dir: Any,
) -> Path:
    output_root = cli_output_dir or (Path(str(config_output_dir)) if config_output_dir else None)
    if output_root is None:
        output_root = job.csv_path.parent
    return output_root / f"{job.output_stem}.with_properties.csv"


def count_nonempty_smiles(dataframe: pd.DataFrame, smiles_col: str) -> int:
    series = dataframe[smiles_col]
    cleaned = series.where(series.notna(), "").astype(str).str.strip()
    mask = series.notna() & cleaned.ne("") & cleaned.str.lower().ne("nan")
    return int(mask.sum())


def build_dataset_spec(job: InputJob, smiles_col: str) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=job.output_stem,
        dataset_label=job.output_stem,
        source_csv=str(job.csv_path),
        smiles_col=smiles_col,
        valid_col=None,
        cyclization_col=None,
        dataset_kind="generated",
        seed="n/a",
        regenerated=False,
    )


def collect_failure_counts(prediction_table: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for property_name, column_name in PEPTIVERSE_FAILURE_COLUMNS.items():
        if column_name not in prediction_table.columns:
            counts[property_name] = 0
            continue
        counts[property_name] = int(prediction_table[column_name].fillna(False).astype(bool).sum())
    return counts


def ensure_columns(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in dataframe.columns:
            dataframe[column] = pd.Series(dtype=float)
    return dataframe


def annotate_one_csv(
    job: InputJob,
    *,
    output_csv: Path,
    predictor: PeptiVerseBatchPredictor,
    global_smiles_col: str | None,
) -> dict[str, Any]:
    source_df = pd.read_csv(job.csv_path)
    smiles_col = detect_smiles_column(source_df, job.smiles_col_override, global_smiles_col)
    spec = build_dataset_spec(job, smiles_col)
    cache = build_dataset_cache(spec, dedupe_canonical=True)

    rdkit_table = compute_basic_rdkit_property_table(cache)
    prediction_table = run_peptiverse_predictions(cache, predictor)
    rdkit_table = ensure_columns(rdkit_table, RDKIT_OUTPUT_COLUMNS)
    prediction_table = ensure_columns(
        prediction_table,
        PEPTIVERSE_OUTPUT_COLUMNS + list(PEPTIVERSE_FAILURE_COLUMNS.values()),
    )

    annotated = cache.valid_dataframe.reset_index(drop=True).copy()
    annotated = annotated.drop(columns=["rdkit_valid"], errors="ignore")
    annotated = pd.concat(
        [
            annotated,
            rdkit_table[RDKIT_OUTPUT_COLUMNS].reset_index(drop=True),
            prediction_table[PEPTIVERSE_OUTPUT_COLUMNS].reset_index(drop=True),
        ],
        axis=1,
    )
    if job.drop_output_columns:
        annotated = annotated.drop(columns=job.drop_output_columns, errors="ignore")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_csv, index=False)

    return {
        "input_csv": str(job.csv_path),
        "output_csv": str(output_csv),
        "rows_in": int(len(source_df)),
        "rows_nonempty_smiles": count_nonempty_smiles(source_df, smiles_col),
        "rows_rdkit_valid": int(cache.dataframe["rdkit_valid"].sum()),
        "rows_unique_canonical": int(len(cache.valid_dataframe)),
        "rows_written": int(len(annotated)),
        "peptiverse_failure_counts": collect_failure_counts(prediction_table),
    }


def ensure_unique_output_paths(output_paths: list[Path]) -> None:
    seen: set[Path] = set()
    duplicates: list[str] = []
    for path in output_paths:
        if path in seen:
            duplicates.append(str(path))
        seen.add(path)
    if duplicates:
        raise ValueError(f"Duplicate output paths detected: {duplicates}")


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config) if args.config else {}
    jobs = build_jobs(args, config)
    device = resolve_device(args, config)
    manifest_path = resolve_manifest_path(args, config)
    weight_root = resolve_weight_root(args, config)
    output_paths = [
        resolve_output_path(job, cli_output_dir=args.output_dir, config_output_dir=config.get("output_dir"))
        for job in jobs
    ]
    ensure_unique_output_paths(output_paths)

    predictor = PeptiVerseBatchPredictor.load(
        manifest_path=manifest_path,
        weight_root=weight_root,
        device=device,
    )

    reports: list[dict[str, Any]] = []
    for job, output_csv in zip(jobs, output_paths):
        report = annotate_one_csv(
            job,
            output_csv=output_csv,
            predictor=predictor,
            global_smiles_col=args.smiles_col,
        )
        reports.append(report)
        predictor.clear_runtime_cache()

    print(json.dumps({"reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
