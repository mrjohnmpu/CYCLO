from __future__ import annotations

import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rdkit import Chem, RDLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.release_paths import add_moses_path
add_moses_path()

RDLogger.DisableLog("rdApp.*")
UNIQUE_PREFIX_THRESHOLDS = [10000, 20000, 30000]


def load_smiles_from_csv(path: Path, smiles_col: str, cyclization_type: str | None = None, type_col: str = "Cyclization") -> list[str]:
    dataframe = pd.read_csv(path)
    if cyclization_type is not None and type_col in dataframe.columns:
        dataframe = dataframe[dataframe[type_col].astype(str) == str(cyclization_type)]
    if smiles_col not in dataframe.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {path}")
    smiles = [str(item).strip() for item in dataframe[smiles_col].dropna().tolist()]
    return [item for item in smiles if item]


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _parse_valid_value(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, int)):
        return bool(value)
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return bool(int(value))

    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "t", "1", "yes", "y", "valid"}:
        return True
    if text in {"false", "f", "0", "no", "n", "invalid"}:
        return False
    try:
        return bool(int(float(text)))
    except ValueError:
        return None


def build_valid_mask(dataframe: pd.DataFrame, smiles_col: str, valid_col: str):
    fallback = dataframe[smiles_col].fillna("").map(lambda smiles: Chem.MolFromSmiles(str(smiles).strip()) is not None)
    if valid_col not in dataframe.columns:
        return fallback

    parsed = dataframe[valid_col].map(_parse_valid_value)
    missing_mask = parsed.isna()
    if missing_mask.any():
        parsed = parsed.where(~missing_mask, fallback)
    return parsed.astype(bool)


def compute_prefix_uniqueness(dataframe: pd.DataFrame, smiles_col: str, valid_col: str, thresholds: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    valid_mask = build_valid_mask(dataframe, smiles_col, valid_col)
    for threshold in thresholds:
        prefix_df = dataframe.iloc[: min(threshold, len(dataframe))].copy()
        prefix_valid = prefix_df[valid_mask.iloc[: len(prefix_df)].values]
        smiles = [canonicalize_smiles(item) for item in prefix_valid[smiles_col].dropna().tolist()]
        smiles = [item for item in smiles if item]
        metrics[f"unique@{threshold}"] = (len(set(smiles)) / len(smiles)) if smiles else 0.0
    return metrics


def load_generated_smiles(path: Path, smiles_col: str, valid_col: str, cyclization_type: str | None = None, type_col: str = "Cyclization"):
    dataframe = pd.read_csv(path)
    if cyclization_type is not None and type_col in dataframe.columns:
        dataframe = dataframe[dataframe[type_col].astype(str) == str(cyclization_type)]
    if smiles_col not in dataframe.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {path}")

    total_count = len(dataframe)
    valid_df = dataframe[build_valid_mask(dataframe, smiles_col, valid_col)].copy()
    valid_smiles = [str(item).strip() for item in valid_df[smiles_col].dropna().tolist()]
    valid_smiles = [item for item in valid_smiles if item]
    unique_valid = len(set(valid_smiles))

    summary = {
        "num_total": total_count,
        "num_valid": len(valid_smiles),
        "validity": (len(valid_smiles) / total_count) if total_count else 0.0,
        "num_unique_valid": unique_valid,
        "uniqueness_valid": (unique_valid / len(valid_smiles)) if valid_smiles else 0.0,
        "dup_rate_valid": 1.0 - ((unique_valid / len(valid_smiles)) if valid_smiles else 0.0),
    }
    prefix_uniqueness = compute_prefix_uniqueness(dataframe, smiles_col, valid_col, UNIQUE_PREFIX_THRESHOLDS)
    return valid_smiles, summary, prefix_uniqueness


def evaluate_generated_csv(
    *,
    generated_csv: str | Path,
    train_csv: str | Path,
    test_csv: str | Path,
    experiment_name: str | None = None,
    cyclization_type: str | None = "head_to_tail",
    smiles_col: str = "SMILES",
    type_col: str = "Cyclization",
    valid_col: str = "Valid",
    device: str | None = None,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    from moses.metrics import get_all_metrics

    generated_csv = Path(generated_csv).resolve()
    train_csv = Path(train_csv).resolve()
    test_csv = Path(test_csv).resolve()
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_n_jobs = n_jobs or max(1, min(8, multiprocessing.cpu_count()))

    train_smiles = load_smiles_from_csv(train_csv, smiles_col, cyclization_type, type_col)
    test_smiles = load_smiles_from_csv(test_csv, smiles_col, cyclization_type, type_col)
    generated_smiles, summary, prefix_uniqueness = load_generated_smiles(
        generated_csv,
        smiles_col,
        valid_col,
        cyclization_type,
        type_col,
    )

    if not generated_smiles:
        raise ValueError(f"No valid generated SMILES found in {generated_csv}")

    start = time.time()
    metrics = get_all_metrics(
        gen=generated_smiles,
        test=test_smiles,
        train=train_smiles,
        k=[],
        n_jobs=resolved_n_jobs,
        device=resolved_device,
    )
    metrics.pop("valid", None)
    elapsed = time.time() - start

    return {
        "experiment_name": experiment_name or generated_csv.stem,
        "generated_csv": str(generated_csv),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "cyclization_type": cyclization_type,
        "n_jobs": resolved_n_jobs,
        "device": resolved_device,
        "elapsed_seconds": elapsed,
        **summary,
        **prefix_uniqueness,
        **metrics,
    }


def write_reports(output_dir: str | Path, prefix: str, metrics_row: dict[str, Any]):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}.json"
    txt_path = output_dir / f"{prefix}.txt"

    dataframe = pd.DataFrame([metrics_row]).reindex(sorted(metrics_row.keys()), axis=1)
    dataframe.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metrics_row, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    lines = ["Evaluation Report", "================="]
    for key in sorted(metrics_row.keys()):
        lines.append(f"{key}: {metrics_row[key]}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, txt_path
