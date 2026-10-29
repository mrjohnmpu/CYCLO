from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


WASSERSTEIN_PROPERTIES = {
    "MW": "MW",
    "LogP": "LogP",
    "TPSA": "TPSA",
    "HBD": "HBD",
    "HBA": "HBA",
    "RotBonds": "RotBonds",
    "Length": "Length",
    "Half-life": "half_life_value",
    "PAMPA": "PAMPA_value",
    "Caco-2": "Caco-2_value",
}

CLASSIFICATION_RATIO_PROPERTIES = {
    "Non-hemolytic": "hemolysis_positive",
    "Non-fouling": "non_fouling_positive",
    "Non-toxic": "toxicity_positive",
}


def merge_property_tables(rdkit_table: pd.DataFrame, peptiverse_table: pd.DataFrame) -> pd.DataFrame:
    common_cols = ["dataset_id", "dataset_label", "SMILES", "canonical_smiles"]
    merged = rdkit_table.merge(peptiverse_table, how="left", on=common_cols)
    return merged


def compute_wasserstein_summary(
    merged_table: pd.DataFrame,
    *,
    reference_dataset_id: str = "Reference",
) -> pd.DataFrame:
    reference = merged_table[merged_table["dataset_id"] == reference_dataset_id].copy()
    rows: list[dict] = []
    for dataset_id, group in merged_table.groupby("dataset_id"):
        if dataset_id == reference_dataset_id:
            continue
        row = {
            "dataset_id": dataset_id,
            "dataset_label": group["dataset_label"].iloc[0],
        }
        for display_name, column_name in WASSERSTEIN_PROPERTIES.items():
            ref_values = pd.to_numeric(reference[column_name], errors="coerce").dropna().to_numpy(dtype=float)
            gen_values = pd.to_numeric(group[column_name], errors="coerce").dropna().to_numpy(dtype=float)
            if ref_values.size == 0 or gen_values.size == 0:
                row[display_name] = np.nan
            else:
                row[display_name] = float(wasserstein_distance(gen_values, ref_values))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_positive_ratio_summary(
    merged_table: pd.DataFrame,
    *,
    dataset_order: Iterable[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = merged_table.groupby(["dataset_id", "dataset_label"])
    for (dataset_id, dataset_label), group in grouped:
        row = {
            "dataset_id": dataset_id,
            "dataset_label": dataset_label,
        }
        for display_name, column_name in CLASSIFICATION_RATIO_PROPERTIES.items():
            values = pd.to_numeric(group[column_name], errors="coerce").dropna().to_numpy(dtype=float)
            row[display_name] = float(np.mean(values)) if values.size else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    if dataset_order is not None and not summary.empty:
        order_map = {dataset_id: idx for idx, dataset_id in enumerate(dataset_order)}
        summary["__order"] = summary["dataset_id"].map(order_map).fillna(len(order_map))
        summary = summary.sort_values(["__order", "dataset_label"]).drop(columns="__order")
    return summary

