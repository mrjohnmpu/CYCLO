from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


DEFAULT_REFERENCE_CSV = Path('./data/restored_validation.csv')
DEFAULT_MANIFEST_CSV = Path(
    'results/benchmark_head_to_tail_baselines_20260411/manifest/benchmark_inputs_with_ours.csv'
)


@dataclass
class DatasetSpec:
    dataset_id: str
    dataset_label: str
    source_csv: str
    smiles_col: str = "SMILES"
    valid_col: Optional[str] = None
    cyclization_col: Optional[str] = "Cyclization"
    dataset_kind: str = "generated"
    seed: str = "unknown"
    regenerated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def load_manifest_datasets(manifest_csv: Path) -> list[DatasetSpec]:
    dataframe = pd.read_csv(manifest_csv)
    entries: list[DatasetSpec] = []
    for row in dataframe.to_dict(orient="records"):
        entries.append(
            DatasetSpec(
                dataset_id=str(row["model_id"]),
                dataset_label=str(row.get("model_label", row["model_id"])),
                source_csv=str(row["source_csv"]),
                smiles_col=str(row.get("smiles_col", "SMILES")),
                valid_col=None if pd.isna(row.get("valid_col")) else str(row.get("valid_col")),
                cyclization_col="Cyclization",
                dataset_kind="generated",
                seed=str(row.get("seed", "unknown")),
                regenerated=bool(row.get("regenerated", False)),
            )
        )
    return entries


def build_reference_dataset(reference_csv: Path) -> DatasetSpec:
    return DatasetSpec(
        dataset_id="Reference",
        dataset_label="Reference",
        source_csv=str(reference_csv),
        smiles_col="SMILES",
        valid_col=None,
        cyclization_col="Cyclization",
        dataset_kind="reference",
        seed="n/a",
        regenerated=False,
    )


def read_source_dataframe(spec: DatasetSpec) -> pd.DataFrame:
    dataframe = pd.read_csv(spec.source_csv)
    if spec.smiles_col not in dataframe.columns:
        raise ValueError(f"{spec.source_csv} is missing required SMILES column '{spec.smiles_col}'")
    return dataframe


def filter_dataframe_by_cyclization(
    dataframe: pd.DataFrame,
    *,
    cyclization_type: Optional[str],
    cyclization_col: Optional[str],
) -> pd.DataFrame:
    if not cyclization_type or not cyclization_col or cyclization_col not in dataframe.columns:
        return dataframe.copy()
    filtered = dataframe[dataframe[cyclization_col].astype(str) == str(cyclization_type)].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def build_inventory_rows(
    datasets: Iterable[DatasetSpec],
    *,
    cyclization_type: Optional[str],
) -> list[dict]:
    inventory_rows: list[dict] = []
    for spec in datasets:
        source_path = Path(spec.source_csv)
        exists = source_path.exists()
        row = spec.to_dict()
        row.update(
            {
                "exists": exists,
                "source_path": str(source_path),
                "filtered_cyclization_type": cyclization_type,
                "num_rows_raw": None,
                "num_rows_after_cyclization_filter": None,
                "source_columns": None,
            }
        )
        if exists:
            dataframe = read_source_dataframe(spec)
            filtered = filter_dataframe_by_cyclization(
                dataframe,
                cyclization_type=cyclization_type,
                cyclization_col=spec.cyclization_col,
            )
            row["num_rows_raw"] = int(len(dataframe))
            row["num_rows_after_cyclization_filter"] = int(len(filtered))
            row["source_columns"] = "|".join(list(dataframe.columns))
        inventory_rows.append(row)
    return inventory_rows


def write_inventory(inventory_rows: list[dict], output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory_rows).to_csv(output_csv, index=False)
    return output_csv

