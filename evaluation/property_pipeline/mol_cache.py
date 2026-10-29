from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem, RDLogger

from .data_inventory import DatasetSpec, filter_dataframe_by_cyclization, read_source_dataframe


@dataclass
class DatasetCache:
    spec: DatasetSpec
    dataframe: pd.DataFrame
    valid_dataframe: pd.DataFrame
    mols: list[Chem.Mol]


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        RDLogger.DisableLog("rdApp.error")
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None
    finally:
        RDLogger.EnableLog("rdApp.error")


def build_dataset_cache(
    spec: DatasetSpec,
    *,
    cyclization_type: Optional[str] = None,
    max_rows: Optional[int] = None,
    dedupe_canonical: bool = False,
) -> DatasetCache:
    dataframe = read_source_dataframe(spec)
    filtered = filter_dataframe_by_cyclization(
        dataframe,
        cyclization_type=cyclization_type,
        cyclization_col=spec.cyclization_col,
    )
    if max_rows is not None:
        filtered = filtered.head(int(max_rows)).copy()
    filtered = filtered.reset_index(drop=True)

    mols: list[Chem.Mol] = []
    validity_flags: list[bool] = []
    canonical_smiles: list[Optional[str]] = []

    for smiles in filtered[spec.smiles_col].astype(str).tolist():
        mol = smiles_to_mol(smiles)
        if mol is None:
            mols.append(None)  # type: ignore[arg-type]
            validity_flags.append(False)
            canonical_smiles.append(None)
            continue
        mols.append(mol)
        validity_flags.append(True)
        canonical_smiles.append(Chem.MolToSmiles(mol))

    annotated = filtered.copy()
    annotated["source_smiles"] = annotated[spec.smiles_col].astype(str)
    annotated["rdkit_valid"] = validity_flags
    annotated["canonical_smiles"] = canonical_smiles

    valid_dataframe = annotated[annotated["rdkit_valid"]].copy()
    if dedupe_canonical:
        valid_dataframe = valid_dataframe.drop_duplicates(subset=["canonical_smiles"], keep="first")
    valid_indices = valid_dataframe.index.tolist()
    valid_dataframe = valid_dataframe.reset_index(drop=True)
    valid_mols = [mols[idx] for idx in valid_indices]
    return DatasetCache(spec=spec, dataframe=annotated, valid_dataframe=valid_dataframe, mols=valid_mols)


def write_cache_tables(cache: DatasetCache, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_csv = output_dir / f"{cache.spec.dataset_id}_with_validity.csv"
    valid_csv = output_dir / f"{cache.spec.dataset_id}_valid_smiles.csv"
    cache.dataframe.to_csv(full_csv, index=False)
    cache.valid_dataframe.to_csv(valid_csv, index=False)
    return full_csv, valid_csv
