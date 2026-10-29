from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors

from .mol_cache import DatasetCache


TEST2_ROOT = Path(__file__).resolve().parents[2]
LENGTH_SCRIPT = TEST2_ROOT / "evaluation" / "analysis" / "compute_peptide_lengths.py"


@dataclass
class LengthEstimator:
    module: Any
    type_oracle: Any

    @classmethod
    def load(cls) -> "LengthEstimator":
        spec = importlib.util.spec_from_file_location("compute_peptide_lengths", LENGTH_SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load canonical length estimator from {LENGTH_SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        type_oracle = module.load_type_oracle(TEST2_ROOT)
        return cls(module=module, type_oracle=type_oracle)

    def infer_length(self, smiles: str) -> dict[str, Any]:
        return self.module.infer_length_from_smiles(smiles, self.type_oracle)


def compute_basic_rdkit_property_table(cache: DatasetCache) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row, mol in zip(cache.valid_dataframe.to_dict(orient="records"), cache.mols):
        rows.append(
            {
                "dataset_id": cache.spec.dataset_id,
                "dataset_label": cache.spec.dataset_label,
                "SMILES": str(row["source_smiles"]),
                "canonical_smiles": row.get("canonical_smiles"),
                "MW": float(rdMolDescriptors.CalcExactMolWt(mol)),
                "LogP": float(Crippen.MolLogP(mol)),
                "HBD": int(Lipinski.NumHDonors(mol)),
                "HBA": int(Lipinski.NumHAcceptors(mol)),
                "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
                "RotBonds": int(Lipinski.NumRotatableBonds(mol)),
            }
        )
    return pd.DataFrame(rows)


def compute_rdkit_property_table(cache: DatasetCache, length_estimator: LengthEstimator) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row, mol in zip(cache.valid_dataframe.to_dict(orient="records"), cache.mols):
        smiles = str(row["source_smiles"])
        length_info = length_estimator.infer_length(smiles)
        rows.append(
            {
                "dataset_id": cache.spec.dataset_id,
                "dataset_label": cache.spec.dataset_label,
                "SMILES": smiles,
                "canonical_smiles": row.get("canonical_smiles"),
                "MW": float(rdMolDescriptors.CalcExactMolWt(mol)),
                "LogP": float(Crippen.MolLogP(mol)),
                "HBD": int(Lipinski.NumHDonors(mol)),
                "HBA": int(Lipinski.NumHAcceptors(mol)),
                "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
                "RotBonds": int(Lipinski.NumRotatableBonds(mol)),
                "Length": length_info["peptide_length"],
                "length_valid_smiles": bool(length_info["valid_smiles"]),
                "length_failure_reasons": ";".join(length_info["failure_reasons"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_rdkit_properties(property_table: pd.DataFrame) -> pd.DataFrame:
    stats_rows: list[dict[str, Any]] = []
    property_names = ["MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds", "Length"]
    for (dataset_id, dataset_label), group in property_table.groupby(["dataset_id", "dataset_label"]):
        for property_name in property_names:
            values = pd.to_numeric(group[property_name], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            stats_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": dataset_label,
                    "property": property_name,
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(stats_rows)
