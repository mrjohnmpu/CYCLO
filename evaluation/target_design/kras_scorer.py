import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem


RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


DEFAULT_MODEL_PATH = Path('./data/kras_kd/kras_xgboost_reg.pkl')
DEFAULT_INVALID_PRED = 5.0


def reverse_sigmoid(scores: np.ndarray, low: float = 0.0, high: float = 2.0, k: float = 1.0) -> np.ndarray:
    """Match the original TransformFunction('rsigmoid', 0, 2, k=1.0)."""
    midpoint = (high + low) / 2.0
    scale = high - low
    return np.asarray(
        [1.0 / (1.0 + 10.0 ** (k * (float(score) - midpoint) / scale)) for score in scores],
        dtype=np.float32,
    )


def fingerprint_from_mol(molecule: Chem.Mol, radius: int = 3, size: int = 2048) -> np.ndarray:
    fp_bits = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=size)
    fp_np = np.zeros((size,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_bits, fp_np)
    return fp_np


def smiles_to_features(smiles: Iterable[str], radius: int = 3, size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    valid_mask: List[bool] = []

    for value in smiles:
        smi = "" if pd.isna(value) else str(value).strip()
        mol = Chem.MolFromSmiles(smi) if smi else None
        valid = mol is not None
        valid_mask.append(valid)
        if valid:
            features.append(fingerprint_from_mol(mol, radius=radius, size=size))

    if features:
        return np.vstack(features).astype(np.float32), np.asarray(valid_mask, dtype=bool)
    return np.zeros((0, size), dtype=np.float32), np.asarray(valid_mask, dtype=bool)


class KRASSmilesScorer:
    """SMILES-only KRAS inhibition scorer migrated from helm-gpt's kras.py."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, radius: int = 3, fp_size: int = 2048):
        self.model_path = Path(model_path)
        self.radius = int(radius)
        self.fp_size = int(fp_size)
        self.predictor = joblib.load(self.model_path)
        model_features = getattr(self.predictor, "n_features_in_", None)
        if model_features is not None and int(model_features) != self.fp_size:
            raise ValueError(f"Model expects {model_features} features, but fp_size={self.fp_size}")

    def score_smiles(self, smiles: Iterable[str]) -> pd.DataFrame:
        smiles_list = list(smiles)
        features, valid_mask = smiles_to_features(smiles_list, radius=self.radius, size=self.fp_size)

        preds = np.full(len(smiles_list), DEFAULT_INVALID_PRED, dtype=np.float32)
        if len(features):
            valid_preds = self.predictor.predict(np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max))
            preds[np.where(valid_mask)[0]] = np.asarray(valid_preds, dtype=np.float32)

        return pd.DataFrame(
            {
                "kras_valid": valid_mask.astype(bool),
                "kras_kd_pred": preds.astype(float),
                "kras_score": reverse_sigmoid(preds).astype(float),
            }
        )

    __call__ = score_smiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append SMILES-only KRAS XGBoost scores to a CSV.")
    parser.add_argument("--input_csv", required=True, help="Input CSV containing SMILES.")
    parser.add_argument("--output_csv", required=True, help="Output CSV with KRAS score columns appended.")
    parser.add_argument("--smiles_col", default="SMILES", help="Name of the SMILES column.")
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH), help="Path to kras_xgboost_reg.pkl.")
    parser.add_argument("--metadata_json", default=None, help="Optional path for a small run metadata JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    df = pd.read_csv(input_csv)
    if args.smiles_col not in df.columns:
        raise KeyError(f"Missing smiles column={args.smiles_col} in {input_csv}")

    scorer = KRASSmilesScorer(args.model_path)
    score_df = scorer.score_smiles(df[args.smiles_col].tolist())
    out_df = pd.concat([df.reset_index(drop=True), score_df], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    if args.metadata_json:
        metadata = {
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "smiles_col": args.smiles_col,
            "model_path": str(Path(args.model_path)),
            "num_rows": int(len(out_df)),
            "num_valid": int(score_df["kras_valid"].sum()),
            "num_invalid": int((~score_df["kras_valid"]).sum()),
        }
        Path(args.metadata_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metadata_json).write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
