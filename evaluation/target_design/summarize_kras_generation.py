from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize KRAS-scored generation CSVs and check PPO gate criteria.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--smiles_col", default="SMILES")
    parser.add_argument("--valid_col", default="Valid")
    parser.add_argument("--kras_valid_col", default="kras_valid")
    parser.add_argument("--kd_col", default="kras_kd_pred")
    parser.add_argument("--min_valid_rate", type=float, default=0.95)
    parser.add_argument("--min_unique_rate", type=float, default=0.95)
    parser.add_argument("--min_kd_le_1p0_rate", type=float, default=0.30)
    parser.add_argument("--min_kd_le_0p8_rate", type=float, default=0.10)
    return parser.parse_args()


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    for column in (args.smiles_col, args.valid_col, args.kras_valid_col, args.kd_col):
        if column not in df.columns:
            raise KeyError(f"Missing {column} in {args.input_csv}; columns={list(df.columns)}")

    valid_flags = df[args.valid_col].astype(bool).to_numpy()
    kras_valid_flags = df[args.kras_valid_col].astype(bool).to_numpy()
    usable = df.loc[valid_flags & kras_valid_flags].copy()
    usable[args.kd_col] = pd.to_numeric(usable[args.kd_col], errors="coerce")
    usable = usable[np.isfinite(usable[args.kd_col])].copy()
    canonical = [canonicalize(smiles) for smiles in usable[args.smiles_col].astype(str).tolist()]
    canonical = [smiles for smiles in canonical if smiles is not None]
    kd = usable[args.kd_col].to_numpy(dtype=float)

    summary = {
        "input_csv": str(args.input_csv),
        "n": int(len(df)),
        "valid_rate": float(valid_flags.mean()) if len(valid_flags) else 0.0,
        "kras_valid_rate": float(kras_valid_flags.mean()) if len(kras_valid_flags) else 0.0,
        "usable_n": int(len(usable)),
        "unique_rate_valid": float(len(set(canonical)) / len(canonical)) if canonical else 0.0,
        "mean_kras_kd_pred": float(kd.mean()) if kd.size else float("nan"),
        "median_kras_kd_pred": float(np.median(kd)) if kd.size else float("nan"),
        "p10_kras_kd_pred": float(np.percentile(kd, 10)) if kd.size else float("nan"),
        "p25_kras_kd_pred": float(np.percentile(kd, 25)) if kd.size else float("nan"),
        "p75_kras_kd_pred": float(np.percentile(kd, 75)) if kd.size else float("nan"),
        "kras_kd_le_1p25_rate": float((kd <= 1.25).mean()) if kd.size else float("nan"),
        "kras_kd_le_1p0_rate": float((kd <= 1.0).mean()) if kd.size else float("nan"),
        "kras_kd_le_0p8_rate": float((kd <= 0.8).mean()) if kd.size else float("nan"),
    }
    summary["passes_ppo_gate"] = bool(
        summary["valid_rate"] >= args.min_valid_rate
        and summary["unique_rate_valid"] >= args.min_unique_rate
        and summary["kras_kd_le_1p0_rate"] >= args.min_kd_le_1p0_rate
        and summary["kras_kd_le_0p8_rate"] >= args.min_kd_le_0p8_rate
    )
    summary["gate_thresholds"] = {
        "min_valid_rate": args.min_valid_rate,
        "min_unique_rate": args.min_unique_rate,
        "min_kd_le_1p0_rate": args.min_kd_le_1p0_rate,
        "min_kd_le_0p8_rate": args.min_kd_le_0p8_rate,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
