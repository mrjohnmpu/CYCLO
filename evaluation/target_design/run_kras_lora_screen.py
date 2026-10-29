from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.target_design.kras_scorer import KRASSmilesScorer


RDLogger.DisableLog("rdApp.*")

DEFAULT_RUN_ROOT = Path('results/kras_design/20260521_lora_n30000_screen_v1')
DEFAULT_VALIDATION_CSV = Path(
    'results/property_annotation_runs/20260429_115158_restored_validation_gpu0/gpu0_outputs/restored_validation.with_properties.with_smiles2peptide.csv'
)
DEFAULT_CANDIDATE_FILES = {
    "head_to_tail": Path(
        'results/property_annotation_runs/20260423_115024_lora_cyclic_n30000_gpu0/outputs/head_to_tail_n30000.with_properties.with_smiles2peptide.csv'
    ),
    "sc_to_tail": Path(
        'results/property_annotation_runs/20260423_115024_lora_cyclic_n30000_gpu0/outputs/sc_to_tail_n30000.with_properties.with_smiles2peptide.csv'
    ),
    "disulfide_bridge": Path(
        'results/property_annotation_runs/20260423_115024_lora_cyclic_n30000_gpu0/outputs/disulfide_bridge_n30000.with_properties.with_smiles2peptide.csv'
    ),
}

TOP_KS = (1000, 100, 20)
KRAS_TOP_FRAC = 0.05
PERM_THRESHOLD = -6.0
HEMO_THRESHOLD = 0.15
TOX_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KRAS best-of-N screen over existing LoRA candidate pools.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional per-candidate-set row cap for smoke tests.")
    parser.add_argument("--calibration-sample-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dirs(run_root: Path) -> dict[str, Path]:
    paths = {
        "calibration": run_root / "calibration",
        "screened": run_root / "screened_candidates",
        "summary": run_root / "summary",
        "report": run_root / "report",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_csv(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, nrows=max_rows)


def add_kras_scores(df: pd.DataFrame, scorer: KRASSmilesScorer, smiles_col: str = "SMILES") -> pd.DataFrame:
    if smiles_col not in df.columns:
        raise KeyError(f"Missing {smiles_col} in columns={list(df.columns)}")
    score_df = scorer.score_smiles(df[smiles_col].tolist())
    return pd.concat([df.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)


def summarize_numeric(df: pd.DataFrame, dataset: str, group: str) -> dict[str, object]:
    row: dict[str, object] = {"dataset": dataset, "group": group, "n": int(len(df))}
    columns = [
        "kras_kd_pred",
        "kras_score",
        "PAMPA_value",
        "hemolysis_score",
        "toxicity_score",
        "peptide_length",
        "frac_natural_resolved",
        "frac_non_natural_resolved",
    ]
    for col in columns:
        if col in df.columns and len(df):
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{col}_p90"] = float(vals.quantile(0.90)) if len(vals) else np.nan
            row[f"{col}_p95"] = float(vals.quantile(0.95)) if len(vals) else np.nan
    if "kras_valid" in df.columns:
        row["kras_valid_rate"] = float(df["kras_valid"].astype(bool).mean()) if len(df) else 0.0
    return row


def murcko(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


def diversity_summary(df: pd.DataFrame, dataset: str, group: str) -> dict[str, object]:
    row: dict[str, object] = {"dataset": dataset, "group": group, "n": int(len(df))}
    if len(df) == 0 or "SMILES" not in df.columns:
        return row
    smiles = df["SMILES"].dropna().astype(str)
    row["unique_smiles"] = int(smiles.nunique())
    row["unique_smiles_rate"] = float(smiles.nunique() / len(df)) if len(df) else 0.0
    scaffolds = smiles.map(murcko)
    nonempty = scaffolds[scaffolds.ne("")]
    row["unique_murcko"] = int(nonempty.nunique())
    row["murcko_per_molecule"] = float(nonempty.nunique() / len(df)) if len(df) else 0.0
    if len(nonempty):
        counts = nonempty.value_counts()
        row["top1_murcko_fraction"] = float(counts.iloc[0] / len(df))
        row["top5_murcko_fraction"] = float(counts.head(5).sum() / len(df))
        row["top10_murcko_fraction"] = float(counts.head(10).sum() / len(df))
    return row


def sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ("kras_score", "PAMPA_value", "hemolysis_score", "toxicity_score"):
        if col not in work.columns:
            work[col] = np.nan
    return work.sort_values(
        ["kras_score", "PAMPA_value", "hemolysis_score", "toxicity_score"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def write_top_files(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ranked_candidates.csv", index=False)
    for top_k in TOP_KS:
        df.head(top_k).to_csv(out_dir / f"top_{top_k}.csv", index=False)


def filter_default(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    cutoff = df["kras_score"].quantile(1.0 - KRAS_TOP_FRAC)
    mask = (
        df["kras_valid"].astype(bool)
        & (df["kras_score"] >= cutoff)
        & (pd.to_numeric(df.get("PAMPA_value"), errors="coerce") >= PERM_THRESHOLD)
        & (pd.to_numeric(df.get("hemolysis_score"), errors="coerce") <= HEMO_THRESHOLD)
        & (pd.to_numeric(df.get("toxicity_score"), errors="coerce") <= TOX_THRESHOLD)
    )
    return df[mask].copy().reset_index(drop=True)


def scaffold_aware_shortlist(df: pd.DataFrame, max_per_scaffold: int = 2, limit: int = 50) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    work = df.copy()
    work["murcko_scaffold"] = work["SMILES"].map(murcko)
    kept = []
    counts: dict[str, int] = {}
    for _, row in work.iterrows():
        scaf = row.get("murcko_scaffold", "") or ""
        count = counts.get(scaf, 0)
        if count >= max_per_scaffold:
            continue
        kept.append(row)
        counts[scaf] = count + 1
        if len(kept) >= limit:
            break
    return pd.DataFrame(kept).reset_index(drop=True) if kept else work.head(0)


def run_calibration(args: argparse.Namespace, paths: dict[str, Path], scorer: KRASSmilesScorer) -> None:
    rows = []
    val = load_csv(args.validation_csv)
    val_sample = val.sample(n=min(args.calibration_sample_n, len(val)), random_state=args.seed).reset_index(drop=True)
    val_scored = add_kras_scores(val_sample, scorer)
    val_scored.to_csv(paths["calibration"] / "validation_sample_kras.csv", index=False)
    rows.append(summarize_numeric(val_scored, "validation", f"sample_{len(val_scored)}"))

    lora_samples = []
    for name, path in DEFAULT_CANDIDATE_FILES.items():
        df = load_csv(path)
        sample = df.sample(n=min(args.calibration_sample_n, len(df)), random_state=args.seed).reset_index(drop=True)
        scored = add_kras_scores(sample, scorer)
        scored.insert(0, "source_type", name)
        scored.to_csv(paths["calibration"] / f"lora_{name}_sample_kras.csv", index=False)
        rows.append(summarize_numeric(scored, name, f"sample_{len(scored)}"))
        lora_samples.append(scored)

    pd.concat(lora_samples, ignore_index=True).to_csv(paths["calibration"] / "lora_all_types_sample_kras.csv", index=False)
    pd.DataFrame(rows).to_csv(paths["calibration"] / "calibration_summary.csv", index=False)


def run_full_screen(args: argparse.Namespace, paths: dict[str, Path], scorer: KRASSmilesScorer):
    summary_rows = []
    diversity_rows = []
    shortlist_paths = []
    for name, path in DEFAULT_CANDIDATE_FILES.items():
        print(f"[{datetime.now().isoformat(timespec='seconds')}] processing {name}", flush=True)
        df = load_csv(path, max_rows=args.max_rows)
        scored = add_kras_scores(df, scorer)
        ranked = sort_candidates(scored)
        out_dir = paths["screened"] / name
        write_top_files(ranked, out_dir)

        default_filtered = sort_candidates(filter_default(ranked))
        default_filtered.to_csv(out_dir / "filtered_default.csv", index=False)
        shortlist_source = default_filtered if len(default_filtered) else ranked
        shortlist = scaffold_aware_shortlist(shortlist_source, max_per_scaffold=2, limit=50)
        shortlist.to_csv(out_dir / "shortlist_scaffold_aware_top50.csv", index=False)
        shortlist_paths.append(str(out_dir / "shortlist_scaffold_aware_top50.csv"))

        for group, frame in [
            ("all_ranked", ranked),
            ("filtered_default", default_filtered),
            ("top_1000", ranked.head(1000)),
            ("top_100", ranked.head(100)),
            ("top_20", ranked.head(20)),
        ]:
            summary_rows.append(summarize_numeric(frame, name, group))

        # Murcko scaffolds are expensive for large cyclic peptides; compute them only
        # for selection-relevant subsets while keeping full ranked CSVs intact.
        for group, frame in [
            ("filtered_default", default_filtered),
            ("top_1000", ranked.head(1000)),
            ("top_100", ranked.head(100)),
            ("shortlist_top50", shortlist),
        ]:
            diversity_rows.append(diversity_summary(frame, name, group))
        print(f"[{datetime.now().isoformat(timespec='seconds')}] finished {name}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    diversity_df = pd.DataFrame(diversity_rows)
    summary_df.to_csv(paths["summary"] / "kras_screen_summary.csv", index=False)
    diversity_df.to_csv(paths["summary"] / "kras_diversity_summary.csv", index=False)
    return summary_df, diversity_df, shortlist_paths


def write_manifest(args: argparse.Namespace, paths: dict[str, Path], shortlist_paths: list[str]) -> None:
    manifest = {
        "run_name": args.run_root.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "run_root": str(args.run_root),
        "max_rows": args.max_rows,
        "kras_model": './data/kras_kd/kras_xgboost_reg.pkl',
        "scorer": './evaluation/target_design/kras_scorer.py',
        "validation_csv": str(args.validation_csv),
        "candidate_files": {key: str(value) for key, value in DEFAULT_CANDIDATE_FILES.items()},
        "reused_existing_lora_generation": True,
        "reused_existing_peptiverse_properties": True,
        "reused_existing_smiles2peptide_columns": True,
        "thresholds": {
            "kras_top_fraction": KRAS_TOP_FRAC,
            "PAMPA_value_min": PERM_THRESHOLD,
            "hemolysis_score_max": HEMO_THRESHOLD,
            "toxicity_score_max": TOX_THRESHOLD,
        },
        "outputs": {
            "calibration_summary": str(paths["calibration"] / "calibration_summary.csv"),
            "screen_summary": str(paths["summary"] / "kras_screen_summary.csv"),
            "diversity_summary": str(paths["summary"] / "kras_diversity_summary.csv"),
            "shortlists": shortlist_paths,
        },
    }
    (args.run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")


def dataframe_csv_block(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "_No rows._"
    return "```csv\n" + df.to_csv(index=False).strip() + "\n```"


def write_report(paths: dict[str, Path], summary_df: pd.DataFrame, diversity_df: pd.DataFrame) -> None:
    lines = [
        "# KRAS LoRA Candidate Screen Summary",
        "",
        "## Scope",
        "- Reused existing LoRA n=30000 candidates for head_to_tail, sc_to_tail, and disulfide_bridge.",
        "- Reused existing PeptiVerse property and SMILES2PEPTIDE annotations.",
        "- Added KRAS XGBoost scores and ranked candidates by KRAS score with ADMET tie-breakers.",
        "",
        "## Key Outputs",
        f"- Calibration summary: {paths['calibration'] / 'calibration_summary.csv'}",
        f"- Screen summary: {paths['summary'] / 'kras_screen_summary.csv'}",
        f"- Diversity summary: {paths['summary'] / 'kras_diversity_summary.csv'}",
        "",
        "## Top-20 Summary",
    ]
    cols = [
        "dataset",
        "group",
        "n",
        "kras_score_mean",
        "kras_score_median",
        "kras_kd_pred_mean",
        "PAMPA_value_mean",
        "hemolysis_score_mean",
        "toxicity_score_mean",
        "peptide_length_mean",
    ]
    top20 = summary_df[summary_df["group"].eq("top_20")]
    lines.append(dataframe_csv_block(top20[[col for col in cols if col in top20.columns]]))
    lines.extend(["", "## Diversity Snapshot"])
    dcols = ["dataset", "group", "n", "unique_smiles", "unique_murcko", "top1_murcko_fraction", "top5_murcko_fraction"]
    dtop = diversity_df[diversity_df["group"].isin(["top_100", "shortlist_top50"])]
    lines.append(dataframe_csv_block(dtop[[col for col in dcols if col in dtop.columns]]))
    lines.extend(
        [
            "",
            "## KRAS-RL Recommendation Rule",
            "- Enter KRAS-RL only if a cyclization type has strong top KRAS scores without severe ADMET degradation or scaffold collapse.",
            "- This run does not start RL automatically.",
        ]
    )
    (paths["report"] / "kras_lora_screen_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    paths = ensure_dirs(args.run_root)
    scorer = KRASSmilesScorer()
    run_calibration(args, paths, scorer)
    summary_df, diversity_df, shortlist_paths = run_full_screen(args, paths, scorer)
    write_manifest(args, paths, shortlist_paths)
    write_report(paths, summary_df, diversity_df)
    print(f"DONE {args.run_root}")


if __name__ == "__main__":
    main()
