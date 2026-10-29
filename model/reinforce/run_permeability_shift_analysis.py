#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics.core import canonicalize_smiles as canonicalize_smiles_core
from evaluation.property_pipeline.mol_cache import smiles_to_mol


DEFAULT_VALID_CSV = Path(
    'results/property_annotation_runs/20260429_115158_restored_validation_gpu0/gpu0_outputs/restored_validation.with_properties.with_smiles2peptide.csv'
)
DEFAULT_PERMEABILITY_COL = "PAMPA_value"
DEFAULT_HEMO_COL = "hemolysis_score"
DEFAULT_SEQUENCE_COL = "three_letter_sequence"
DEFAULT_SMILES_COL = "SMILES"
DEFAULT_TOPOLOGY = "head_to_tail"


DESCRIPTOR_COLUMNS_DIRECT = [
    "MW",
    "LogP",
    "HBD",
    "HBA",
    "TPSA",
    "RotBonds",
    "peptide_length",
    "frac_natural_resolved",
    "frac_non_natural_resolved",
    "n_natural_resolved",
    "n_non_natural_resolved",
]

DIRECT_SEQUENCE_DESCRIPTOR_COLUMNS = [
    "peptide_length",
    "frac_natural_resolved",
    "frac_non_natural_resolved",
    "n_natural_resolved",
    "n_non_natural_resolved",
]
MOTIF_MIN_AFTER_ONLY_COUNT = 20

PERMEABILITY_TOP_FRACTIONS = [0.01, 0.05, 0.10, 0.20]
REQUIRED_DESCRIPTOR_COLUMNS = [
    "MolWt",
    "HeavyAtomCount",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "AromaticRingCount",
    "FractionCSP3",
    "FormalCharge",
    "NumAmideBonds",
    "NumHeteroAtoms",
    "N_count",
    "O_count",
    "S_count",
    "C_count",
    "N_fraction",
    "O_fraction",
    "S_fraction",
    "C_fraction",
]
SELECTED_DESCRIPTOR_COLUMNS = [
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "MolWt",
    "peptide_length",
    "n_methyl_residue_fraction",
    "hydrophobic_residue_fraction",
    "charged_residue_fraction",
]

HYDROPHOBIC_RESIDUES = {
    "Ala",
    "Val",
    "Leu",
    "Ile",
    "Met",
    "Phe",
    "Trp",
    "Pro",
}
POLAR_RESIDUES = {
    "Ser",
    "Thr",
    "Asn",
    "Gln",
    "Cys",
    "Tyr",
}
CHARGED_RESIDUES = {
    "Asp",
    "Glu",
    "Lys",
    "Arg",
    "His",
}
AROMATIC_RESIDUES = {
    "Phe",
    "Tyr",
    "Trp",
    "His",
}


@dataclass
class InputSpec:
    stage: str
    csv_path: Path


@dataclass
class AnalysisContext:
    args: argparse.Namespace
    output_root: Path
    tables_dir: Path
    figures_dir: Path
    report_dir: Path
    metadata_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze permeability shifts before and after RL.")
    parser.add_argument("--before_csv", type=Path, required=True)
    parser.add_argument("--after_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--smiles_col", type=str, default=DEFAULT_SMILES_COL)
    parser.add_argument("--permeability_col", type=str, default=DEFAULT_PERMEABILITY_COL)
    parser.add_argument("--hemo_col", type=str, default=DEFAULT_HEMO_COL)
    parser.add_argument("--sequence_col", type=str, default=DEFAULT_SEQUENCE_COL)
    parser.add_argument("--valid_csv", type=Path, default=DEFAULT_VALID_CSV)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--top_fraction", type=float, default=0.10)
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--fp_bits", type=int, default=2048)
    parser.add_argument("--topology", type=str, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--smoke_rows", type=int, default=None)
    parser.add_argument("--targets", type=str, default="all")
    parser.add_argument("--skip_report", action="store_true")
    parser.add_argument("--skip_tables", action="store_true")
    return parser.parse_args()


def ensure_layout(output_root: Path) -> AnalysisContext:
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    report_dir = output_root / "report"
    metadata_dir = output_root / "metadata"
    for path in [output_root, tables_dir, figures_dir, report_dir, metadata_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return AnalysisContext(
        args=None,  # type: ignore[arg-type]
        output_root=output_root,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        report_dir=report_dir,
        metadata_dir=metadata_dir,
    )


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def parse_targets(targets_arg: str) -> set[str]:
    if not targets_arg or targets_arg.strip().lower() == "all":
        return {"all"}
    return {item.strip() for item in targets_arg.split(",") if item.strip()}


def should_run_target(selected_targets: set[str], target_name: str, experiment_name: str) -> bool:
    if "all" in selected_targets:
        return True
    return target_name in selected_targets or experiment_name in selected_targets


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def autodetect_column(dataframe: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    lower_map = {column.lower(): column for column in dataframe.columns}
    for candidate in candidates:
        column = lower_map.get(candidate.lower())
        if column:
            return column
    raise ValueError(f"Unable to identify required column '{label}' from columns {list(dataframe.columns)}")


def detect_sequence_column(dataframe: pd.DataFrame, explicit: Optional[str]) -> str | None:
    if explicit:
        if explicit in dataframe.columns:
            return explicit
        lower_map = {column.lower(): column for column in dataframe.columns}
        found = lower_map.get(explicit.lower())
        if found:
            return found
    for candidate in ["three_letter_sequence", "Sequence", "sequence"]:
        lower_map = {column.lower(): column for column in dataframe.columns}
        found = lower_map.get(candidate.lower())
        if found:
            return found
    return None


def canonicalize_smiles(smiles: str) -> str | None:
    return canonicalize_smiles_core(smiles)


def is_valid_smiles(smiles: str) -> bool:
    return canonicalize_smiles(smiles) is not None


def load_stage_csv(
    spec: InputSpec,
    *,
    smiles_col: str,
    permeability_col: str,
    hemo_col: str,
    sequence_col: str | None,
    smoke_rows: int | None,
    topology: str,
) -> pd.DataFrame:
    dataframe = pd.read_csv(spec.csv_path)
    if smoke_rows is not None and smoke_rows > 0:
        dataframe = dataframe.head(int(smoke_rows)).copy()

    smiles_column = autodetect_column(dataframe, [smiles_col, "SMILES", "Smiles", "smiles"], "smiles")
    permeability_column = autodetect_column(
        dataframe,
        [permeability_col, "PAMPA_value", "permeability", "Permeability"],
        "permeability",
    )
    hemo_column = autodetect_column(
        dataframe,
        [hemo_col, "hemolysis_score", "HemolysisScore"],
        "hemolysis",
    )
    detected_sequence_col = detect_sequence_column(dataframe, sequence_col)

    standardized = dataframe.copy()
    standardized["stage"] = spec.stage
    standardized["SMILES"] = standardized[smiles_column].astype(str).str.strip()
    standardized["permeability"] = pd.to_numeric(standardized[permeability_column], errors="coerce")
    standardized["hemolysis_score"] = pd.to_numeric(standardized[hemo_column], errors="coerce")
    standardized["non_hemolytic_score"] = 1.0 - standardized["hemolysis_score"]
    standardized["permeability_missing"] = standardized["permeability"].isna()
    if "Cyclization" not in standardized.columns:
        standardized["Cyclization"] = topology
    standardized["Cyclization"] = standardized["Cyclization"].fillna(topology).astype(str)
    if detected_sequence_col is not None:
        standardized["three_letter_sequence"] = standardized[detected_sequence_col].fillna("").astype(str)
    else:
        standardized["three_letter_sequence"] = ""
    standardized["canonical_smiles_existing"] = standardized.get("canonical_smiles")

    for column in DIRECT_SEQUENCE_DESCRIPTOR_COLUMNS:
        if column in standardized.columns:
            standardized[column] = pd.to_numeric(standardized[column], errors="coerce")

    for column in ["MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds"]:
        if column in standardized.columns:
            standardized[column] = pd.to_numeric(standardized[column], errors="coerce")

    canonical_existing = standardized.get("canonical_smiles")
    if canonical_existing is not None:
        canonical_existing = canonical_existing.where(canonical_existing.notna(), "").astype(str).str.strip()
        canonical_existing = canonical_existing.where(canonical_existing != "", None)
    recomputed_canonical = standardized["SMILES"].map(canonicalize_smiles)
    standardized["canonical_smiles"] = (
        canonical_existing.where(canonical_existing.notna(), recomputed_canonical)
        if canonical_existing is not None
        else recomputed_canonical
    )
    standardized["is_valid_smiles"] = standardized["canonical_smiles"].notna()
    standardized["dropped_invalid_permeability"] = standardized["permeability"].isna()
    return standardized


def load_validation_reference(path: Path, *, smiles_col: str, topology: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    smiles_column = autodetect_column(dataframe, [smiles_col, "SMILES"], "validation smiles")
    if "Cyclization" in dataframe.columns:
        dataframe = dataframe[dataframe["Cyclization"].astype(str) == topology].copy()
    dataframe["SMILES"] = dataframe[smiles_column].astype(str).str.strip()
    dataframe["canonical_smiles"] = dataframe["SMILES"].map(canonicalize_smiles)
    dataframe = dataframe[dataframe["canonical_smiles"].notna()].copy()
    dataframe = dataframe.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    return dataframe


def bootstrap_statistic(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    fn,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values_a.size == 0 or values_b.size == 0:
        return (math.nan, math.nan)
    samples = []
    for _ in range(int(n_bootstrap)):
        sample_a = rng.choice(values_a, size=values_a.size, replace=True)
        sample_b = rng.choice(values_b, size=values_b.size, replace=True)
        samples.append(float(fn(sample_b) - fn(sample_a)))
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def cliffs_delta(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if values_a.size == 0 or values_b.size == 0:
        return math.nan
    gt = 0
    lt = 0
    for value_b in values_b:
        gt += np.sum(value_b > values_a)
        lt += np.sum(value_b < values_a)
    return float((gt - lt) / (values_a.size * values_b.size))


def cohens_d(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if values_a.size < 2 or values_b.size < 2:
        return math.nan
    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    var_a = np.var(values_a, ddof=1)
    var_b = np.var(values_b, ddof=1)
    pooled = ((values_a.size - 1) * var_a + (values_b.size - 1) * var_b) / (values_a.size + values_b.size - 2)
    if pooled <= 0:
        return math.nan
    return float((mean_b - mean_a) / math.sqrt(pooled))


def gini_coefficient(counts: np.ndarray) -> float:
    if counts.size == 0:
        return math.nan
    values = np.sort(counts.astype(float))
    if np.allclose(values.sum(), 0.0):
        return 0.0
    index = np.arange(1, values.size + 1)
    return float((np.sum((2 * index - values.size - 1) * values)) / (values.size * np.sum(values)))


def scaffold_entropy(counts: np.ndarray) -> float:
    if counts.size == 0:
        return math.nan
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def murcko_scaffold_smiles(smiles: str) -> tuple[str, str]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return "", ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return "", ""
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return (
        Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True),
        Chem.MolToSmiles(generic, canonical=True, isomericSmiles=True) if generic.GetNumAtoms() else "",
    )


def morgan_bitvect(smiles: str, *, radius: int, fp_bits: int):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_bits)
    return generator.GetFingerprint(mol)


def tanimoto_similarity(fp_a, fp_b) -> float:
    if fp_a is None or fp_b is None:
        return math.nan
    from rdkit.DataStructs import TanimotoSimilarity

    return float(TanimotoSimilarity(fp_a, fp_b))


def summarize_distribution(values: np.ndarray, *, stage: str, subset: str) -> dict[str, Any]:
    if values.size == 0:
        return {"stage": stage, "subset": subset, "n": 0}
    quantiles = np.quantile(values, [0.05, 0.25, 0.75, 0.95])
    top1 = max(1, int(math.ceil(values.size * 0.01)))
    top5 = max(1, int(math.ceil(values.size * 0.05)))
    top10 = max(1, int(math.ceil(values.size * 0.10)))
    sorted_values = np.sort(values)
    return {
        "stage": stage,
        "subset": subset,
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q75": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "iqr": float(quantiles[2] - quantiles[1]),
        "top_1_percent_mean": float(np.mean(sorted_values[-top1:])),
        "top_5_percent_mean": float(np.mean(sorted_values[-top5:])),
        "top_10_percent_mean": float(np.mean(sorted_values[-top10:])),
    }


def make_ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, values.size + 1) / values.size
    return x, y


def detect_residue_class(token: str) -> str | None:
    cleaned = token.strip()
    if not cleaned:
        return None
    base = (
        cleaned.replace("(N-Me)", "")
        .replace("(NMe)", "")
        .replace("N-Me-", "")
        .replace("NMe-", "")
        .replace("N-methyl-", "")
    )
    base = base.replace("(", "").replace(")", "")
    title = base[:1].upper() + base[1:].lower() if base else ""
    if title in HYDROPHOBIC_RESIDUES:
        return "hydrophobic"
    if title in POLAR_RESIDUES:
        return "polar"
    if title in CHARGED_RESIDUES:
        return "charged"
    if title in AROMATIC_RESIDUES:
        return "aromatic"
    return None


def tokenize_sequence(sequence: str) -> list[str]:
    text = sequence.strip()
    if not text:
        return []
    if text.startswith("cyclo(") and text.endswith(")"):
        text = text[6:-1]
    tokens: list[str] = []
    buffer: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
            buffer.append(char)
        elif char == ")":
            depth = max(0, depth - 1)
            buffer.append(char)
        elif char == "-" and depth == 0:
            token = "".join(buffer).strip()
            if token:
                tokens.append(token)
            buffer = []
        else:
            buffer.append(char)
    token = "".join(buffer).strip()
    if token:
        tokens.append(token)
    return tokens


def is_n_methyl_token(token: str) -> bool:
    lowered = token.lower()
    return "n-me" in lowered or "nme" in lowered or "n-methyl" in lowered


def normalize_token(token: str) -> str:
    return token.strip()


def derive_sequence_descriptors(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    hydrophobic = []
    polar = []
    charged = []
    aromatic = []
    n_methyl_count = []
    n_methyl_fraction = []
    for sequence in frame["three_letter_sequence"].fillna("").astype(str):
        tokens = tokenize_sequence(sequence)
        token_count = len(tokens)
        counts = {"hydrophobic": 0, "polar": 0, "charged": 0, "aromatic": 0}
        methyls = 0
        for token in tokens:
            residue_class = detect_residue_class(token)
            if residue_class is not None:
                counts[residue_class] += 1
            if is_n_methyl_token(token):
                methyls += 1
        hydrophobic.append(counts["hydrophobic"] / token_count if token_count else math.nan)
        polar.append(counts["polar"] / token_count if token_count else math.nan)
        charged.append(counts["charged"] / token_count if token_count else math.nan)
        aromatic.append(counts["aromatic"] / token_count if token_count else math.nan)
        n_methyl_count.append(methyls if token_count else math.nan)
        n_methyl_fraction.append(methyls / token_count if token_count else math.nan)
    frame["hydrophobic_residue_fraction"] = hydrophobic
    frame["polar_residue_fraction"] = polar
    frame["charged_residue_fraction"] = charged
    frame["aromatic_residue_fraction"] = aromatic
    frame["n_methyl_residue_count"] = n_methyl_count
    frame["n_methyl_residue_fraction"] = n_methyl_fraction
    return frame


def compute_additional_rdkit_descriptors(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    rows: list[dict[str, Any]] = []
    for smiles in frame["SMILES"].astype(str):
        mol = smiles_to_mol(smiles)
        if mol is None:
            rows.append({column: math.nan for column in REQUIRED_DESCRIPTOR_COLUMNS})
            continue
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atom_count = len(atom_symbols)
        n_count = atom_symbols.count("N")
        o_count = atom_symbols.count("O")
        s_count = atom_symbols.count("S")
        c_count = atom_symbols.count("C")
        rows.append(
            {
                "MolWt": float(rdMolDescriptors.CalcExactMolWt(mol)),
                "HeavyAtomCount": int(rdMolDescriptors.CalcNumHeavyAtoms(mol)),
                "MolLogP": float(Crippen.MolLogP(mol)),
                "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
                "NumHDonors": int(Lipinski.NumHDonors(mol)),
                "NumHAcceptors": int(Lipinski.NumHAcceptors(mol)),
                "NumRotatableBonds": int(Lipinski.NumRotatableBonds(mol)),
                "RingCount": int(rdMolDescriptors.CalcNumRings(mol)),
                "AromaticRingCount": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
                "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
                "FormalCharge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
                "NumAmideBonds": int(rdMolDescriptors.CalcNumAmideBonds(mol)),
                "NumHeteroAtoms": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
                "N_count": int(n_count),
                "O_count": int(o_count),
                "S_count": int(s_count),
                "C_count": int(c_count),
                "N_fraction": float(n_count / atom_count) if atom_count else math.nan,
                "O_fraction": float(o_count / atom_count) if atom_count else math.nan,
                "S_fraction": float(s_count / atom_count) if atom_count else math.nan,
                "C_fraction": float(c_count / atom_count) if atom_count else math.nan,
            }
        )
    descriptor_df = pd.DataFrame(rows)
    alias_sources = {
        "MolWt": "MW",
        "MolLogP": "LogP",
        "NumHDonors": "HBD",
        "NumHAcceptors": "HBA",
        "NumRotatableBonds": "RotBonds",
    }
    for target_col, source_col in alias_sources.items():
        if source_col in frame.columns and target_col not in frame.columns:
            frame[target_col] = pd.to_numeric(frame[source_col], errors="coerce")
    for column in descriptor_df.columns:
        if column not in frame.columns:
            frame[column] = descriptor_df[column].reset_index(drop=True)
        elif frame[column].isna().any():
            fallback = descriptor_df[column].reset_index(drop=True)
            frame[column] = frame[column].where(frame[column].notna(), fallback)
    return frame


def prepare_analysis_dataframe(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    before_df = load_stage_csv(
        InputSpec(stage="before", csv_path=args.before_csv),
        smiles_col=args.smiles_col,
        permeability_col=args.permeability_col,
        hemo_col=args.hemo_col,
        sequence_col=args.sequence_col,
        smoke_rows=args.smoke_rows,
        topology=args.topology,
    )
    after_df = load_stage_csv(
        InputSpec(stage="after", csv_path=args.after_csv),
        smiles_col=args.smiles_col,
        permeability_col=args.permeability_col,
        hemo_col=args.hemo_col,
        sequence_col=args.sequence_col,
        smoke_rows=args.smoke_rows,
        topology=args.topology,
    )
    combined = pd.concat([before_df, after_df], ignore_index=True)
    combined = derive_sequence_descriptors(combined)
    combined = compute_additional_rdkit_descriptors(combined)
    validation_df = load_validation_reference(args.valid_csv, smiles_col=args.smiles_col, topology=args.topology)

    metadata = {
        "reused_columns": [
            column for column in DESCRIPTOR_COLUMNS_DIRECT if column in combined.columns
        ]
        + [column for column in ["PAMPA_value", "hemolysis_score", "canonical_smiles", "three_letter_sequence"] if column in combined.columns],
        "computed_columns": [
            "is_valid_smiles",
            "canonical_smiles",
            "non_hemolytic_score",
            "MolWt",
            "HeavyAtomCount",
            "MolLogP",
            "NumHDonors",
            "NumHAcceptors",
            "RingCount",
            "AromaticRingCount",
            "FractionCSP3",
            "FormalCharge",
            "NumAmideBonds",
            "NumHeteroAtoms",
            "N_count",
            "O_count",
            "S_count",
            "C_count",
            "N_fraction",
            "O_fraction",
            "S_fraction",
            "C_fraction",
            "hydrophobic_residue_fraction",
            "polar_residue_fraction",
            "charged_residue_fraction",
            "aromatic_residue_fraction",
            "n_methyl_residue_count",
            "n_methyl_residue_fraction",
        ],
    }
    return combined, validation_df, metadata


def maybe_write_csv(dataframe: pd.DataFrame, path: Path, *, enabled: bool) -> None:
    if enabled:
        write_csv(dataframe, path)


def plot_filled_kde_pair(
    before_values: np.ndarray,
    after_values: np.ndarray,
    *,
    xlabel: str,
    output_stem: Path,
) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    series = [
        (before_values, "Before", "#B3B3B3", "#6E6E6E"),
        (after_values, "After", "#0072B2", "#004C78"),
    ]
    for values, label, fill_color, line_color in series:
        if values.size < 2:
            continue
        grid = np.linspace(values.min(), values.max(), 256)
        kde = stats.gaussian_kde(values)
        density = kde(grid)
        ax.fill_between(grid, density, color=fill_color, alpha=0.35, label=label)
        ax.plot(grid, density, color=line_color, linewidth=2.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, output_stem)


def compute_token_enrichment(analysis_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    token_counters: dict[str, dict[str, Any]] = {}
    for stage in ["before", "after"]:
        stage_df = analysis_df[analysis_df["stage"] == stage].copy()
        token_counter: dict[str, int] = {}
        total_tokens = 0
        sequences_with_n_methyl = 0
        n_methyl_tokens = 0
        for sequence in stage_df["three_letter_sequence"].fillna("").astype(str):
            tokens = [normalize_token(token) for token in tokenize_sequence(sequence)]
            has_n_methyl = False
            for token in tokens:
                token_counter[token] = token_counter.get(token, 0) + 1
                total_tokens += 1
                if is_n_methyl_token(token):
                    n_methyl_tokens += 1
                    has_n_methyl = True
            if has_n_methyl:
                sequences_with_n_methyl += 1
        token_counters[stage] = {
            "counter": token_counter,
            "total_tokens": total_tokens,
            "num_sequences": len(stage_df),
            "sequences_with_n_methyl": sequences_with_n_methyl,
            "n_methyl_tokens": n_methyl_tokens,
        }
        summary_rows.append(
            {
                "stage": stage,
                "num_sequences": int(len(stage_df)),
                "total_tokens": int(total_tokens),
                "n_methyl_tokens": int(n_methyl_tokens),
                "n_methyl_token_fraction": float(n_methyl_tokens / total_tokens) if total_tokens else math.nan,
                "sequences_with_n_methyl": int(sequences_with_n_methyl),
                "sequence_fraction_with_n_methyl": (
                    float(sequences_with_n_methyl / len(stage_df)) if len(stage_df) else math.nan
                ),
            }
        )

    before_counter = token_counters["before"]["counter"]
    after_counter = token_counters["after"]["counter"]
    total_before = token_counters["before"]["total_tokens"]
    total_after = token_counters["after"]["total_tokens"]
    num_before = token_counters["before"]["num_sequences"]
    num_after = token_counters["after"]["num_sequences"]
    all_tokens = sorted(set(before_counter) | set(after_counter))
    for token in all_tokens:
        before_count = int(before_counter.get(token, 0))
        after_count = int(after_counter.get(token, 0))
        before_fraction = float(before_count / total_before) if total_before else math.nan
        after_fraction = float(after_count / total_after) if total_after else math.nan
        before_prevalence = float(before_count / num_before) if num_before else math.nan
        after_prevalence = float(after_count / num_after) if num_after else math.nan
        rows.append(
            {
                "token": token,
                "before_count": before_count,
                "after_count": after_count,
                "before_fraction": before_fraction,
                "after_fraction": after_fraction,
                "delta_fraction": after_fraction - before_fraction,
                "before_sequence_prevalence": before_prevalence,
                "after_sequence_prevalence": after_prevalence,
                "delta_sequence_prevalence": after_prevalence - before_prevalence,
                "is_n_methyl_token": is_n_methyl_token(token),
                "appears_only_after": before_count == 0 and after_count > 0,
                "highlight_after_only": before_count == 0 and after_count >= MOTIF_MIN_AFTER_ONLY_COUNT,
            }
        )
    token_df = pd.DataFrame(rows).sort_values(["delta_fraction", "after_count"], ascending=[False, False]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows)
    return token_df, summary_df


def experiment1_distribution_shift(
    analysis_df: pd.DataFrame,
    *,
    context: AnalysisContext,
    n_bootstrap: int,
    rng: np.random.Generator,
    selected_targets: set[str],
    write_tables: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    rows = []
    test_rows = []
    topk_rows = []

    subsets = {
        "all_with_permeability": analysis_df[analysis_df["permeability"].notna()].copy(),
        "valid_smiles_with_permeability": analysis_df[
            analysis_df["permeability"].notna() & analysis_df["is_valid_smiles"]
        ].copy(),
    }

    for subset_name, subset in subsets.items():
        before_values = subset[subset["stage"] == "before"]["permeability"].to_numpy(dtype=float)
        after_values = subset[subset["stage"] == "after"]["permeability"].to_numpy(dtype=float)
        rows.append(summarize_distribution(before_values, stage="before", subset=subset_name))
        rows.append(summarize_distribution(after_values, stage="after", subset=subset_name))

        if before_values.size and after_values.size:
            mean_ci = bootstrap_statistic(before_values, after_values, fn=np.mean, n_bootstrap=n_bootstrap, rng=rng)
            median_ci = bootstrap_statistic(before_values, after_values, fn=np.median, n_bootstrap=n_bootstrap, rng=rng)
            test_rows.append(
                {
                    "subset": subset_name,
                    "mean_difference": float(np.mean(after_values) - np.mean(before_values)),
                    "median_difference": float(np.median(after_values) - np.median(before_values)),
                    "relative_mean_change": float((np.mean(after_values) - np.mean(before_values)) / abs(np.mean(before_values)))
                    if not np.isclose(np.mean(before_values), 0.0)
                    else math.nan,
                    "mannwhitney_u_p": float(stats.mannwhitneyu(before_values, after_values, alternative="two-sided").pvalue),
                    "ks_p": float(stats.ks_2samp(before_values, after_values).pvalue),
                    "welch_t_p": float(stats.ttest_ind(before_values, after_values, equal_var=False).pvalue),
                    "wasserstein_distance": float(stats.wasserstein_distance(before_values, after_values)),
                    "cliffs_delta": cliffs_delta(before_values, after_values),
                    "cohens_d": cohens_d(before_values, after_values),
                    "bootstrap_mean_ci_low": mean_ci[0],
                    "bootstrap_mean_ci_high": mean_ci[1],
                    "bootstrap_median_ci_low": median_ci[0],
                    "bootstrap_median_ci_high": median_ci[1],
                }
            )
            before_sorted = np.sort(before_values)
            after_sorted = np.sort(after_values)
            for fraction in PERMEABILITY_TOP_FRACTIONS:
                before_k = max(1, int(math.ceil(before_sorted.size * fraction)))
                after_k = max(1, int(math.ceil(after_sorted.size * fraction)))
                before_top = before_sorted[-before_k:]
                after_top = after_sorted[-after_k:]
                topk_rows.append(
                    {
                        "subset": subset_name,
                        "top_fraction": fraction,
                        "before_top_k_mean": float(np.mean(before_top)),
                        "after_top_k_mean": float(np.mean(after_top)),
                        "delta": float(np.mean(after_top) - np.mean(before_top)),
                        "fold_change_if_mean_positive": (
                            float(np.mean(after_top) / np.mean(before_top))
                            if np.mean(before_top) > 0
                            else math.nan
                        ),
                        "after_fraction_above_before_median": float(np.mean(after_values > np.median(before_values))),
                        "after_fraction_above_before_q75": float(np.mean(after_values > np.quantile(before_values, 0.75))),
                        "after_fraction_above_before_q90": float(np.mean(after_values > np.quantile(before_values, 0.90))),
                        "after_fraction_above_before_q95": float(np.mean(after_values > np.quantile(before_values, 0.95))),
                    }
                )

    summary_df = pd.DataFrame(rows)
    tests_df = pd.DataFrame(test_rows)
    topk_df = pd.DataFrame(topk_rows)
    maybe_write_csv(summary_df, context.tables_dir / "experiment1_permeability_summary.csv", enabled=write_tables)
    maybe_write_csv(tests_df, context.tables_dir / "experiment1_distribution_tests.csv", enabled=write_tables)
    maybe_write_csv(topk_df, context.tables_dir / "experiment1_topk_summary.csv", enabled=write_tables)

    subset = subsets["all_with_permeability"]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    plot_data = [
        subset[subset["stage"] == "before"]["permeability"].to_numpy(dtype=float),
        subset[subset["stage"] == "after"]["permeability"].to_numpy(dtype=float),
    ]
    violin = ax.violinplot(plot_data, positions=[1, 2], showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violin["bodies"], ["#999999", "#0072B2"]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
    ax.boxplot(
        plot_data,
        positions=[1, 2],
        widths=0.18,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        medianprops={"color": "black"},
    )
    ax.set_xticks([1, 2], ["Before", "After"])
    ax.set_ylabel("PAMPA_value")
    if should_run_target(selected_targets, "experiment1_permeability_violin_box", "experiment1"):
        save_figure(fig, context.figures_dir / "experiment1_permeability_violin_box")
    else:
        plt.close(fig)

    if should_run_target(selected_targets, "experiment1_permeability_kde", "experiment1"):
        plot_filled_kde_pair(
            plot_data[0],
            plot_data[1],
            xlabel="PAMPA_value",
            output_stem=context.figures_dir / "experiment1_permeability_kde",
        )

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for values, label, color in zip(plot_data, ["Before", "After"], ["#666666", "#0072B2"]):
        x, y = make_ecdf(values)
        ax.step(x, y, where="post", color=color, linewidth=2, label=label)
    ax.legend(frameon=False)
    ax.set_xlabel("PAMPA_value")
    ax.set_ylabel("ECDF")
    if should_run_target(selected_targets, "experiment1_permeability_ecdf", "experiment1"):
        save_figure(fig, context.figures_dir / "experiment1_permeability_ecdf")
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for label, stage in [("Before", "before"), ("After", "after")]:
        stage_rows = topk_df[(topk_df["subset"] == "all_with_permeability")].copy()
        y_col = "before_top_k_mean" if stage == "before" else "after_top_k_mean"
        ax.plot(stage_rows["top_fraction"], stage_rows[y_col], marker="o", linewidth=2, label=label)
    ax.legend(frameon=False)
    ax.set_xlabel("Top fraction")
    ax.set_ylabel("Mean PAMPA_value")
    if should_run_target(selected_targets, "experiment1_topk_enrichment_curve", "experiment1"):
        save_figure(fig, context.figures_dir / "experiment1_topk_enrichment_curve")
    else:
        plt.close(fig)

    hemo_subset = analysis_df[analysis_df["non_hemolytic_score"].notna()].copy()
    hemo_before = hemo_subset[hemo_subset["stage"] == "before"]["non_hemolytic_score"].to_numpy(dtype=float)
    hemo_after = hemo_subset[hemo_subset["stage"] == "after"]["non_hemolytic_score"].to_numpy(dtype=float)
    if should_run_target(selected_targets, "experiment1_hemo_kde", "experiment1"):
        plot_filled_kde_pair(
            hemo_before,
            hemo_after,
            xlabel="Non-hemolytic score (1 - hemolysis_score)",
            output_stem=context.figures_dir / "experiment1_hemo_kde",
        )

    results["summary_df"] = summary_df
    results["tests_df"] = tests_df
    results["topk_df"] = topk_df
    return results


def experiment2_generation_quality(
    analysis_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    context: AnalysisContext,
    write_tables: bool,
) -> dict[str, Any]:
    rows = []
    scaffold_rows = []
    top_scaffold_rows = []
    reward_hacking_rows = []
    duplicate_rows = []
    token_enrichment_df, token_summary_df = compute_token_enrichment(analysis_df)

    validation_canonical = set(validation_df["canonical_smiles"].dropna().astype(str))

    valid_stage_frames: dict[str, pd.DataFrame] = {}
    for stage in ["before", "after"]:
        stage_df = analysis_df[analysis_df["stage"] == stage].copy()
        valid_df = stage_df[stage_df["is_valid_smiles"]].copy()
        valid_df["canonical_smiles"] = valid_df["canonical_smiles"].fillna("")
        valid_df = valid_df[valid_df["canonical_smiles"] != ""].copy()
        valid_stage_frames[stage] = valid_df

        total = len(stage_df)
        unique_canonical = valid_df["canonical_smiles"].nunique()
        duplicate_fraction = 1.0 - (unique_canonical / len(valid_df)) if len(valid_df) else math.nan
        novelty_validation = math.nan
        if len(valid_df):
            unique_set = set(valid_df["canonical_smiles"].dropna().astype(str))
            novelty_validation = (
                float(len([smiles for smiles in unique_set if smiles not in validation_canonical]) / len(unique_set))
                if unique_set
                else math.nan
            )

        length_values = pd.to_numeric(stage_df.get("peptide_length"), errors="coerce")
        topology_counts = stage_df["Cyclization"].value_counts(normalize=True).to_dict() if "Cyclization" in stage_df.columns else {}
        rows.append(
            {
                "stage": stage,
                "n_total": int(total),
                "n_valid_smiles": int(len(valid_df)),
                "valid_fraction": float(len(valid_df) / total) if total else math.nan,
                "n_invalid_smiles": int(total - len(valid_df)),
                "invalid_fraction": float((total - len(valid_df)) / total) if total else math.nan,
                "n_unique_canonical_smiles": int(unique_canonical),
                "unique_fraction_among_valid": float(unique_canonical / len(valid_df)) if len(valid_df) else math.nan,
                "duplicate_count": int(len(valid_df) - unique_canonical),
                "duplicate_fraction": duplicate_fraction,
                "novel_fraction_vs_validation": novelty_validation,
                "length_mean": float(np.nanmean(length_values)) if np.any(~np.isnan(length_values)) else math.nan,
                "length_median": float(np.nanmedian(length_values)) if np.any(~np.isnan(length_values)) else math.nan,
                "topology_distribution_json": json.dumps(topology_counts, ensure_ascii=False, sort_keys=True),
            }
        )

        duplicate_table = (
            valid_df.groupby("canonical_smiles")
            .agg(count=("canonical_smiles", "size"), permeability_mean=("permeability", "mean"), permeability_max=("permeability", "max"))
            .reset_index()
        )
        duplicate_table = duplicate_table[duplicate_table["count"] > 1].copy()
        if not duplicate_table.empty:
            duplicate_table["stage"] = stage
            duplicate_rows.append(duplicate_table)

        scaffold_info = valid_df["canonical_smiles"].map(murcko_scaffold_smiles)
        valid_df["murcko_scaffold"] = scaffold_info.map(lambda item: item[0])
        valid_df["generic_scaffold"] = scaffold_info.map(lambda item: item[1])

        for scaffold_type in ["murcko_scaffold", "generic_scaffold"]:
            nonempty = valid_df[valid_df[scaffold_type] != ""].copy()
            counts = nonempty[scaffold_type].value_counts()
            total_scaffolds = counts.sum()
            top_fractions = counts.head(10).sum() / total_scaffolds if total_scaffolds else math.nan
            top1 = counts.iloc[0] / total_scaffolds if total_scaffolds else math.nan
            top5 = counts.head(5).sum() / total_scaffolds if total_scaffolds else math.nan
            top10 = counts.head(10).sum() / total_scaffolds if total_scaffolds else math.nan
            singleton_fraction = float((counts == 1).sum() / len(counts)) if len(counts) else math.nan
            scaffold_rows.append(
                {
                    "stage": stage,
                    "scaffold_type": scaffold_type,
                    "n_valid": int(len(valid_df)),
                    "n_unique_scaffolds": int(len(counts)),
                    "scaffold_per_molecule_ratio": float(len(counts) / len(valid_df)) if len(valid_df) else math.nan,
                    "n_singleton_scaffolds": int((counts == 1).sum()),
                    "singleton_scaffold_fraction": singleton_fraction,
                    "top1_scaffold_fraction": float(top1) if not math.isnan(top1) else math.nan,
                    "top5_scaffold_fraction": float(top5) if not math.isnan(top5) else math.nan,
                    "top10_scaffold_fraction": float(top10) if not math.isnan(top10) else math.nan,
                    "scaffold_entropy": scaffold_entropy(counts.to_numpy(dtype=float)),
                    "gini_coefficient_of_scaffold_frequency": gini_coefficient(counts.to_numpy(dtype=float)),
                }
            )
            top_table = (
                nonempty.groupby(scaffold_type)
                .agg(
                    count=(scaffold_type, "size"),
                    mean_permeability=("permeability", "mean"),
                    median_permeability=("permeability", "median"),
                    max_permeability=("permeability", "max"),
                )
                .reset_index()
                .rename(columns={scaffold_type: "scaffold_smiles"})
                .sort_values("count", ascending=False)
                .head(20)
            )
            if not top_table.empty:
                top_table["stage"] = stage
                top_table["scaffold_type"] = scaffold_type
                top_table["fraction"] = top_table["count"] / len(nonempty) if len(nonempty) else math.nan
                top_scaffold_rows.append(top_table)

    quality_df = pd.DataFrame(rows)
    scaffold_df = pd.DataFrame(scaffold_rows)
    duplicate_df = pd.concat(duplicate_rows, ignore_index=True) if duplicate_rows else pd.DataFrame(
        columns=["canonical_smiles", "count", "permeability_mean", "permeability_max", "stage"]
    )
    top_scaffold_df = pd.concat(top_scaffold_rows, ignore_index=True) if top_scaffold_rows else pd.DataFrame(
        columns=["scaffold_smiles", "count", "mean_permeability", "median_permeability", "max_permeability", "stage", "scaffold_type", "fraction"]
    )

    before_row = quality_df.set_index("stage").loc["before"]
    after_row = quality_df.set_index("stage").loc["after"]
    scaffold_after = scaffold_df[(scaffold_df["stage"] == "after") & (scaffold_df["scaffold_type"] == "murcko_scaffold")]
    scaffold_before = scaffold_df[(scaffold_df["stage"] == "before") & (scaffold_df["scaffold_type"] == "murcko_scaffold")]
    if not scaffold_after.empty and not scaffold_before.empty:
        before_sc = scaffold_before.iloc[0]
        after_sc = scaffold_after.iloc[0]
        flag_checks = {
            "valid_fraction_drop_gt_0_05": float(after_row["valid_fraction"]) < float(before_row["valid_fraction"]) - 0.05,
            "unique_fraction_drop_gt_0_10": float(after_row["unique_fraction_among_valid"]) < float(before_row["unique_fraction_among_valid"]) - 0.10,
            "top1_scaffold_fraction_gt_0_10": float(after_sc["top1_scaffold_fraction"]) > 0.10,
            "top5_scaffold_fraction_gt_0_30": float(after_sc["top5_scaffold_fraction"]) > 0.30,
            "top10_scaffold_fraction_gt_0_50": float(after_sc["top10_scaffold_fraction"]) > 0.50,
            "unique_scaffolds_below_70pct_before": float(after_sc["n_unique_scaffolds"]) < 0.70 * float(before_sc["n_unique_scaffolds"]),
            "duplicate_fraction_increase_gt_0_10": float(after_row["duplicate_fraction"]) > float(before_row["duplicate_fraction"]) + 0.10,
        }
        for name, value in flag_checks.items():
            reward_hacking_rows.append({"flag": name, "triggered": bool(value)})

    maybe_write_csv(quality_df, context.tables_dir / "experiment2_generation_quality_summary.csv", enabled=write_tables)
    maybe_write_csv(scaffold_df, context.tables_dir / "experiment2_scaffold_summary.csv", enabled=write_tables)
    maybe_write_csv(top_scaffold_df, context.tables_dir / "experiment2_top_scaffolds.csv", enabled=write_tables)
    maybe_write_csv(pd.DataFrame(reward_hacking_rows), context.tables_dir / "experiment2_reward_hacking_flags.csv", enabled=write_tables)
    maybe_write_csv(duplicate_df, context.tables_dir / "experiment2_duplicate_frequency.csv", enabled=write_tables)
    maybe_write_csv(token_enrichment_df, context.tables_dir / "experiment2_token_enrichment.csv", enabled=write_tables)
    maybe_write_csv(token_summary_df, context.tables_dir / "experiment2_n_methyl_summary.csv", enabled=write_tables)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    metric_names = ["valid_fraction", "unique_fraction_among_valid", "novel_fraction_vs_validation"]
    x = np.arange(len(metric_names))
    width = 0.35
    before_vals = [float(before_row[name]) for name in metric_names]
    after_vals = [float(after_row[name]) for name in metric_names]
    ax.bar(x - width / 2, before_vals, width=width, color="#999999", label="Before")
    ax.bar(x + width / 2, after_vals, width=width, color="#0072B2", label="After")
    ax.set_xticks(x, ["Validity", "Uniqueness", "Novelty vs valid"])
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    save_figure(fig, context.figures_dir / "experiment2_valid_unique_novelty_bar")

    scaffold_plot = scaffold_df[scaffold_df["scaffold_type"] == "murcko_scaffold"].copy()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    metric_names = ["scaffold_per_molecule_ratio", "singleton_scaffold_fraction", "top1_scaffold_fraction"]
    x = np.arange(len(metric_names))
    before_sc_vals = [float(scaffold_plot[scaffold_plot["stage"] == "before"][name].iloc[0]) for name in metric_names]
    after_sc_vals = [float(scaffold_plot[scaffold_plot["stage"] == "after"][name].iloc[0]) for name in metric_names]
    ax.bar(x - width / 2, before_sc_vals, width=width, color="#999999", label="Before")
    ax.bar(x + width / 2, after_sc_vals, width=width, color="#0072B2", label="After")
    ax.set_xticks(x, ["Scaffold ratio", "Singleton frac", "Top1 frac"])
    ax.legend(frameon=False)
    save_figure(fig, context.figures_dir / "experiment2_scaffold_diversity_bar")

    top_scaffold_plot = top_scaffold_df[top_scaffold_df["scaffold_type"] == "murcko_scaffold"].copy()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for stage, color in [("before", "#999999"), ("after", "#0072B2")]:
        subset = top_scaffold_plot[top_scaffold_plot["stage"] == stage].head(20)
        ax.plot(np.arange(1, len(subset) + 1), subset["fraction"], marker="o", linewidth=1.5, color=color, label=stage.capitalize())
    ax.set_xlabel("Scaffold rank")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    save_figure(fig, context.figures_dir / "experiment2_top_scaffold_frequency")

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for stage, color in [("before", "#999999"), ("after", "#0072B2")]:
        subset = top_scaffold_plot[top_scaffold_plot["stage"] == stage].copy()
        subset = subset.sort_values("count", ascending=False).reset_index(drop=True)
        ax.plot(np.arange(1, len(subset) + 1), subset["count"], marker="o", linewidth=1.5, color=color, label=stage.capitalize())
    ax.set_xlabel("Scaffold rank")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    save_figure(fig, context.figures_dir / "experiment2_scaffold_rank_frequency")

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    before_lengths = pd.to_numeric(analysis_df[analysis_df["stage"] == "before"]["peptide_length"], errors="coerce").dropna().to_numpy(dtype=float)
    after_lengths = pd.to_numeric(analysis_df[analysis_df["stage"] == "after"]["peptide_length"], errors="coerce").dropna().to_numpy(dtype=float)
    length_min = int(min(before_lengths.min(), after_lengths.min()))
    length_max = int(max(before_lengths.max(), after_lengths.max()))
    bins = np.arange(length_min, length_max + 2) - 0.5
    ax.hist(before_lengths, bins=bins, alpha=0.4, color="#999999", label="Before")
    ax.hist(after_lengths, bins=bins, alpha=0.4, color="#0072B2", label="After")
    ax.legend(frameon=False)
    ax.set_xlabel("Peptide length")
    ax.set_ylabel("Count")
    save_figure(fig, context.figures_dir / "experiment2_length_distribution")

    token_plot_df = token_enrichment_df.reindex(token_enrichment_df["delta_fraction"].abs().sort_values(ascending=False).index).head(20).copy()
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.28 * len(token_plot_df))))
    y = np.arange(len(token_plot_df))
    ax.barh(y, token_plot_df["delta_fraction"], color=np.where(token_plot_df["delta_fraction"] >= 0, "#0072B2", "#999999"))
    ax.set_yticks(y, token_plot_df["token"])
    ax.invert_yaxis()
    ax.set_xlabel("After - Before token fraction")
    ax.set_ylabel("Residue / motif token")
    save_figure(fig, context.figures_dir / "experiment2_token_enrichment_bar")

    n_methyl_plot = token_enrichment_df[token_enrichment_df["is_n_methyl_token"]].copy()
    n_methyl_plot = n_methyl_plot.reindex(n_methyl_plot["delta_fraction"].abs().sort_values(ascending=False).index).head(20)
    if not n_methyl_plot.empty:
        fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.28 * len(n_methyl_plot))))
        y = np.arange(len(n_methyl_plot))
        ax.barh(y, n_methyl_plot["delta_fraction"], color="#D55E00")
        ax.set_yticks(y, n_methyl_plot["token"])
        ax.invert_yaxis()
        ax.set_xlabel("After - Before token fraction")
        ax.set_ylabel("N-methyl token")
        save_figure(fig, context.figures_dir / "experiment2_n_methyl_token_enrichment")

    return {
        "quality_df": quality_df,
        "scaffold_df": scaffold_df,
        "top_scaffold_df": top_scaffold_df,
        "reward_hacking_df": pd.DataFrame(reward_hacking_rows),
        "token_enrichment_df": token_enrichment_df,
        "token_summary_df": token_summary_df,
    }


def experiment3_descriptor_shift(
    analysis_df: pd.DataFrame,
    *,
    context: AnalysisContext,
    write_tables: bool,
) -> dict[str, Any]:
    valid_df = analysis_df[analysis_df["is_valid_smiles"] & analysis_df["permeability"].notna()].copy()
    descriptor_values = valid_df.copy()
    descriptor_value_columns = [
        "stage",
        "canonical_smiles",
        "permeability",
        *[column for column in REQUIRED_DESCRIPTOR_COLUMNS if column in descriptor_values.columns],
        "peptide_length",
        "hydrophobic_residue_fraction",
        "polar_residue_fraction",
        "charged_residue_fraction",
        "aromatic_residue_fraction",
        "n_non_natural_resolved",
        "frac_non_natural_resolved",
        "n_methyl_residue_count",
        "n_methyl_residue_fraction",
    ]
    descriptor_values = descriptor_values[[column for column in descriptor_value_columns if column in descriptor_values.columns]].copy()
    maybe_write_csv(descriptor_values, context.tables_dir / "experiment3_descriptor_values.csv", enabled=write_tables)

    descriptor_names = [
        column
        for column in descriptor_values.columns
        if column not in {"stage", "canonical_smiles", "permeability"}
    ]
    shift_rows = []
    corr_rows = []
    top_bottom_rows = []

    before_df = valid_df[valid_df["stage"] == "before"].copy()
    after_df = valid_df[valid_df["stage"] == "after"].copy()

    for descriptor in descriptor_names:
        before_values = pd.to_numeric(before_df[descriptor], errors="coerce").dropna().to_numpy(dtype=float)
        after_values = pd.to_numeric(after_df[descriptor], errors="coerce").dropna().to_numpy(dtype=float)
        if before_values.size == 0 or after_values.size == 0:
            continue
        shift_rows.append(
            {
                "descriptor": descriptor,
                "before_mean": float(np.mean(before_values)),
                "after_mean": float(np.mean(after_values)),
                "delta_mean": float(np.mean(after_values) - np.mean(before_values)),
                "before_median": float(np.median(before_values)),
                "after_median": float(np.median(after_values)),
                "delta_median": float(np.median(after_values) - np.median(before_values)),
                "before_std": float(np.std(before_values, ddof=1)) if before_values.size > 1 else 0.0,
                "after_std": float(np.std(after_values, ddof=1)) if after_values.size > 1 else 0.0,
                "before_q25": float(np.quantile(before_values, 0.25)),
                "before_q75": float(np.quantile(before_values, 0.75)),
                "after_q25": float(np.quantile(after_values, 0.25)),
                "after_q75": float(np.quantile(after_values, 0.75)),
                "cohen_d": cohens_d(before_values, after_values),
                "cliffs_delta": cliffs_delta(before_values, after_values),
                "mannwhitney_p": float(stats.mannwhitneyu(before_values, after_values, alternative="two-sided").pvalue),
                "ks_p": float(stats.ks_2samp(before_values, after_values).pvalue),
                "wasserstein_distance": float(stats.wasserstein_distance(before_values, after_values)),
            }
        )

        combined_values = pd.to_numeric(valid_df[descriptor], errors="coerce")
        combined_perm = pd.to_numeric(valid_df["permeability"], errors="coerce")
        mask = combined_values.notna() & combined_perm.notna()
        if mask.sum() >= 3:
            corr_rows.append(
                {
                    "descriptor": descriptor,
                    "spearman_all": float(stats.spearmanr(combined_values[mask], combined_perm[mask]).statistic),
                    "spearman_before": float(
                        stats.spearmanr(
                            pd.to_numeric(before_df[descriptor], errors="coerce"),
                            pd.to_numeric(before_df["permeability"], errors="coerce"),
                            nan_policy="omit",
                        ).statistic
                    ),
                    "spearman_after": float(
                        stats.spearmanr(
                            pd.to_numeric(after_df[descriptor], errors="coerce"),
                            pd.to_numeric(after_df["permeability"], errors="coerce"),
                            nan_policy="omit",
                        ).statistic
                    ),
                    "pearson_all": float(stats.pearsonr(combined_values[mask], combined_perm[mask]).statistic),
                }
            )

        for stage_name, stage_df in [("before", before_df), ("after", after_df)]:
            stage_numeric = stage_df[pd.to_numeric(stage_df[descriptor], errors="coerce").notna()].copy()
            if len(stage_numeric) < 10:
                continue
            stage_numeric = stage_numeric.sort_values("permeability")
            band = max(1, int(math.ceil(len(stage_numeric) * 0.10)))
            bottom = pd.to_numeric(stage_numeric.head(band)[descriptor], errors="coerce").dropna().to_numpy(dtype=float)
            top = pd.to_numeric(stage_numeric.tail(band)[descriptor], errors="coerce").dropna().to_numpy(dtype=float)
            if bottom.size == 0 or top.size == 0:
                continue
            top_bottom_rows.append(
                {
                    "stage": stage_name,
                    "descriptor": descriptor,
                    "bottom_mean": float(np.mean(bottom)),
                    "top_mean": float(np.mean(top)),
                    "delta_top_minus_bottom": float(np.mean(top) - np.mean(bottom)),
                    "effect_size": cohens_d(bottom, top),
                    "mannwhitney_p": float(stats.mannwhitneyu(bottom, top, alternative="two-sided").pvalue),
                }
            )

    shift_df = pd.DataFrame(shift_rows).sort_values("descriptor").reset_index(drop=True)
    corr_df = pd.DataFrame(corr_rows).sort_values("descriptor").reset_index(drop=True)
    top_bottom_df = pd.DataFrame(top_bottom_rows).sort_values(["stage", "descriptor"]).reset_index(drop=True)
    maybe_write_csv(shift_df, context.tables_dir / "experiment3_descriptor_shift_summary.csv", enabled=write_tables)
    maybe_write_csv(corr_df, context.tables_dir / "experiment3_descriptor_correlation.csv", enabled=write_tables)
    maybe_write_csv(top_bottom_df, context.tables_dir / "experiment3_top_bottom_descriptor_summary.csv", enabled=write_tables)

    set_plot_style()
    heatmap_df = shift_df[shift_df["descriptor"].isin(SELECTED_DESCRIPTOR_COLUMNS)].copy()
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.35 * len(heatmap_df))))
    matrix = heatmap_df[["delta_mean", "cohen_d", "wasserstein_distance"]].to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(heatmap_df)), heatmap_df["descriptor"].tolist())
    ax.set_xticks(np.arange(3), ["delta_mean", "cohen_d", "wasserstein"])
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    save_figure(fig, context.figures_dir / "experiment3_descriptor_shift_heatmap")

    selected = [descriptor for descriptor in SELECTED_DESCRIPTOR_COLUMNS if descriptor in valid_df.columns]
    fig, axes = plt.subplots(len(selected), 1, figsize=(5.5, max(6.0, 1.4 * len(selected))), sharex=False)
    if len(selected) == 1:
        axes = [axes]
    for ax, descriptor in zip(axes, selected):
        before_vals = pd.to_numeric(before_df[descriptor], errors="coerce").dropna().to_numpy(dtype=float)
        after_vals = pd.to_numeric(after_df[descriptor], errors="coerce").dropna().to_numpy(dtype=float)
        violin = ax.violinplot([before_vals, after_vals], positions=[1, 2], showmeans=False, showmedians=False, showextrema=False)
        for body, color in zip(violin["bodies"], ["#999999", "#0072B2"]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.35)
        ax.boxplot(
            [before_vals, after_vals],
            positions=[1, 2],
            widths=0.18,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "black"},
            medianprops={"color": "black"},
        )
        ax.set_xticks([1, 2], ["Before", "After"])
        ax.set_ylabel(descriptor)
    save_figure(fig, context.figures_dir / "experiment3_descriptor_violin_selected")

    scatter_descriptors = [descriptor for descriptor in ["MolLogP", "TPSA", "peptide_length"] if descriptor in valid_df.columns]
    fig, axes = plt.subplots(1, len(scatter_descriptors), figsize=(5.0 * len(scatter_descriptors), 3.2), sharey=True)
    if len(scatter_descriptors) == 1:
        axes = [axes]
    for ax, descriptor in zip(axes, scatter_descriptors):
        for stage_name, color in [("before", "#999999"), ("after", "#0072B2")]:
            stage_df = valid_df[valid_df["stage"] == stage_name].copy()
            x = pd.to_numeric(stage_df[descriptor], errors="coerce")
            y = pd.to_numeric(stage_df["permeability"], errors="coerce")
            mask = x.notna() & y.notna()
            ax.scatter(x[mask], y[mask], s=8, alpha=0.25, color=color, label=stage_name.capitalize())
        ax.set_xlabel(descriptor)
        ax.legend(frameon=False)
    axes[0].set_ylabel("PAMPA_value")
    save_figure(fig, context.figures_dir / "experiment3_descriptor_vs_permeability_scatter")

    corr_plot = corr_df[corr_df["descriptor"].isin(selected)].copy()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(corr_plot))
    ax.bar(x, corr_plot["spearman_all"], color="#0072B2", alpha=0.8)
    ax.set_xticks(x, corr_plot["descriptor"], rotation=30, ha="right")
    ax.set_ylabel("Spearman (all)")
    save_figure(fig, context.figures_dir / "experiment3_correlation_bar")

    return {
        "shift_df": shift_df,
        "corr_df": corr_df,
        "top_bottom_df": top_bottom_df,
    }


def stage_scaffold_table(dataframe: pd.DataFrame, *, radius: int, fp_bits: int) -> pd.DataFrame:
    frame = dataframe.copy()
    scaffold_pairs = frame["canonical_smiles"].fillna("").map(lambda smiles: murcko_scaffold_smiles(smiles) if smiles else ("", ""))
    frame["murcko_scaffold"] = scaffold_pairs.map(lambda item: item[0])
    frame["generic_scaffold"] = scaffold_pairs.map(lambda item: item[1])
    frame["morgan_fp"] = frame["canonical_smiles"].fillna("").map(lambda smiles: morgan_bitvect(smiles, radius=radius, fp_bits=fp_bits) if smiles else None)
    return frame


def match_pairs(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    *,
    radius: int,
    fp_bits: int,
) -> pd.DataFrame:
    before_pool = stage_scaffold_table(before_df, radius=radius, fp_bits=fp_bits).reset_index(drop=True)
    after_pool = stage_scaffold_table(after_df, radius=radius, fp_bits=fp_bits).reset_index(drop=True)
    used_before: set[int] = set()
    pair_rows: list[dict[str, Any]] = []

    matching_rules = [
        {"same_length": True, "length_tol": 0, "mw_tol": 100.0, "tpsa_tol": 50.0, "logp_tol": 2.0, "tanimoto_min": 0.30, "rule": "exact_length"},
        {"same_length": False, "length_tol": 1, "mw_tol": 100.0, "tpsa_tol": 50.0, "logp_tol": 2.0, "tanimoto_min": 0.30, "rule": "length_pm1"},
        {"same_length": False, "length_tol": 2, "mw_tol": 150.0, "tpsa_tol": 50.0, "logp_tol": 2.0, "tanimoto_min": 0.20, "rule": "length_pm2_relaxed"},
    ]

    for after_index, after_row in after_pool.iterrows():
        best: tuple[float, int, dict[str, Any]] | None = None
        for rule in matching_rules:
            candidates = before_pool.loc[~before_pool.index.isin(used_before)].copy()
            if candidates.empty:
                continue
            if pd.notna(after_row.get("peptide_length")):
                length_values = pd.to_numeric(candidates["peptide_length"], errors="coerce")
                target_length = float(after_row["peptide_length"])
                if rule["same_length"]:
                    candidates = candidates[length_values == target_length]
                else:
                    candidates = candidates[(length_values - target_length).abs() <= rule["length_tol"]]
            candidates = candidates[
                (pd.to_numeric(candidates["MolWt"], errors="coerce") - float(after_row["MolWt"])).abs() <= rule["mw_tol"]
            ]
            candidates = candidates[
                (pd.to_numeric(candidates["TPSA"], errors="coerce") - float(after_row["TPSA"])).abs() <= rule["tpsa_tol"]
            ]
            candidates = candidates[
                (pd.to_numeric(candidates["MolLogP"], errors="coerce") - float(after_row["MolLogP"])).abs() <= rule["logp_tol"]
            ]
            if candidates.empty:
                continue
            for before_index, before_row in candidates.iterrows():
                similarity = tanimoto_similarity(after_row["morgan_fp"], before_row["morgan_fp"])
                if math.isnan(similarity) or similarity < rule["tanimoto_min"]:
                    continue
                score = similarity
                if best is None or score > best[0]:
                    best = (
                        score,
                        before_index,
                        {
                            "after_index": int(after_index),
                            "before_index": int(before_index),
                            "after_smiles": after_row["SMILES"],
                            "before_smiles": before_row["SMILES"],
                            "after_permeability": float(after_row["permeability"]),
                            "before_permeability": float(before_row["permeability"]),
                            "delta_permeability": float(after_row["permeability"] - before_row["permeability"]),
                            "tanimoto_similarity": float(similarity),
                            "same_topology": bool(after_row["Cyclization"] == before_row["Cyclization"]),
                            "same_length": bool(after_row.get("peptide_length") == before_row.get("peptide_length")),
                            "length_after": after_row.get("peptide_length"),
                            "length_before": before_row.get("peptide_length"),
                            "MolWt_after": float(after_row["MolWt"]),
                            "MolWt_before": float(before_row["MolWt"]),
                            "MolLogP_after": float(after_row["MolLogP"]),
                            "MolLogP_before": float(before_row["MolLogP"]),
                            "TPSA_after": float(after_row["TPSA"]),
                            "TPSA_before": float(before_row["TPSA"]),
                            "Murcko_after": after_row["murcko_scaffold"],
                            "Murcko_before": before_row["murcko_scaffold"],
                            "same_murcko": bool(after_row["murcko_scaffold"] == before_row["murcko_scaffold"]),
                            "generic_scaffold_after": after_row["generic_scaffold"],
                            "generic_scaffold_before": before_row["generic_scaffold"],
                            "same_generic_scaffold": bool(after_row["generic_scaffold"] == before_row["generic_scaffold"]),
                            "matching_rule": rule["rule"],
                        },
                    )
            if best is not None:
                break
        if best is not None:
            used_before.add(best[1])
            pair_rows.append(best[2])

    return pd.DataFrame(pair_rows)


def experiment4_controlled_comparison(
    analysis_df: pd.DataFrame,
    *,
    context: AnalysisContext,
    n_bootstrap: int,
    rng: np.random.Generator,
    radius: int,
    fp_bits: int,
    write_tables: bool,
) -> dict[str, Any]:
    valid_df = analysis_df[analysis_df["is_valid_smiles"] & analysis_df["permeability"].notna()].copy()
    before_df = valid_df[valid_df["stage"] == "before"].copy()
    after_df = valid_df[valid_df["stage"] == "after"].copy()

    length_rows = []
    if "peptide_length" in valid_df.columns:
        before_groups = before_df.groupby("peptide_length")
        after_groups = after_df.groupby("peptide_length")
        shared_lengths = sorted(set(before_groups.groups.keys()).intersection(after_groups.groups.keys()))
        for length in shared_lengths:
            group_before = pd.to_numeric(before_groups.get_group(length)["permeability"], errors="coerce").dropna().to_numpy(dtype=float)
            group_after = pd.to_numeric(after_groups.get_group(length)["permeability"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(group_before) < 20 or len(group_after) < 20:
                continue
            length_rows.append(
                {
                    "length": int(length),
                    "n_before": int(len(group_before)),
                    "n_after": int(len(group_after)),
                    "before_mean_permeability": float(np.mean(group_before)),
                    "after_mean_permeability": float(np.mean(group_after)),
                    "delta_mean": float(np.mean(group_after) - np.mean(group_before)),
                    "before_median_permeability": float(np.median(group_before)),
                    "after_median_permeability": float(np.median(group_after)),
                    "delta_median": float(np.median(group_after) - np.median(group_before)),
                    "mannwhitney_p": float(stats.mannwhitneyu(group_before, group_after, alternative="two-sided").pvalue),
                    "effect_size": cohens_d(group_before, group_after),
                }
            )
    length_df = pd.DataFrame(length_rows)
    maybe_write_csv(length_df, context.tables_dir / "experiment4_length_stratified_summary.csv", enabled=write_tables)

    scaffold_rows = []
    before_sc = stage_scaffold_table(before_df, radius=radius, fp_bits=fp_bits)
    after_sc = stage_scaffold_table(after_df, radius=radius, fp_bits=fp_bits)
    for scaffold_type in ["murcko_scaffold", "generic_scaffold"]:
        before_groups = before_sc[before_sc[scaffold_type] != ""].groupby(scaffold_type)
        after_groups = after_sc[after_sc[scaffold_type] != ""].groupby(scaffold_type)
        shared_scaffolds = sorted(set(before_groups.groups.keys()).intersection(after_groups.groups.keys()))
        for scaffold in shared_scaffolds:
            group_before = pd.to_numeric(before_groups.get_group(scaffold)["permeability"], errors="coerce").dropna().to_numpy(dtype=float)
            group_after = pd.to_numeric(after_groups.get_group(scaffold)["permeability"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(group_before) < 5 or len(group_after) < 5:
                continue
            scaffold_rows.append(
                {
                    "scaffold_type": scaffold_type,
                    "scaffold_smiles": scaffold,
                    "n_before": int(len(group_before)),
                    "n_after": int(len(group_after)),
                    "before_mean_permeability": float(np.mean(group_before)),
                    "after_mean_permeability": float(np.mean(group_after)),
                    "delta_mean": float(np.mean(group_after) - np.mean(group_before)),
                    "before_median_permeability": float(np.median(group_before)),
                    "after_median_permeability": float(np.median(group_after)),
                    "delta_median": float(np.median(group_after) - np.median(group_before)),
                    "mannwhitney_p": float(stats.mannwhitneyu(group_before, group_after, alternative="two-sided").pvalue),
                    "effect_size": cohens_d(group_before, group_after),
                }
            )
    scaffold_df = pd.DataFrame(scaffold_rows)
    maybe_write_csv(scaffold_df, context.tables_dir / "experiment4_scaffold_controlled_summary.csv", enabled=write_tables)

    matched_pairs_df = match_pairs(before_df, after_df, radius=radius, fp_bits=fp_bits)
    maybe_write_csv(matched_pairs_df, context.tables_dir / "experiment4_matched_pairs.csv", enabled=write_tables)

    if not matched_pairs_df.empty:
        deltas = matched_pairs_df["delta_permeability"].to_numpy(dtype=float)
        mean_ci = bootstrap_statistic(
            np.zeros_like(deltas),
            deltas,
            fn=np.mean,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        median_ci = bootstrap_statistic(
            np.zeros_like(deltas),
            deltas,
            fn=np.median,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        matched_summary_df = pd.DataFrame(
            [
                {
                    "n_pairs": int(len(matched_pairs_df)),
                    "mean_delta": float(np.mean(deltas)),
                    "median_delta": float(np.median(deltas)),
                    "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                    "q25_delta": float(np.quantile(deltas, 0.25)),
                    "q75_delta": float(np.quantile(deltas, 0.75)),
                    "fraction_delta_positive": float(np.mean(deltas > 0)),
                    "wilcoxon_signed_rank_p": float(stats.wilcoxon(deltas).pvalue) if len(deltas) >= 5 else math.nan,
                    "paired_ttest_p": float(stats.ttest_1samp(deltas, popmean=0.0).pvalue) if len(deltas) >= 2 else math.nan,
                    "bootstrap_mean_delta_ci_low": mean_ci[0],
                    "bootstrap_mean_delta_ci_high": mean_ci[1],
                    "bootstrap_median_delta_ci_low": median_ci[0],
                    "bootstrap_median_delta_ci_high": median_ci[1],
                    "mean_tanimoto": float(matched_pairs_df["tanimoto_similarity"].mean()),
                    "median_tanimoto": float(matched_pairs_df["tanimoto_similarity"].median()),
                    "fraction_same_murcko": float(matched_pairs_df["same_murcko"].mean()),
                    "fraction_same_generic_scaffold": float(matched_pairs_df["same_generic_scaffold"].mean()),
                }
            ]
        )
    else:
        matched_summary_df = pd.DataFrame(
            [
                {
                    "n_pairs": 0,
                    "mean_delta": math.nan,
                    "median_delta": math.nan,
                    "std_delta": math.nan,
                    "q25_delta": math.nan,
                    "q75_delta": math.nan,
                    "fraction_delta_positive": math.nan,
                    "wilcoxon_signed_rank_p": math.nan,
                    "paired_ttest_p": math.nan,
                    "bootstrap_mean_delta_ci_low": math.nan,
                    "bootstrap_mean_delta_ci_high": math.nan,
                    "bootstrap_median_delta_ci_low": math.nan,
                    "bootstrap_median_delta_ci_high": math.nan,
                    "mean_tanimoto": math.nan,
                    "median_tanimoto": math.nan,
                    "fraction_same_murcko": math.nan,
                    "fraction_same_generic_scaffold": math.nan,
                }
            ]
        )
    maybe_write_csv(matched_summary_df, context.tables_dir / "experiment4_matched_pair_summary.csv", enabled=write_tables)

    regression_rows = []
    try:
        import statsmodels.api as sm

        regression_df = valid_df.copy()
        regression_df["stage_after"] = (regression_df["stage"] == "after").astype(int)
        predictor_cols = ["stage_after", "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "peptide_length"]
        regression_df = regression_df.dropna(subset=["permeability"] + predictor_cols)
        X = sm.add_constant(regression_df[predictor_cols])
        y = regression_df["permeability"]
        model = sm.OLS(y, X).fit()
        for term in ["stage_after", *predictor_cols[1:]]:
            regression_rows.append(
                {
                    "model": "ols_main",
                    "term": term,
                    "coef": float(model.params[term]),
                    "pvalue": float(model.pvalues[term]),
                    "stderr": float(model.bse[term]),
                    "ci_low": float(model.conf_int().loc[term, 0]),
                    "ci_high": float(model.conf_int().loc[term, 1]),
                    "r_squared": float(model.rsquared),
                }
            )
    except Exception as exc:
        regression_rows.append(
            {
                "model": "ols_main",
                "term": "error",
                "coef": math.nan,
                "pvalue": math.nan,
                "stderr": math.nan,
                "ci_low": math.nan,
                "ci_high": math.nan,
                "r_squared": math.nan,
                "note": f"statsmodels unavailable or fit failed: {exc}",
            }
        )
    regression_df = pd.DataFrame(regression_rows)
    maybe_write_csv(regression_df, context.tables_dir / "experiment4_regression_summary.csv", enabled=write_tables)

    set_plot_style()
    if not length_df.empty:
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        ax.plot(length_df["length"], length_df["before_mean_permeability"], marker="o", color="#999999", label="Before")
        ax.plot(length_df["length"], length_df["after_mean_permeability"], marker="o", color="#0072B2", label="After")
        ax.legend(frameon=False)
        ax.set_xlabel("Peptide length")
        ax.set_ylabel("Mean PAMPA_value")
        save_figure(fig, context.figures_dir / "experiment4_same_length_permeability")

    if not scaffold_df.empty:
        top_scaffolds = scaffold_df.sort_values("n_before", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        x = np.arange(len(top_scaffolds))
        ax.bar(x - 0.2, top_scaffolds["before_mean_permeability"], width=0.4, color="#999999", label="Before")
        ax.bar(x + 0.2, top_scaffolds["after_mean_permeability"], width=0.4, color="#0072B2", label="After")
        ax.set_xticks(x, [str(i + 1) for i in range(len(top_scaffolds))])
        ax.set_xlabel("Shared scaffold rank")
        ax.set_ylabel("Mean PAMPA_value")
        ax.legend(frameon=False)
        save_figure(fig, context.figures_dir / "experiment4_same_scaffold_permeability")

    if not matched_pairs_df.empty:
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.hist(matched_pairs_df["delta_permeability"], bins=40, color="#0072B2", alpha=0.75)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("After - Before PAMPA_value")
        ax.set_ylabel("Count")
        save_figure(fig, context.figures_dir / "experiment4_matched_delta_distribution")

        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.scatter(
            matched_pairs_df["before_permeability"],
            matched_pairs_df["after_permeability"],
            s=12,
            alpha=0.35,
            color="#0072B2",
        )
        low = float(min(matched_pairs_df["before_permeability"].min(), matched_pairs_df["after_permeability"].min()))
        high = float(max(matched_pairs_df["before_permeability"].max(), matched_pairs_df["after_permeability"].max()))
        ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        ax.set_xlabel("Before PAMPA_value")
        ax.set_ylabel("After PAMPA_value")
        save_figure(fig, context.figures_dir / "experiment4_matched_pair_before_after")

    return {
        "length_df": length_df,
        "scaffold_df": scaffold_df,
        "matched_pairs_df": matched_pairs_df,
        "matched_summary_df": matched_summary_df,
        "regression_df": regression_df,
    }


def build_report(
    *,
    context: AnalysisContext,
    metadata: dict[str, Any],
    exp1: dict[str, Any],
    exp2: dict[str, Any],
    exp3: dict[str, Any],
    exp4: dict[str, Any],
) -> None:
    tests_df = exp1["tests_df"]
    summary_df = exp1["summary_df"]
    quality_df = exp2["quality_df"].set_index("stage")
    matched_summary_df = exp4["matched_summary_df"]

    before_summary = summary_df[(summary_df["stage"] == "before") & (summary_df["subset"] == "all_with_permeability")].iloc[0]
    after_summary = summary_df[(summary_df["stage"] == "after") & (summary_df["subset"] == "all_with_permeability")].iloc[0]
    test_row = tests_df[tests_df["subset"] == "all_with_permeability"].iloc[0]
    before_quality = quality_df.loc["before"]
    after_quality = quality_df.loc["after"]
    matched_row = matched_summary_df.iloc[0]
    top_descriptor_rows = exp3["shift_df"].copy().sort_values("delta_mean", ascending=False).head(3)
    bottom_descriptor_rows = exp3["shift_df"].copy().sort_values("delta_mean", ascending=True).head(3)
    token_summary_df = exp2["token_summary_df"].set_index("stage")
    token_enrichment_df = exp2["token_enrichment_df"]
    n_methyl_top = token_enrichment_df[token_enrichment_df["is_n_methyl_token"]].copy().sort_values("delta_fraction", ascending=False).head(5)

    if after_summary["mean"] > before_summary["mean"] and after_summary["median"] > before_summary["median"]:
        shift_note = "permeability distribution likely shifted broadly upward"
    elif after_summary["mean"] > before_summary["mean"]:
        shift_note = "mean permeability increased, but the gain may be concentrated in a smaller high-scoring subset"
    else:
        shift_note = "no clear positive global permeability shift was observed"

    reward_hacking_flags = exp2["reward_hacking_df"]
    triggered_flags = reward_hacking_flags[reward_hacking_flags["triggered"] == True]["flag"].tolist() if not reward_hacking_flags.empty else []
    if triggered_flags:
        hacking_note = f"potential reward-hacking flags were triggered: {', '.join(triggered_flags)}"
    else:
        hacking_note = "no automatic reward-hacking flags were triggered under the preset thresholds"

    matched_note = (
        "matched-pair analysis supports residual local improvement after controlling for simple confounders"
        if pd.notna(matched_row["mean_delta"]) and float(matched_row["mean_delta"]) > 0
        else "matched-pair analysis does not yet support a clear residual local improvement"
    )

    lines = [
        "# RL Permeability Analysis Summary",
        "",
        "## Executive Summary",
        "",
        f"- Mean PAMPA_value changed from `{before_summary['mean']:.4f}` to `{after_summary['mean']:.4f}`.",
        f"- Median PAMPA_value changed from `{before_summary['median']:.4f}` to `{after_summary['median']:.4f}`.",
        f"- Distribution-level interpretation: {shift_note}.",
        f"- Validity changed from `{before_quality['valid_fraction']:.4f}` to `{after_quality['valid_fraction']:.4f}`; uniqueness among valid molecules changed from `{before_quality['unique_fraction_among_valid']:.4f}` to `{after_quality['unique_fraction_among_valid']:.4f}`.",
        f"- Reward-hacking diagnostic: {hacking_note}.",
        f"- Matched-pair conclusion: {matched_note}.",
        f"- N-methyl token fraction changed from `{token_summary_df.loc['before','n_methyl_token_fraction']:.4f}` to `{token_summary_df.loc['after','n_methyl_token_fraction']:.4f}`.",
        "",
        "## Experiment 1 Summary",
        "",
        f"- Bootstrap mean-difference 95% CI: [`{test_row['bootstrap_mean_ci_low']:.4f}`, `{test_row['bootstrap_mean_ci_high']:.4f}`]",
        f"- Bootstrap median-difference 95% CI: [`{test_row['bootstrap_median_ci_low']:.4f}`, `{test_row['bootstrap_median_ci_high']:.4f}`]",
        f"- K-S p-value: `{test_row['ks_p']:.4e}`",
        f"- Wasserstein distance: `{test_row['wasserstein_distance']:.4f}`",
        "",
            "## Experiment 2 Summary",
            "",
            f"- Novelty was evaluated only against validation and changed from `{before_quality['novel_fraction_vs_validation']:.4f}` to `{after_quality['novel_fraction_vs_validation']:.4f}`.",
            f"- Length mean changed from `{before_quality['length_mean']:.3f}` to `{after_quality['length_mean']:.3f}`.",
            f"- Topology distributions are preserved in the standardized table via the `Cyclization` field.",
            f"- Sequence fraction with any N-methyl token changed from `{token_summary_df.loc['before','sequence_fraction_with_n_methyl']:.4f}` to `{token_summary_df.loc['after','sequence_fraction_with_n_methyl']:.4f}`.",
            "",
            "## Experiment 3 Summary",
        "",
        "- Largest positive descriptor shifts:",
    ]
    for _, row in top_descriptor_rows.iterrows():
        lines.append(f"  - `{row['descriptor']}`: delta_mean = `{row['delta_mean']:.4f}`")
    lines.append("- Largest negative descriptor shifts:")
    for _, row in bottom_descriptor_rows.iterrows():
        lines.append(f"  - `{row['descriptor']}`: delta_mean = `{row['delta_mean']:.4f}`")
    lines.extend(
        [
            "",
            "## Motif Enrichment Summary",
            "",
        ]
    )
    if not n_methyl_top.empty:
        lines.append("- Top enriched N-methyl-related tokens:")
        for _, row in n_methyl_top.iterrows():
            lines.append(
                f"  - `{row['token']}`: before_fraction=`{row['before_fraction']:.4f}`, "
                f"after_fraction=`{row['after_fraction']:.4f}`, delta=`{row['delta_fraction']:.4f}`"
            )
    else:
        lines.append("- No N-methyl-specific token enrichment was detected under the current tokenizer rules.")
    lines.extend(
        [
            "",
            "## Experiment 4 Summary",
            "",
            f"- Same-length summary rows: `{len(exp4['length_df'])}`",
            f"- Shared-scaffold summary rows: `{len(exp4['scaffold_df'])}`",
            f"- Matched pairs found: `{int(matched_row['n_pairs'])}`",
            f"- Mean matched delta: `{matched_row['mean_delta']}`",
            "",
            "## Reuse vs Recompute",
            "",
            f"- Directly reused columns: `{', '.join(metadata['reused_columns'])}`",
            f"- Newly computed columns: `{', '.join(metadata['computed_columns'])}`",
        ]
    )
    write_text("\n".join(lines) + "\n", context.report_dir / "rl_permeability_analysis_summary.md")


def write_run_record(
    *,
    args: argparse.Namespace,
    context: AnalysisContext,
    metadata: dict[str, Any],
) -> None:
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    lines = [
        "# Run Record",
        "",
        f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Command: `{command}`",
        f"- Before CSV: `{args.before_csv}`",
        f"- After CSV: `{args.after_csv}`",
        f"- Validation CSV: `{args.valid_csv}`",
        f"- Permeability column: `{args.permeability_col}`",
        f"- Hemo column: `{args.hemo_col}`",
        f"- Sequence column: `{args.sequence_col}`",
        f"- Topology assumption: `{args.topology}`",
        f"- n_bootstrap: `{args.n_bootstrap}`",
        f"- random_seed: `{args.random_seed}`",
        f"- fp_radius: `{args.fp_radius}`",
        f"- fp_bits: `{args.fp_bits}`",
        f"- smoke_rows: `{args.smoke_rows}`",
        f"- targets: `{args.targets}`",
        f"- skip_tables: `{args.skip_tables}`",
        f"- skip_report: `{args.skip_report}`",
        "",
        "## Reused Columns",
        "",
        *[f"- `{column}`" for column in metadata["reused_columns"]],
        "",
        "## Newly Computed Columns",
        "",
        *[f"- `{column}`" for column in metadata["computed_columns"]],
        "",
        "## Added motif outputs",
        "",
        "- `tables/experiment2_token_enrichment.csv`",
        "- `tables/experiment2_n_methyl_summary.csv`",
        "- `figures/experiment2_token_enrichment_bar.pdf` / `.png`",
        "- `figures/experiment2_n_methyl_token_enrichment.pdf` / `.png`",
        "",
    ]
    write_text("\n".join(lines), context.output_root / "run_record.md")


def main() -> None:
    args = parse_args()
    context = ensure_layout(args.out_dir)
    context.args = args
    rng = np.random.default_rng(args.random_seed)
    selected_targets = parse_targets(args.targets)
    write_tables = not args.skip_tables

    analysis_df, validation_df, metadata = prepare_analysis_dataframe(args)
    write_csv(analysis_df, context.metadata_dir / "standardized_analysis_table.csv")
    write_csv(validation_df, context.metadata_dir / "validation_reference_table.csv")
    write_run_record(args=args, context=context, metadata=metadata)

    exp1 = exp2 = exp3 = exp4 = None

    if any(
        should_run_target(selected_targets, target_name, "experiment1")
        for target_name in [
            "experiment1",
            "experiment1_permeability_violin_box",
            "experiment1_permeability_kde",
            "experiment1_permeability_ecdf",
            "experiment1_topk_enrichment_curve",
            "experiment1_hemo_kde",
        ]
    ):
        exp1 = experiment1_distribution_shift(
            analysis_df,
            context=context,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
            selected_targets=selected_targets,
            write_tables=write_tables,
        )

    if any(
        should_run_target(selected_targets, target_name, "experiment2")
        for target_name in [
            "experiment2",
            "experiment2_valid_unique_novelty_bar",
            "experiment2_scaffold_diversity_bar",
            "experiment2_top_scaffold_frequency",
            "experiment2_scaffold_rank_frequency",
            "experiment2_length_distribution",
            "experiment2_token_enrichment_bar",
            "experiment2_n_methyl_token_enrichment",
        ]
    ):
        exp2 = experiment2_generation_quality(
            analysis_df,
            validation_df,
            context=context,
            write_tables=write_tables,
        )

    if any(
        should_run_target(selected_targets, target_name, "experiment3")
        for target_name in [
            "experiment3",
            "experiment3_descriptor_shift_heatmap",
            "experiment3_descriptor_violin_selected",
            "experiment3_descriptor_vs_permeability_scatter",
            "experiment3_correlation_bar",
        ]
    ):
        exp3 = experiment3_descriptor_shift(
            analysis_df,
            context=context,
            write_tables=write_tables,
        )

    if any(
        should_run_target(selected_targets, target_name, "experiment4")
        for target_name in [
            "experiment4",
            "experiment4_same_length_permeability",
            "experiment4_same_scaffold_permeability",
            "experiment4_matched_delta_distribution",
            "experiment4_matched_pair_before_after",
        ]
    ):
        exp4 = experiment4_controlled_comparison(
            analysis_df,
            context=context,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
            radius=args.fp_radius,
            fp_bits=args.fp_bits,
            write_tables=write_tables,
        )

    if "all" in selected_targets and not args.skip_report:
        build_report(
            context=context,
            metadata=metadata,
            exp1=exp1,
            exp2=exp2,
            exp3=exp3,
            exp4=exp4,
        )

    summary = {
        "output_root": str(context.output_root),
        "tables_dir": str(context.tables_dir),
        "figures_dir": str(context.figures_dir),
        "report_dir": str(context.report_dir),
        "metadata_dir": str(context.metadata_dir),
        "rows_total": int(len(analysis_df)),
        "rows_before": int((analysis_df["stage"] == "before").sum()),
        "rows_after": int((analysis_df["stage"] == "after").sum()),
        "targets": sorted(selected_targets),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
