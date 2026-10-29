
# Scaffold diversity and novelty panels: editable control script
# -------------------------------------------------------------
# What this script draws:
#   Panel A: Scaffold diversity across methods
#   Panel B: Scaffold rank–frequency distribution
#   Panel C: Scaffold novelty relative to training set
#
# Required packages:
#   pandas, numpy, matplotlib, rdkit
#
# RDKit install example:
#   conda install -c conda-forge rdkit
#
# How to use:
#   1. Edit CONFIG below.
#   2. Replace METHOD_FILES with your real generated CSV files.
#   3. Set TRAIN_FILE to your real training CSV if you want true scaffold novelty.
#   4. Run: python scaffold_panels_control_console.py

import os
import math
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["axes.unicode_minus"] = False


# =============================
# 1. CONFIG: edit this section
# =============================

CONFIG = {
    # Input files. Replace None with your real baseline CSV paths.
    # The current uploaded file is placed as AAE based on the filename.
    "METHOD_FILES": {
        "AAE": 'results/property_annotation_runs/20260427_220128_aae_n30000_dual_gpu/gpu0_outputs/aae_head_to_tail_n30000.with_properties.with_smiles2peptide.csv',
        "VAE": 'results/property_annotation_runs/20260428_104307_vae_n30000_dual_gpu/gpu0_outputs/vae_head_to_tail_n30000.with_properties.with_smiles2peptide.csv',
        "CharRNN": 'results/property_annotation_runs/20260428_130517_char_rnn_n30000_dual_gpu/gpu0_outputs/char_rnn_head_to_tail_n30000.with_properties.with_smiles2peptide.csv',
        "PepINVENT_30": 'results/property_annotation_runs/20260428_160615_pepinvent_nine_csv_dual_gpu/gpu0_outputs/pepinvent_generated_masked_30_head_to_tail_processed.with_properties.with_smiles2peptide.csv',
        "PepINVENT_50": 'results/property_annotation_runs/20260428_160615_pepinvent_nine_csv_dual_gpu/gpu0_outputs/pepinvent_generated_masked_50_head_to_tail_keep_ends_processed.with_properties.with_smiles2peptide.csv',
        "PepINVENT_70": 'results/property_annotation_runs/20260428_160615_pepinvent_nine_csv_dual_gpu/gpu1_outputs/pepinvent_generated_masked_70_head_to_tail_keep_ends_processed.with_properties.with_smiles2peptide.csv',
        "CycGPT_LoRA": 'results/property_annotation_runs/20260423_115024_lora_cyclic_n30000_gpu0/outputs/head_to_tail_n30000.with_properties.with_smiles2peptide.csv',
    },

    # Training set CSV. Set this to your training file for real Panel C.
    # Example: "/mnt/data/restored_train_with_type.csv"
    "TRAIN_FILE": 'results/property_annotation_runs/20260429_115158_restored_validation_gpu0/gpu0_outputs/restored_validation.with_properties.with_smiles2peptide.csv',
    "TRAIN_CYCLIZATION_COLUMN": "Cyclization",

    # Choose one:
    # "head_to_tail", "sc_to_tail", "disulfide_bridge"
    # Also accepts "head-to-tail" because the code below normalizes '-' to '_'.
    "TRAIN_CYCLIZATION_TYPE": "head_to_tail",

    # Whether to apply cyclization filtering to TRAIN_FILE.
    "FILTER_TRAIN_BY_CYCLIZATION": True,

    # SMILES column candidates. The first existing column will be used.
    "SMILES_COLUMNS": ["canonical_smiles", "SMILES", "smiles", "Source Mol", "Target Mol"],

    # Sampling. Full Murcko decomposition can be slow for long cyclic peptides.
    # Use an integer such as 6000 for preview. Use None for full data.
    "SAMPLE_SIZE_PER_METHOD": 6000,
    "SAMPLE_SIZE_TRAIN": 50000,
    "RANDOM_SEED": 42,

    # Cache computed scaffolds to avoid recomputing RDKit Murcko scaffolds.
    "CACHE_DIR": './experiments/scaffold_statistics/scaffold_cache',
    "REBUILD_CACHE": False,

    # Preview behavior when files are missing.
    # Set both to False for final true-data analysis.
    "SIMULATE_MISSING_METHODS": True,
    "SIMULATE_NOVELTY_WITHOUT_TRAINING": True,

    # Panel styles.
    # Panel A: "dot" or "bar"
    # Panel C: "dot" or "stacked"
    "PANEL_A_STYLE": "dot",
    "PANEL_C_STYLE": "stacked",

    # Output.
    "OUTPUT_DIR": './experiments/scaffold_statistics/scaffold_panels_output',
    "SAVE_FORMATS": ["png", "pdf"],
    "DPI": 600,

    # Figure sizing.
    "FIGSIZE_A": (7.2, 4.4),
    "FIGSIZE_B": (7.2, 5.0),
    "FIGSIZE_C": (7.4, 4.6),

    # Y-axis range for Panel A.
    # Use (95, 100.1) to zoom in; use (0, 100) for conservative full-scale view.
    "PANEL_A_YLIM": (95.5, 100.1),

    # Colors: restrained publication-style palette.
    "COLORS": {
        "AAE": "#8F8F8F",
        "VAE": "#B5B5B5",
        "CharRNN": "#C7935D",
        "PepINVENT_30": "#A9D3C5",
        "PepINVENT_50": "#7BBBA7",
        "PepINVENT_70": "#57A691",
        "PepINVENT_80": "#3E8D79",
        "CycGPT_LoRA": "#2F5DA8",
        "Murcko": "#4F81C7",
        "Generic": "#A8C4E8",
        "Seen": "#D9D9D9",
        "Novel": "#4F81C7",
    },

    # Whether to print preview notes on the figure.
    # Turn this off for final manuscript export.
    "ADD_PREVIEW_NOTE": True,
}


# =============================
# 2. Utilities
# =============================

def setup_style():
    """Matplotlib style tuned for clean manuscript figures."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def format_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def read_csv_robust(path: str) -> pd.DataFrame:
    """Read CSV using common encodings."""
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1")


def find_smiles_column(df: pd.DataFrame, candidates) -> str:
    """Find SMILES column from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise ValueError(f"No SMILES column found. Available columns: {list(df.columns)}")

def normalize_cyclization_value(x: str) -> str:
    """Normalize cyclization labels such as head-to-tail and head_to_tail."""
    return str(x).strip().lower().replace("-", "_")


def filter_by_cyclization(
    df: pd.DataFrame,
    cyclization_column: str,
    target_cyclization: str,
) -> pd.DataFrame:
    """Filter dataframe by cyclization type."""
    if cyclization_column not in df.columns:
        raise ValueError(
            f"Cyclization column '{cyclization_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    target = normalize_cyclization_value(target_cyclization)

    mask = (
        df[cyclization_column]
        .dropna()
        .astype(str)
        .map(normalize_cyclization_value)
        .eq(target)
    )

    filtered = df.loc[mask].copy()

    if filtered.empty:
        available = sorted(
            df[cyclization_column]
            .dropna()
            .astype(str)
            .map(normalize_cyclization_value)
            .unique()
        )
        raise ValueError(
            f"No rows found for cyclization type '{target_cyclization}'. "
            f"Available normalized values: {available}"
        )

    print(
        f"Filtered training set by {cyclization_column} = {target}: "
        f"{len(filtered):,}/{len(df):,} rows retained"
    )

    return filtered


def cache_key(method: str, path: str, sample_size, seed: int, extra_key: str = "") -> str:
    """Make a stable cache key."""
    raw = f"{method}|{path}|{sample_size}|{seed}|{extra_key}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def scaffold_from_smiles(smiles: str) -> Tuple[Optional[str], Optional[str], bool]:
    """Return Murcko and generic scaffold SMILES from a molecule SMILES."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, None, False

        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return "", "", True

        murcko = Chem.MolToSmiles(scaf, canonical=True)
        generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaf)
        generic = Chem.MolToSmiles(generic_mol, canonical=True)
        return murcko, generic, True
    except Exception:
        return None, None, False


def compute_scaffolds_for_file(
    method: str,
    csv_path: str,
    smiles_columns,
    sample_size=None,
    seed: int = 42,
    cache_dir: str = "./scaffold_cache",
    rebuild_cache: bool = False,
    cyclization_column: Optional[str] = None,
    target_cyclization: Optional[str] = None,
) -> pd.DataFrame:
    """Compute Murcko and generic scaffolds for one CSV."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    extra_key = ""
    if cyclization_column is not None and target_cyclization is not None:
        extra_key = f"{cyclization_column}={normalize_cyclization_value(target_cyclization)}"

    key = cache_key(method, csv_path, sample_size, seed, extra_key=extra_key)
    cache_file = cache_dir / f"{method}_{key}_scaffolds.csv"

    if cache_file.exists() and not rebuild_cache:
        return pd.read_csv(cache_file)

    df = read_csv_robust(csv_path)

    if cyclization_column is not None and target_cyclization is not None:
        df = filter_by_cyclization(
            df=df,
            cyclization_column=cyclization_column,
            target_cyclization=target_cyclization,
        )

    smi_col = find_smiles_column(df, smiles_columns)
    smi = df[smi_col].dropna().astype(str).drop_duplicates()

    if sample_size is not None and len(smi) > sample_size:
        smi = smi.sample(n=sample_size, random_state=seed)

    rows = []
    total = len(smi)
    for i, s in enumerate(smi, start=1):
        murcko, generic, valid = scaffold_from_smiles(s)
        if valid:
            rows.append({
                "method": method,
                "smiles": s,
                "murcko_scaffold": murcko,
                "generic_scaffold": generic,
            })
        if i % 1000 == 0:
            print(f"{method}: processed {i:,}/{total:,} SMILES")

    out = pd.DataFrame(rows)
    out.to_csv(cache_file, index=False)
    print(f"Saved cache: {cache_file}")
    return out


def scaffold_stats(scaf_df: pd.DataFrame) -> Dict:
    """Compute scaffold diversity statistics."""
    n_mol = len(scaf_df)
    murcko_counts = scaf_df["murcko_scaffold"].value_counts()
    generic_counts = scaf_df["generic_scaffold"].value_counts()

    def one_level(counts):
        n_unique = len(counts)
        n_singletons = int((counts == 1).sum())
        top10 = int(counts.head(10).sum()) if n_unique else 0
        return {
            "n_unique": n_unique,
            "unique_fraction": n_unique / n_mol if n_mol else np.nan,
            "singleton_fraction": n_singletons / n_unique if n_unique else np.nan,
            "top10_coverage": top10 / n_mol if n_mol else np.nan,
            "rank_counts": counts.values.astype(int),
        }

    return {
        "n_molecules": n_mol,
        "murcko": one_level(murcko_counts),
        "generic": one_level(generic_counts),
    }


def generate_rank_counts(n_molecules, n_unique_target, alpha, seed):
    """Generate placeholder rank-frequency data for preview."""
    rng = np.random.default_rng(seed)
    n_unique_target = max(10, int(n_unique_target))
    ranks = np.arange(1, n_unique_target + 1)
    probs = 1.0 / np.power(ranks, alpha)
    probs = probs / probs.sum()
    counts = rng.multinomial(n_molecules, probs)
    counts = counts[counts > 0]
    return np.sort(counts)[::-1]


def make_simulated_stats(methods, reference_n=6000):
    """Placeholder diversity/novelty values for layout preview."""
    diversity = {
        "AAE":          {"murcko": 0.992, "generic": 0.985},
        "VAE":          {"murcko": 0.988, "generic": 0.975},
        "CharRNN":      {"murcko": 0.981, "generic": 0.962},
        "PepINVENT_30": {"murcko": 0.991, "generic": 0.978},
        "PepINVENT_50": {"murcko": 0.993, "generic": 0.981},
        "PepINVENT_70": {"murcko": 0.994, "generic": 0.983},
        "PepINVENT_80": {"murcko": 0.995, "generic": 0.985},
        "CycGPT_LoRA":  {"murcko": 0.996, "generic": 0.987},
    }
    novelty = {
        "AAE":          {"murcko": 0.73, "generic": 0.61},
        "VAE":          {"murcko": 0.75, "generic": 0.63},
        "CharRNN":      {"murcko": 0.70, "generic": 0.58},
        "PepINVENT_30": {"murcko": 0.77, "generic": 0.65},
        "PepINVENT_50": {"murcko": 0.79, "generic": 0.67},
        "PepINVENT_70": {"murcko": 0.81, "generic": 0.69},
        "PepINVENT_80": {"murcko": 0.82, "generic": 0.70},
        "CycGPT_LoRA":  {"murcko": 0.86, "generic": 0.75},
    }

    rank_freq = {}
    alphas = {
        "AAE": 1.02,
        "VAE": 1.12,
        "CharRNN": 1.24,
        "PepINVENT_30": 1.08,
        "PepINVENT_50": 1.05,
        "PepINVENT_70": 1.03,
        "PepINVENT_80": 1.02,
        "CycGPT_LoRA": 1.00,
    }
    for i, m in enumerate(methods):
        frac = diversity.get(m, {"murcko": 0.99})["murcko"]
        rank_freq[m] = generate_rank_counts(reference_n, int(frac * reference_n), alphas.get(m, 1.05), i + 1)
    return diversity, novelty, rank_freq


def scaffold_novelty(generated: pd.DataFrame, training: pd.DataFrame) -> Dict[str, float]:
    """Compute scaffold novelty relative to training scaffold sets."""

    gen_m = set(generated["murcko_scaffold"].dropna().astype(str))
    gen_g = set(generated["generic_scaffold"].dropna().astype(str))
    train_m = set(training["murcko_scaffold"].dropna().astype(str))
    train_g = set(training["generic_scaffold"].dropna().astype(str))

    # Remove empty scaffold strings if present
    gen_m.discard("")
    gen_g.discard("")
    train_m.discard("")
    train_g.discard("")

    m_novel = len(gen_m - train_m) / len(gen_m) if len(gen_m) > 0 else np.nan
    g_novel = len(gen_g - train_g) / len(gen_g) if len(gen_g) > 0 else np.nan

    return {
        "murcko": m_novel,
        "generic": g_novel,
    }


# =============================
# 3. Load data and compute stats
# =============================

def load_all_results(config):
    methods = list(config["METHOD_FILES"].keys())
    method_tables = {}
    diversity = {}
    rank_freq = {}
    preview_notes = []

    for method, path in config["METHOD_FILES"].items():
        if path is not None and os.path.exists(path):
            print(f"\nComputing scaffolds for {method}: {path}")
            tab = compute_scaffolds_for_file(
                method=method,
                csv_path=path,
                smiles_columns=config["SMILES_COLUMNS"],
                sample_size=config["SAMPLE_SIZE_PER_METHOD"],
                seed=config["RANDOM_SEED"],
                cache_dir=config["CACHE_DIR"],
                rebuild_cache=config["REBUILD_CACHE"],
            )
            method_tables[method] = tab
            st = scaffold_stats(tab)
            diversity[method] = {
                "murcko": st["murcko"]["unique_fraction"],
                "generic": st["generic"]["unique_fraction"],
            }
            rank_freq[method] = st["murcko"]["rank_counts"]
        else:
            preview_notes.append(method)

    # Fill missing methods with simulated placeholders if enabled.
    if preview_notes and config["SIMULATE_MISSING_METHODS"]:
        print("\nSimulating missing methods for layout preview:", preview_notes)
        sim_div, sim_nov, sim_rank = make_simulated_stats(methods, reference_n=config["SAMPLE_SIZE_PER_METHOD"] or 6000)
        for method in preview_notes:
            diversity[method] = sim_div[method]
            rank_freq[method] = sim_rank[method]
    elif preview_notes:
        methods = [m for m in methods if m not in preview_notes]

    # Novelty.
    novelty = {}
    train_path = config.get("TRAIN_FILE")
    if train_path is not None and os.path.exists(train_path):
        print(f"\nComputing training scaffolds: {train_path}")
        train_cyclization_column = None
        train_cyclization_type = None

        if config.get("FILTER_TRAIN_BY_CYCLIZATION", False):
            train_cyclization_column = config.get("TRAIN_CYCLIZATION_COLUMN", "Cyclization")
            train_cyclization_type = config.get("TRAIN_CYCLIZATION_TYPE", None)

            if train_cyclization_type is None:
                raise ValueError(
                    "FILTER_TRAIN_BY_CYCLIZATION=True, but TRAIN_CYCLIZATION_TYPE is not set."
                )

        train_tab = compute_scaffolds_for_file(
            method="TRAIN",
            csv_path=train_path,
            smiles_columns=config["SMILES_COLUMNS"],
            sample_size=config["SAMPLE_SIZE_TRAIN"],
            seed=config["RANDOM_SEED"],
            cache_dir=config["CACHE_DIR"],
            rebuild_cache=config["REBUILD_CACHE"],
            cyclization_column=train_cyclization_column,
            target_cyclization=train_cyclization_type,
        )
        for method in methods:
            if method in method_tables:
                novelty[method] = scaffold_novelty(method_tables[method], train_tab)
            else:
                print("!!!!! nan exist !!!!!")
                novelty[method] = {"murcko": np.nan, "generic": np.nan}
    else:
        if config["SIMULATE_NOVELTY_WITHOUT_TRAINING"]:
            print("\nTraining file not provided. Simulating novelty for layout preview.")
            _, sim_nov, _ = make_simulated_stats(methods, reference_n=config["SAMPLE_SIZE_PER_METHOD"] or 6000)
            novelty = {m: sim_nov[m] for m in methods}
        else:
            novelty = {m: {"murcko": np.nan, "generic": np.nan} for m in methods}

    return methods, diversity, rank_freq, novelty, preview_notes


# =============================
# 4. Plotting functions
# =============================

def save_figure(fig, out_base: str, config):
    out_dir = Path(config["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for fmt in config["SAVE_FORMATS"]:
        out_path = out_dir / f"{out_base}.{fmt}"
        if fmt.lower() == "png":
            fig.savefig(out_path, dpi=config["DPI"])
        else:
            fig.savefig(out_path)
        saved.append(str(out_path))
    return saved


def plot_panel_a(methods, diversity, config):
    """Panel A: scaffold diversity across methods."""
    colors = config["COLORS"]
    setup_style()

    x = np.arange(len(methods))
    murcko_vals = np.array([diversity[m]["murcko"] * 100 for m in methods])
    generic_vals = np.array([diversity[m]["generic"] * 100 for m in methods])

    fig, ax = plt.subplots(figsize=config["FIGSIZE_A"])

    if config["PANEL_A_STYLE"] == "bar":
        w = 0.36
        ax.bar(x - w/2, murcko_vals, width=w, color=colors["Murcko"], edgecolor="black", linewidth=0.5, label="Murcko scaffold")
        ax.bar(x + w/2, generic_vals, width=w, color=colors["Generic"], edgecolor="black", linewidth=0.5, label="Generic scaffold")
    else:
        for i, m in enumerate(methods):
            ax.plot([x[i] - 0.12, x[i] + 0.12], [murcko_vals[i], generic_vals[i]],
                    linewidth=1.1, color="#C7C7C7", zorder=1)
            ax.scatter(x[i] - 0.12, murcko_vals[i], s=42, color=colors["Murcko"],
                       edgecolor="black", linewidth=0.5, zorder=3)
            ax.scatter(x[i] + 0.12, generic_vals[i], s=42, color=colors["Generic"],
                       edgecolor="black", linewidth=0.5, zorder=3)

    if "CycGPT_LoRA" in methods:
        idx = methods.index("CycGPT_LoRA")
        ax.scatter(x[idx] - 0.12, murcko_vals[idx], s=72, facecolor="none",
                   edgecolor=colors["CycGPT_LoRA"], linewidth=1.2, zorder=4)
        ax.scatter(x[idx] + 0.12, generic_vals[idx], s=72, facecolor="none",
                   edgecolor=colors["CycGPT_LoRA"], linewidth=1.2, zorder=4)

    ax.set_ylabel("Unique scaffold fraction (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=32, ha="right")
    ax.set_ylim(*config["PANEL_A_YLIM"])
    ax.set_title("a  Scaffold diversity across methods", loc="left", fontweight="bold")
    format_ax(ax)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6.5,
               markerfacecolor=colors["Murcko"], markeredgecolor="black", markeredgewidth=0.5,
               label="Murcko scaffold"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6.5,
               markerfacecolor=colors["Generic"], markeredgecolor="black", markeredgewidth=0.5,
               label="Generic scaffold"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left")

    if config["ADD_PREVIEW_NOTE"]:
        ax.text(0.995, 0.02, "Preview mode: replace placeholder files for final analysis.",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color="#555555")

    fig.tight_layout()
    return fig


def plot_panel_b(methods, rank_freq, config):
    """Panel B: scaffold rank-frequency curves."""
    colors = config["COLORS"]
    setup_style()

    fig, ax = plt.subplots(figsize=config["FIGSIZE_B"])

    for m in methods:
        counts = rank_freq[m]
        ranks = np.arange(1, len(counts) + 1)
        lw = 2.1 if m == "CycGPT_LoRA" else 1.5
        color = colors.get(m, "#666666")
        ax.plot(ranks, counts, linewidth=lw, color=color, label=m)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Murcko scaffold rank")
    ax.set_ylabel("Frequency")
    ax.set_title("b  Scaffold rank–frequency distribution", loc="left", fontweight="bold")
    format_ax(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    if config["ADD_PREVIEW_NOTE"]:
        ax.text(0.995, 0.02, "Preview mode: simulated curves are used for missing methods.",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color="#555555")

    fig.tight_layout()
    return fig


def plot_panel_c(methods, novelty, config):
    """Panel C: scaffold novelty relative to training set."""
    colors = config["COLORS"]
    setup_style()

    x = np.arange(len(methods))
    murcko_vals = np.array([novelty[m]["murcko"] * 100 for m in methods])
    generic_vals = np.array([novelty[m]["generic"] * 100 for m in methods])

    fig, ax = plt.subplots(figsize=config["FIGSIZE_C"])

    if config["PANEL_C_STYLE"] == "stacked":
        w = 0.30
        seen_m = 100 - murcko_vals
        seen_g = 100 - generic_vals

        ax.bar(x - w/2, seen_m, width=w, color=colors["Seen"], edgecolor="black", linewidth=0.5)
        ax.bar(x - w/2, murcko_vals, width=w, bottom=seen_m, color=colors["Murcko"], edgecolor="black", linewidth=0.5)

        ax.bar(x + w/2, seen_g, width=w, color="#EFEFEF", edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, generic_vals, width=w, bottom=seen_g, color=colors["Generic"], edgecolor="black", linewidth=0.5)

        for i in range(len(methods)):
            ax.text(x[i] - w/2, 101.2, "M", ha="center", va="bottom", fontsize=7.8)
            ax.text(x[i] + w/2, 101.2, "G", ha="center", va="bottom", fontsize=7.8)

        ax.text(0.01, 1.01, "M, Murcko scaffold; G, generic scaffold",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="#444444")

        legend_handles = [
            Line2D([0], [0], color=colors["Seen"], linewidth=8, label="Seen in training set"),
            Line2D([0], [0], color=colors["Novel"], linewidth=8, label="Novel to training set"),
        ]
        ax.legend(handles=legend_handles, frameon=False, loc="upper left")
        ax.set_ylim(0, 104)
        ax.set_ylabel("Scaffold fraction (%)")

    else:
        # Recommended simpler style: direct novelty fraction.
        for i, m in enumerate(methods):
            ax.plot([x[i] - 0.12, x[i] + 0.12], [murcko_vals[i], generic_vals[i]],
                    linewidth=1.1, color="#C7C7C7", zorder=1)
            ax.scatter(x[i] - 0.12, murcko_vals[i], s=42, color=colors["Murcko"],
                       edgecolor="black", linewidth=0.5, zorder=3)
            ax.scatter(x[i] + 0.12, generic_vals[i], s=42, color=colors["Generic"],
                       edgecolor="black", linewidth=0.5, zorder=3)

        if "CycGPT_LoRA" in methods:
            idx = methods.index("CycGPT_LoRA")
            ax.scatter(x[idx] - 0.12, murcko_vals[idx], s=72, facecolor="none",
                       edgecolor=colors["CycGPT_LoRA"], linewidth=1.2, zorder=4)
            ax.scatter(x[idx] + 0.12, generic_vals[idx], s=72, facecolor="none",
                       edgecolor=colors["CycGPT_LoRA"], linewidth=1.2, zorder=4)

        handles = [
            Line2D([0], [0], marker="o", linestyle="None", markersize=6.5,
                   markerfacecolor=colors["Murcko"], markeredgecolor="black", markeredgewidth=0.5,
                   label="Murcko scaffold"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=6.5,
                   markerfacecolor=colors["Generic"], markeredgecolor="black", markeredgewidth=0.5,
                   label="Generic scaffold"),
        ]
        ax.legend(handles=handles, frameon=False, loc="lower left")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Novel scaffold fraction (%)")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=32, ha="right")
    ax.set_title("c  Scaffold novelty relative to training set", loc="left", fontweight="bold")
    format_ax(ax)

    if config["ADD_PREVIEW_NOTE"]:
        ax.text(0.995, 0.02, "Preview mode: true novelty requires TRAIN_FILE.",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color="#555555")

    fig.tight_layout()
    return fig


# =============================
# 5. Main
# =============================

def main():
    methods, diversity, rank_freq, novelty, preview_notes = load_all_results(CONFIG)

    output_dir = Path(CONFIG["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary table for checking values.
    summary_rows = []
    for m in methods:
        summary_rows.append({
            "method": m,
            "unique_murcko_fraction": diversity[m]["murcko"],
            "unique_generic_fraction": diversity[m]["generic"],
            "murcko_scaffold_novelty": novelty[m]["murcko"],
            "generic_scaffold_novelty": novelty[m]["generic"],
            "n_rank_scaffolds": len(rank_freq[m]),
            "top1_frequency": int(rank_freq[m][0]) if len(rank_freq[m]) else np.nan,
        })
    if CONFIG.get("FILTER_TRAIN_BY_CYCLIZATION", False):
        print(
            f"\nTraining scaffold library filtered by "
            f"{CONFIG['TRAIN_CYCLIZATION_COLUMN']} = {CONFIG['TRAIN_CYCLIZATION_TYPE']}"
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "scaffold_panel_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\nPanel summary:")
    print(summary_df)
    print(f"\nSaved summary: {summary_csv}")

    fig_a = plot_panel_a(methods, diversity, CONFIG)
    saved_a = save_figure(fig_a, "panelA_scaffold_diversity", CONFIG)
    plt.show()

    fig_b = plot_panel_b(methods, rank_freq, CONFIG)
    saved_b = save_figure(fig_b, "panelB_scaffold_rank_frequency", CONFIG)
    plt.show()

    fig_c = plot_panel_c(methods, novelty, CONFIG)
    saved_c = save_figure(fig_c, "panelC_scaffold_novelty", CONFIG)
    plt.show()

    print("\nSaved figures:")
    for p in saved_a + saved_b + saved_c:
        print(p)

    if preview_notes:
        print("\nPreview warning:")
        print("The following methods used simulated placeholders because METHOD_FILES are missing:")
        print(", ".join(preview_notes))
    if CONFIG["TRAIN_FILE"] is None:
        print("\nNovelty warning:")
        print("TRAIN_FILE is not provided. Panel C is simulated unless SIMULATE_NOVELTY_WITHOUT_TRAINING=False.")


if __name__ == "__main__":
    main()
