#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from xml.sax.saxutils import escape


TEST2_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = TEST2_ROOT / "output" / "generated" / "lora" / "gen_results_head_to_tail_30000.csv"
DEFAULT_OUTPUT_ROOT = Path('results/head_to_tail_scaffold_statistics_20260412')
TARGET_CYCLIZATION = "head_to_tail"
TOP_N = 20
TOP_DRAW_N = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute scaffold statistics for generated head-to-tail cyclic peptides.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cyclization-type", type=str, default=TARGET_CYCLIZATION)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--top-draw-n", type=int, default=TOP_DRAW_N)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_length_module() -> Any:
    length_script = TEST2_ROOT / "evaluation" / "analysis" / "compute_peptide_lengths.py"
    spec = importlib.util.spec_from_file_location("compute_peptide_lengths", length_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load canonical length estimator from {length_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def detect_column(df: pd.DataFrame, candidates: list[str], label: str) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if label == "valid":
        return None
    raise ValueError(f"Could not find {label} column in {list(df.columns)}")


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def infer_length(module: Any, type_oracle: Any, smiles: str) -> int | None:
    info = module.infer_length_from_smiles(smiles, type_oracle)
    return info["peptide_length"] if info["valid_smiles"] else None


def murcko_scaffold_info(mol: Chem.Mol) -> tuple[str, str, int, int]:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return "", "", 0, 0
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)
    generic_smiles = Chem.MolToSmiles(generic, canonical=True, isomericSmiles=True) if generic.GetNumAtoms() else ""
    return scaffold_smiles, generic_smiles, int(scaffold.GetNumAtoms()), int(scaffold.GetRingInfo().NumRings())


def build_scaffold_table(
    dataframe: pd.DataFrame,
    *,
    source_csv: Path,
    cyclization_type: str,
    length_module: Any,
    type_oracle: Any,
) -> tuple[pd.DataFrame, dict[str, int]]:
    smiles_col = detect_column(dataframe, ["SMILES", "Smiles", "smiles"], "SMILES")
    valid_col = detect_column(dataframe, ["VALID", "Valid", "valid", "is_valid"], "valid")
    cyclization_col = detect_column(
        dataframe,
        ["Cyclization", "cyclization", "condition_type"],
        "cyclization",
    )
    canonical_col = "canonical_smiles" if "canonical_smiles" in dataframe.columns else None
    peptide_length_col = "peptide_length" if "peptide_length" in dataframe.columns else None

    stats = {
        "input_rows": int(len(dataframe)),
        "valid_flag_rows": 0,
        "cyclization_match_rows": 0,
        "rdkit_valid_rows": 0,
        "length_valid_rows": 0,
        "unique_canonical_rows": 0,
        "duplicate_canonical_rows": 0,
        "nonempty_murcko_rows": 0,
    }

    rows: list[dict[str, Any]] = []
    seen_canonical: set[str] = set()
    for row_index, row in dataframe.iterrows():
        smiles = str(row[smiles_col]).strip()
        if not smiles or smiles.lower() == "nan":
            continue

        valid_flag = True if valid_col is None else coerce_bool(row[valid_col])
        if valid_flag:
            stats["valid_flag_rows"] += 1
        else:
            continue

        if cyclization_col is not None:
            cyclization_value = str(row[cyclization_col])
            if cyclization_value != cyclization_type:
                continue
            stats["cyclization_match_rows"] += 1

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        stats["rdkit_valid_rows"] += 1

        if canonical_col is not None and pd.notna(row[canonical_col]) and str(row[canonical_col]).strip():
            canonical_smiles = str(row[canonical_col]).strip()
        else:
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if canonical_smiles in seen_canonical:
            stats["duplicate_canonical_rows"] += 1
            continue
        seen_canonical.add(canonical_smiles)
        stats["unique_canonical_rows"] += 1

        if peptide_length_col is not None:
            peptide_length = pd.to_numeric(pd.Series([row[peptide_length_col]]), errors="coerce").iloc[0]
            if pd.isna(peptide_length):
                continue
            peptide_length = int(peptide_length)
        else:
            peptide_length = infer_length(length_module, type_oracle, canonical_smiles)
            if peptide_length is None:
                continue
        stats["length_valid_rows"] += 1

        murcko_smiles, generic_smiles, murcko_atom_count, murcko_ring_count = murcko_scaffold_info(mol)
        if murcko_smiles:
            stats["nonempty_murcko_rows"] += 1

        rows.append(
            {
                "source_row_index": int(row_index),
                "source_csv": str(source_csv),
                "SMILES": smiles,
                "canonical_smiles": canonical_smiles,
                "length": int(peptide_length),
                "murcko_scaffold": murcko_smiles,
                "generic_scaffold": generic_smiles,
                "murcko_atom_count": int(murcko_atom_count),
                "murcko_ring_count": int(murcko_ring_count),
            }
        )

    return pd.DataFrame(rows), stats


def summarize_scaffold_counter(counter: Counter[str], *, top_n: int) -> pd.DataFrame:
    total = sum(counter.values())
    rows = []
    cumulative = 0
    for rank, (scaffold, count) in enumerate(counter.most_common(top_n), start=1):
        cumulative += count
        rows.append(
            {
                "rank": rank,
                "scaffold": scaffold,
                "count": int(count),
                "fraction": float(count / total) if total else 0.0,
                "cumulative_fraction": float(cumulative / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_occurrence_histogram_table(counter: Counter[str]) -> pd.DataFrame:
    occurrence_counter = Counter(counter.values())
    total_scaffolds = len(counter)
    cumulative = 0
    rows = []
    for occurrence in sorted(occurrence_counter):
        scaffold_count = int(occurrence_counter[occurrence])
        cumulative += scaffold_count
        rows.append(
            {
                "occurrence": int(occurrence),
                "scaffold_count": scaffold_count,
                "fraction_of_scaffolds": float(scaffold_count / total_scaffolds) if total_scaffolds else 0.0,
                "cumulative_fraction_of_scaffolds": float(cumulative / total_scaffolds) if total_scaffolds else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_rank_frequency_table(counter: Counter[str]) -> pd.DataFrame:
    total = sum(counter.values())
    cumulative = 0
    rows = []
    for rank, (scaffold, count) in enumerate(counter.most_common(), start=1):
        cumulative += count
        rows.append(
            {
                "rank": int(rank),
                "scaffold": scaffold,
                "count": int(count),
                "fraction": float(count / total) if total else 0.0,
                "cumulative_fraction": float(cumulative / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_length_scaffold_summary(scaffold_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for length, group in scaffold_table.groupby("length", sort=True):
        murcko_nonempty = group[group["murcko_scaffold"] != ""]
        rows.append(
            {
                "length": int(length),
                "molecule_count": int(len(group)),
                "unique_murcko_scaffold_count": int(murcko_nonempty["murcko_scaffold"].nunique()),
                "unique_generic_scaffold_count": int(group[group["generic_scaffold"] != ""]["generic_scaffold"].nunique()),
                "murcko_singleton_count": int((murcko_nonempty["murcko_scaffold"].value_counts() == 1).sum()),
                "murcko_empty_count": int((group["murcko_scaffold"] == "").sum()),
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_top_scaffold_bars(top_df: pd.DataFrame, *, value_label: str, output_dir: Path, stem: str) -> None:
    if top_df.empty:
        return
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    x = np.arange(len(top_df))
    ax.bar(x, top_df["count"].to_numpy(), color="#0072B2", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{rank}" for rank in top_df["rank"]], rotation=0)
    ax.set_xlabel("Scaffold rank")
    ax.set_ylabel("Count")
    ax.set_title(value_label)
    ax.grid(False)
    save_figure(fig, output_dir, stem)


def plot_occurrence_histogram(
    occurrence_df: pd.DataFrame,
    *,
    title: str,
    output_dir: Path,
    stem: str,
) -> None:
    if occurrence_df.empty:
        return
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    x = occurrence_df["occurrence"].to_numpy(dtype=int)
    y = occurrence_df["scaffold_count"].to_numpy(dtype=int)
    ax.bar(x, y, color="#0072B2", alpha=0.85, width=0.9)
    ax.set_xlabel("Scaffold occurrence count")
    ax.set_ylabel("Number of scaffolds")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(False)
    if len(x) > 12:
        ticks = np.unique(np.linspace(x.min(), x.max(), num=10, dtype=int))
        ax.set_xticks(ticks)
    save_figure(fig, output_dir, stem)


def plot_rank_frequency_curve(
    rank_df: pd.DataFrame,
    *,
    title: str,
    output_dir: Path,
    stem: str,
    log_scale: bool = False,
) -> None:
    if rank_df.empty:
        return
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    x = rank_df["rank"].to_numpy(dtype=int)
    y = rank_df["count"].to_numpy(dtype=int)
    ax.plot(x, y, color="#0072B2", linewidth=1.8)
    ax.set_xlabel("Scaffold rank")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        top_n = min(100, len(rank_df))
        if top_n >= 2:
            inset = inset_axes(
                ax,
                width="40%",
                height="52%",
                loc="center right",
                borderpad=1.2,
            )
            inset_x = x[:top_n]
            inset_y = y[:top_n]
            inset.plot(inset_x, inset_y, color="#D55E00", linewidth=1.5)
            inset.set_xlim(1, int(inset_x[-1]))
            inset.set_xlabel("Top rank", fontsize=6)
            inset.set_ylabel("Freq.", fontsize=6)
            inset.set_title("Top 100", fontsize=7)
            inset.tick_params(axis="both", labelsize=6, length=2)
            inset.grid(False)
            for spine in inset.spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor("#444444")
    ax.grid(False)
    save_figure(fig, output_dir, stem)


def scaffold_tile_svg(scaffold_smiles: str, *, width: int, height: int) -> str:
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        raise ValueError(f"Invalid scaffold SMILES: {scaffold_smiles}")
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.continuousHighlight = False
    opts.bondLineWidth = 1.4
    opts.minFontSize = 10
    opts.maxFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", "", 1)


def extract_svg_body(svg_text: str) -> str:
    start = svg_text.find(">")
    end = svg_text.rfind("</svg>")
    if start < 0 or end < 0:
        raise ValueError("Malformed SVG text.")
    return svg_text[start + 1 : end]


def build_top_scaffold_grid_svg(top_df: pd.DataFrame, output_path: Path) -> None:
    if top_df.empty:
        return
    width = 1800
    tile_w = 420
    tile_h = 250
    cols = 3
    rows = int(np.ceil(len(top_df) / cols))
    height = rows * 320 + 40
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='white' />",
    ]
    for idx, row in top_df.iterrows():
        r = idx // cols
        c = idx % cols
        x = 40 + c * 560
        y = 30 + r * 320
        body = extract_svg_body(scaffold_tile_svg(row["scaffold"], width=tile_w, height=tile_h))
        parts.append(
            f"<g transform='translate({x},{y})'>"
            f"<rect x='0' y='0' width='{tile_w}' height='{tile_h}' fill='none' stroke='#DDDDDD' stroke-width='1' rx='8' ry='8' />"
            f"<svg x='0' y='0' width='{tile_w}' height='{tile_h}' viewBox='0 0 {tile_w} {tile_h}'>{body}</svg>"
            f"<text x='{tile_w/2:.1f}' y='{tile_h + 26}' text-anchor='middle' font-family='Arial, Helvetica, sans-serif' font-size='20' font-weight='700' fill='#222222'>#{int(row['rank'])} | count={int(row['count'])}</text>"
            f"<text x='{tile_w/2:.1f}' y='{tile_h + 52}' text-anchor='middle' font-family='Arial, Helvetica, sans-serif' font-size='14' fill='#555555'>{escape(str(row['scaffold'])[:80])}</text>"
            f"</g>"
        )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def export_svg_to_pdf_png(svg_path: Path) -> None:
    import cairosvg

    cairosvg.svg2pdf(url=str(svg_path), write_to=str(svg_path.with_suffix(".pdf")), dpi=300)
    cairosvg.svg2png(url=str(svg_path), write_to=str(svg_path.with_suffix(".png")), dpi=300)


def main() -> None:
    args = parse_args()
    if not args.force and (args.output_root / "summary" / "scaffold_summary.json").exists():
        print(json.dumps({"status": "already_exists", "output_root": str(args.output_root)}, indent=2))
        return

    dataframe = pd.read_csv(args.input_csv)
    if "peptide_length" in dataframe.columns:
        length_module = None
        type_oracle = None
    else:
        length_module = load_length_module()
        type_oracle = length_module.load_type_oracle(TEST2_ROOT)
    scaffold_table, load_stats = build_scaffold_table(
        dataframe,
        source_csv=args.input_csv,
        cyclization_type=args.cyclization_type,
        length_module=length_module,
        type_oracle=type_oracle,
    )
    if scaffold_table.empty:
        raise ValueError(f"No valid scaffold rows could be produced from {args.input_csv}")

    murcko_counter = Counter(scaffold for scaffold in scaffold_table["murcko_scaffold"] if scaffold)
    generic_counter = Counter(scaffold for scaffold in scaffold_table["generic_scaffold"] if scaffold)

    top_murcko_df = summarize_scaffold_counter(murcko_counter, top_n=args.top_n)
    top_generic_df = summarize_scaffold_counter(generic_counter, top_n=args.top_n)
    murcko_occurrence_df = build_occurrence_histogram_table(murcko_counter)
    generic_occurrence_df = build_occurrence_histogram_table(generic_counter)
    murcko_rank_df = build_rank_frequency_table(murcko_counter)
    generic_rank_df = build_rank_frequency_table(generic_counter)
    length_summary_df = build_length_scaffold_summary(scaffold_table)

    analysis_dir = args.output_root / "analysis"
    summary_dir = args.output_root / "summary"
    figures_dir = args.output_root / "figures"

    write_csv(scaffold_table, analysis_dir / "scaffold_table.csv")
    write_csv(top_murcko_df, summary_dir / "top_murcko_scaffolds.csv")
    write_csv(top_generic_df, summary_dir / "top_generic_scaffolds.csv")
    write_csv(murcko_occurrence_df, summary_dir / "murcko_occurrence_histogram.csv")
    write_csv(generic_occurrence_df, summary_dir / "generic_occurrence_histogram.csv")
    write_csv(murcko_rank_df, summary_dir / "murcko_rank_frequency.csv")
    write_csv(generic_rank_df, summary_dir / "generic_rank_frequency.csv")
    write_csv(length_summary_df, summary_dir / "length_scaffold_summary.csv")

    top_murcko_draw_df = top_murcko_df.head(args.top_draw_n).copy()
    murcko_grid_svg_path = figures_dir / "top_murcko_scaffolds_grid.svg"
    build_top_scaffold_grid_svg(top_murcko_draw_df, murcko_grid_svg_path)
    export_svg_to_pdf_png(murcko_grid_svg_path)

    top_generic_draw_df = top_generic_df.head(args.top_draw_n).copy()
    generic_grid_svg_path = figures_dir / "top_generic_scaffolds_grid.svg"
    build_top_scaffold_grid_svg(top_generic_draw_df, generic_grid_svg_path)
    export_svg_to_pdf_png(generic_grid_svg_path)

    plot_top_scaffold_bars(
        top_murcko_df.head(min(args.top_n, 20)),
        value_label="Top Murcko scaffolds in generated head-to-tail peptides",
        output_dir=figures_dir,
        stem="top_murcko_scaffolds_bar",
    )
    plot_top_scaffold_bars(
        top_generic_df.head(min(args.top_n, 20)),
        value_label="Top generic Murcko scaffolds in generated head-to-tail peptides",
        output_dir=figures_dir,
        stem="top_generic_scaffolds_bar",
    )
    plot_occurrence_histogram(
        murcko_occurrence_df,
        title="Murcko scaffold occurrence histogram",
        output_dir=figures_dir,
        stem="murcko_occurrence_histogram",
    )
    plot_occurrence_histogram(
        generic_occurrence_df,
        title="Generic scaffold occurrence histogram",
        output_dir=figures_dir,
        stem="generic_occurrence_histogram",
    )
    plot_rank_frequency_curve(
        murcko_rank_df,
        title="Murcko scaffold rank-frequency curve",
        output_dir=figures_dir,
        stem="murcko_rank_frequency",
        log_scale=False,
    )
    plot_rank_frequency_curve(
        murcko_rank_df,
        title="Murcko scaffold rank-frequency curve (log-log)",
        output_dir=figures_dir,
        stem="murcko_rank_frequency_loglog",
        log_scale=True,
    )
    plot_rank_frequency_curve(
        generic_rank_df,
        title="Generic scaffold rank-frequency curve",
        output_dir=figures_dir,
        stem="generic_rank_frequency",
        log_scale=False,
    )
    plot_rank_frequency_curve(
        generic_rank_df,
        title="Generic scaffold rank-frequency curve (log-log)",
        output_dir=figures_dir,
        stem="generic_rank_frequency_loglog",
        log_scale=True,
    )

    murcko_singleton_count = int(sum(1 for count in murcko_counter.values() if count == 1))
    murcko_doubleton_count = int(sum(1 for count in murcko_counter.values() if count == 2))
    murcko_ge5_count = int(sum(1 for count in murcko_counter.values() if count >= 5))
    generic_singleton_count = int(sum(1 for count in generic_counter.values() if count == 1))
    generic_doubleton_count = int(sum(1 for count in generic_counter.values() if count == 2))
    generic_ge5_count = int(sum(1 for count in generic_counter.values() if count >= 5))

    summary = {
        "source_csv": str(args.input_csv),
        "target_cyclization_type": args.cyclization_type,
        "output_root": str(args.output_root),
        "input_rows": int(load_stats["input_rows"]),
        "valid_flag_rows": int(load_stats["valid_flag_rows"]),
        "cyclization_match_rows": int(load_stats["cyclization_match_rows"]),
        "rdkit_valid_rows": int(load_stats["rdkit_valid_rows"]),
        "length_valid_rows": int(load_stats["length_valid_rows"]),
        "unique_canonical_rows": int(load_stats["unique_canonical_rows"]),
        "duplicate_canonical_rows": int(load_stats["duplicate_canonical_rows"]),
        "nonempty_murcko_rows": int(load_stats["nonempty_murcko_rows"]),
        "unique_murcko_scaffold_count": int(len(murcko_counter)),
        "unique_generic_scaffold_count": int(len(generic_counter)),
        "murcko_singleton_count": murcko_singleton_count,
        "murcko_doubleton_count": murcko_doubleton_count,
        "murcko_ge5_count": murcko_ge5_count,
        "generic_singleton_count": generic_singleton_count,
        "generic_doubleton_count": generic_doubleton_count,
        "generic_ge5_count": generic_ge5_count,
        "murcko_singleton_fraction": float(murcko_singleton_count / len(murcko_counter)) if murcko_counter else 0.0,
        "generic_singleton_fraction": float(generic_singleton_count / len(generic_counter)) if generic_counter else 0.0,
        "top_murcko_top1_fraction": float(top_murcko_df["fraction"].iloc[0]) if not top_murcko_df.empty else 0.0,
        "top_murcko_top5_cumulative_fraction": float(top_murcko_df.head(5)["fraction"].sum()) if not top_murcko_df.empty else 0.0,
        "top_generic_top1_fraction": float(top_generic_df["fraction"].iloc[0]) if not top_generic_df.empty else 0.0,
        "top_generic_top5_cumulative_fraction": float(top_generic_df.head(5)["fraction"].sum()) if not top_generic_df.empty else 0.0,
        "length_min": int(scaffold_table["length"].min()),
        "length_max": int(scaffold_table["length"].max()),
        "length_mean": float(scaffold_table["length"].mean()),
        "top_murcko_scaffolds_csv": str(summary_dir / "top_murcko_scaffolds.csv"),
        "top_generic_scaffolds_csv": str(summary_dir / "top_generic_scaffolds.csv"),
        "murcko_occurrence_histogram_csv": str(summary_dir / "murcko_occurrence_histogram.csv"),
        "generic_occurrence_histogram_csv": str(summary_dir / "generic_occurrence_histogram.csv"),
        "murcko_rank_frequency_csv": str(summary_dir / "murcko_rank_frequency.csv"),
        "generic_rank_frequency_csv": str(summary_dir / "generic_rank_frequency.csv"),
        "length_scaffold_summary_csv": str(summary_dir / "length_scaffold_summary.csv"),
        "scaffold_table_csv": str(analysis_dir / "scaffold_table.csv"),
        "figures": {
            "top_murcko_grid_svg": str(murcko_grid_svg_path),
            "top_murcko_grid_pdf": str(murcko_grid_svg_path.with_suffix(".pdf")),
            "top_murcko_grid_png": str(murcko_grid_svg_path.with_suffix(".png")),
            "top_generic_grid_svg": str(generic_grid_svg_path),
            "top_generic_grid_pdf": str(generic_grid_svg_path.with_suffix(".pdf")),
            "top_generic_grid_png": str(generic_grid_svg_path.with_suffix(".png")),
            "top_murcko_bar_pdf": str(figures_dir / "top_murcko_scaffolds_bar.pdf"),
            "top_generic_bar_pdf": str(figures_dir / "top_generic_scaffolds_bar.pdf"),
            "murcko_occurrence_histogram_pdf": str(figures_dir / "murcko_occurrence_histogram.pdf"),
            "generic_occurrence_histogram_pdf": str(figures_dir / "generic_occurrence_histogram.pdf"),
            "murcko_rank_frequency_pdf": str(figures_dir / "murcko_rank_frequency.pdf"),
            "murcko_rank_frequency_loglog_pdf": str(figures_dir / "murcko_rank_frequency_loglog.pdf"),
            "generic_rank_frequency_pdf": str(figures_dir / "generic_rank_frequency.pdf"),
            "generic_rank_frequency_loglog_pdf": str(figures_dir / "generic_rank_frequency_loglog.pdf"),
        },
    }
    write_json(summary, summary_dir / "scaffold_summary.json")

    print(
        json.dumps(
            {
                "summary_json": str(summary_dir / "scaffold_summary.json"),
                "scaffold_table_csv": str(analysis_dir / "scaffold_table.csv"),
                "top_murcko_scaffold_count": int(len(murcko_counter)),
                "top_generic_scaffold_count": int(len(generic_counter)),
                "unique_canonical_rows": int(load_stats["unique_canonical_rows"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
