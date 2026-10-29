#!/usr/bin/env python3
"""Plot train/validation peptide length and non-natural residue distributions."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import socket
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


DEFAULT_TRAIN_ROOT = Path(
    'results/peptide_parser/restored_train_cyclic_30000_parse_seed42_20260611'
)
DEFAULT_VALIDATION_CSV = Path(
    'results/peptide_parser/restored_validation_parse_smiles2peptide_full_20260611/parsed_peptides.with_cyclization.csv'
)
DEFAULT_OUTPUT_DIR = Path('results/figures/train_validation_sequence_stats_20260612')
DEFAULT_FORMATS = ("pdf", "svg", "png", "tiff")
DEFAULT_SEED = 42

TOPOLOGY_LABELS = OrderedDict(
    [
        ("head_to_tail", "Head-to-tail"),
        ("sc_to_tail", "Side-chain-to-tail"),
        ("disulfide_bridge", "Disulfide bridge"),
    ]
)
LINEAR_LABEL = "Linear"
DATASET_ORDER = ("Train", "Validation")
DATASET_COLORS = {
    "Train": "#4C78A8",
    "Validation": "#F58518",
}
TOPOLOGY_COLORS = {
    "head_to_tail": "#8DA0CB",
    "sc_to_tail": "#7BC8BD",
    "disulfide_bridge": "#E6C65C",
    "linear": "#B8B8B8",
}
NATURAL_RESIDUE_NAMES = {
    "alanine",
    "arginine",
    "asparagine",
    "aspartate",
    "aspartic acid",
    "cysteine",
    "glutamate",
    "glutamate/glutamic acid",
    "glutamic acid",
    "glutamine",
    "glycine",
    "histidine",
    "isoleucine",
    "leucine",
    "lysine",
    "methionine",
    "phenylalanine",
    "proline",
    "serine",
    "threonine",
    "tryptophan",
    "tyrosine",
    "valine",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Nature-style train/validation distribution plots for peptide length "
            "and per-sequence non-natural residue fraction."
        )
    )
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS), choices=DEFAULT_FORMATS)
    parser.add_argument(
        "--max-rows-per-group",
        type=int,
        default=None,
        help="Optional deterministic subsample per Dataset/Topology group for smoke tests.",
    )
    parser.add_argument(
        "--include-linear-validation",
        action="store_true",
        help="Include validation linear peptides as a validation-only group.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=None,
        help="Optional minimum peptide_length to include in plots, e.g. 6 filters out lengths < 6.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def read_train_tables(train_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for topology in TOPOLOGY_LABELS:
        path = train_root / topology / "parsed_peptides.csv"
        require_file(path, f"Training parsed CSV for {topology}")
        frame = pd.read_csv(path)
        frame["Dataset"] = "Train"
        frame["Cyclization"] = topology
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def read_validation_table(validation_csv: Path, include_linear: bool) -> pd.DataFrame:
    require_file(validation_csv, "Validation parsed CSV")
    frame = pd.read_csv(validation_csv)
    if "Cyclization" not in frame.columns:
        raise KeyError(f"Validation CSV must include Cyclization column: {validation_csv}")
    keep = set(TOPOLOGY_LABELS)
    if include_linear:
        keep.add("linear")
    frame = frame[frame["Cyclization"].isin(keep)].copy()
    frame["Dataset"] = "Validation"
    frame["source_file"] = str(validation_csv)
    return frame


def normalize_status(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().lower()


def split_residue_sequence(sequence: object) -> list[str]:
    if sequence is None or (isinstance(sequence, float) and math.isnan(sequence)):
        return []
    return [part.strip() for part in str(sequence).split("|") if part.strip()]


def is_natural_residue(residue: str) -> bool:
    token = residue.strip()
    lower = token.lower()
    if not token:
        return False
    if token.startswith("#") or "[nmet]" in lower or "n-methyl" in lower or "n-me" in lower:
        return False
    if "[" in token or "]" in token:
        return False
    if token.startswith("L-") or token.startswith("D-"):
        token = token[2:]
    token = " ".join(token.strip().lower().split())
    return token in NATURAL_RESIDUE_NAMES


def compute_residue_stats(sequence: object, peptide_length: object) -> dict[str, object]:
    residues = split_residue_sequence(sequence)
    length = pd.to_numeric(pd.Series([peptide_length]), errors="coerce").iloc[0]
    if pd.isna(length) or length <= 0:
        return {
            "sequence_residue_count": len(residues),
            "natural_count": pd.NA,
            "nonnatural_count": pd.NA,
            "nonnatural_fraction": pd.NA,
            "length_sequence_count_mismatch": bool(residues),
        }
    length_int = int(length)
    nonnatural_count = sum(not is_natural_residue(residue) for residue in residues)
    natural_count = len(residues) - nonnatural_count
    return {
        "sequence_residue_count": len(residues),
        "natural_count": natural_count,
        "nonnatural_count": nonnatural_count,
        "nonnatural_fraction": nonnatural_count / length_int,
        "length_sequence_count_mismatch": len(residues) != length_int,
    }


def topology_label(topology: str) -> str:
    if topology == "linear":
        return LINEAR_LABEL
    return TOPOLOGY_LABELS[topology]


def build_plotting_table(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.concat(
        [
            read_train_tables(args.train_root),
            read_validation_table(args.validation_csv, args.include_linear_validation),
        ],
        ignore_index=True,
    )
    required = {"parse_status", "peptide_length", "peptide_sequence", "Cyclization"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    raw["parse_status_norm"] = raw["parse_status"].map(normalize_status)
    raw["peptide_length_numeric"] = pd.to_numeric(raw["peptide_length"], errors="coerce")
    raw["is_plottable"] = raw["parse_status_norm"].eq("success") & raw["peptide_length_numeric"].notna()

    stats = pd.DataFrame(
        [
            compute_residue_stats(sequence, length)
            for sequence, length in zip(raw["peptide_sequence"], raw["peptide_length_numeric"], strict=False)
        ]
    )
    table = pd.concat([raw.reset_index(drop=True), stats], axis=1)
    table["Topology"] = table["Cyclization"].map(topology_label)

    plotting = table[table["is_plottable"]].copy()
    plotting = plotting[pd.to_numeric(plotting["nonnatural_fraction"], errors="coerce").notna()].copy()
    plotting["peptide_length"] = plotting["peptide_length_numeric"].astype(int)
    plotting["nonnatural_fraction"] = pd.to_numeric(plotting["nonnatural_fraction"], errors="coerce")

    if args.max_rows_per_group is not None:
        if args.max_rows_per_group <= 0:
            raise ValueError("--max-rows-per-group must be positive when provided")
    if args.min_length is not None:
        if args.min_length <= 0:
            raise ValueError("--min-length must be a positive integer when provided")
        plotting = plotting[plotting["peptide_length"] >= args.min_length].copy()

    if args.max_rows_per_group is not None:
        sampled_indices: list[int] = []
        grouped = plotting.groupby(["Dataset", "Cyclization"], sort=False)
        for _, group_indices in grouped.indices.items():
            group_index = pd.Index(group_indices)
            n_sample = min(len(group_index), args.max_rows_per_group)
            sampled = group_index.to_series().sample(n_sample, random_state=args.seed)
            sampled_indices.extend(sampled.tolist())
        plotting = plotting.iloc[sampled_indices].reset_index(drop=True)

    keep_columns = [
        "Dataset",
        "Cyclization",
        "Topology",
        "source_index",
        "mol_id",
        "input_smiles",
        "parse_status",
        "peptide_sequence",
        "monomer_id_sequence",
        "peptide_length",
        "sequence_residue_count",
        "length_sequence_count_mismatch",
        "natural_count",
        "nonnatural_count",
        "nonnatural_fraction",
        "source_file",
    ]
    keep_columns = [column for column in keep_columns if column in plotting.columns]
    return plotting[keep_columns].copy(), table


def summarize(table: pd.DataFrame, plotting: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = table.groupby(["Dataset", "Cyclization"], sort=False)
    for (dataset, cyclization), source_group in groups:
        plot_group = plotting[(plotting["Dataset"] == dataset) & (plotting["Cyclization"] == cyclization)]
        rows.append(
            {
                "Dataset": dataset,
                "Cyclization": cyclization,
                "Topology": topology_label(cyclization),
                "rows_total": int(len(source_group)),
                "rows_success": int(source_group["parse_status_norm"].eq("success").sum()),
                "rows_failed": int(source_group["parse_status_norm"].eq("failed").sum()),
                "rows_plotted": int(len(plot_group)),
                "length_median": safe_stat(plot_group["peptide_length"], np.median),
                "length_q25": safe_stat(plot_group["peptide_length"], lambda x: np.quantile(x, 0.25)),
                "length_q75": safe_stat(plot_group["peptide_length"], lambda x: np.quantile(x, 0.75)),
                "nonnatural_fraction_median": safe_stat(plot_group["nonnatural_fraction"], np.median),
                "nonnatural_fraction_q25": safe_stat(
                    plot_group["nonnatural_fraction"], lambda x: np.quantile(x, 0.25)
                ),
                "nonnatural_fraction_q75": safe_stat(
                    plot_group["nonnatural_fraction"], lambda x: np.quantile(x, 0.75)
                ),
                "overall_nonnatural_fraction": overall_nonnatural_fraction(plot_group),
                "sequence_length_mismatch_count": int(source_group["length_sequence_count_mismatch"].fillna(False).sum()),
            }
        )
    return pd.DataFrame(rows)


def safe_stat(values: pd.Series, func) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return None
    return float(func(clean))


def overall_nonnatural_fraction(frame: pd.DataFrame) -> float | None:
    nonnatural = pd.to_numeric(frame["nonnatural_count"], errors="coerce").fillna(0).sum()
    natural = pd.to_numeric(frame["natural_count"], errors="coerce").fillna(0).sum()
    total = natural + nonnatural
    if total == 0:
        return None
    return float(nonnatural / total)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ordered_topologies(plotting: pd.DataFrame) -> list[str]:
    order = list(TOPOLOGY_LABELS)
    if "linear" in set(plotting["Cyclization"]):
        order.append("linear")
    return [topology for topology in order if topology in set(plotting["Cyclization"])]


def save_figure(fig: mpl.figure.Figure, output_prefix: Path, formats: Iterable[str]) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = output_prefix.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "tiff"}:
            save_kwargs["dpi"] = 600
        fig.savefig(path, **save_kwargs)
        written.append(path)
    plt.close(fig)
    return written


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Write a small Markdown table without requiring pandas' optional tabulate dependency."""
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(format_markdown_cell)
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines)


def format_markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def make_violin_plot(
    plotting: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    output_prefix: Path,
    formats: Iterable[str],
    ylim: tuple[float, float] | None = None,
    integer_y: bool = False,
) -> list[Path]:
    topology_order = ordered_topologies(plotting)
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    x_base = np.arange(len(DATASET_ORDER), dtype=float)
    offsets = build_topology_offsets(topology_order, spacing=0.23)
    width = 0.20

    for i, dataset in enumerate(DATASET_ORDER):
        for topology in topology_order:
            subset = plotting[(plotting["Cyclization"] == topology) & (plotting["Dataset"] == dataset)]
            values = pd.to_numeric(subset[value_column], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            position = x_base[i] + offsets[topology]
            if len(values) > 1:
                violin = ax.violinplot(
                    values,
                    positions=[position],
                    widths=width,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(TOPOLOGY_COLORS[topology])
                    body.set_edgecolor("black")
                    body.set_linewidth(0.45)
                    body.set_alpha(0.72)
            else:
                ax.scatter(
                    [position],
                    values,
                    s=8,
                    color=TOPOLOGY_COLORS[topology],
                    edgecolor="black",
                    linewidth=0.3,
                    zorder=3,
                )
            ax.boxplot(
                values,
                positions=[position],
                widths=0.085,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 0.75},
                boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.55},
                whiskerprops={"color": "black", "linewidth": 0.5},
                capprops={"color": "black", "linewidth": 0.5},
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_xlim(x_base[0] - 0.48, x_base[-1] + 0.48)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if integer_y:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="both", direction="out")
    ax.grid(False)
    handles = [
        Patch(
            facecolor=TOPOLOGY_COLORS[topology],
            edgecolor="black",
            linewidth=0.45,
            label=topology_label(topology),
            alpha=0.72,
        )
        for topology in topology_order
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
        handlelength=1.0,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.45)
    return save_figure(fig, output_prefix, formats)


def build_topology_offsets(topology_order: list[str], spacing: float) -> dict[str, float]:
    center = (len(topology_order) - 1) / 2
    return {topology: (idx - center) * spacing for idx, topology in enumerate(topology_order)}


def make_fraction_scatter_plot(
    plotting: pd.DataFrame,
    *,
    output_prefix: Path,
    formats: Iterable[str],
    seed: int,
) -> list[Path]:
    topology_order = ordered_topologies(plotting)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    x_base = np.arange(len(DATASET_ORDER), dtype=float)
    dataset_positions = dict(zip(DATASET_ORDER, x_base, strict=False))
    offsets = build_topology_offsets(topology_order, spacing=0.23)
    rng = np.random.default_rng(seed)

    for dataset in DATASET_ORDER:
        for topology in topology_order:
            subset = plotting[(plotting["Cyclization"] == topology) & (plotting["Dataset"] == dataset)]
            if subset.empty:
                continue
            y = pd.to_numeric(subset["nonnatural_fraction"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y)
            y = y[mask]
            x = np.full(len(y), dataset_positions[dataset] + offsets[topology], dtype=float)
            jitter = rng.normal(0.0, 0.025, size=len(y))
            ax.scatter(
                x + jitter,
                y,
                s=2.1,
                alpha=0.055,
                color=TOPOLOGY_COLORS[topology],
                edgecolors="none",
                rasterized=True,
                label=topology_label(topology),
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Per-sequence non-natural fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="both", direction="out")
    ax.grid(False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TOPOLOGY_COLORS[topology],
               markeredgecolor="none", markersize=4, label=topology_label(topology))
        for topology in topology_order
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", handlelength=1.1)
    fig.tight_layout(pad=0.5)
    return save_figure(fig, output_prefix, formats)


def write_run_record(
    output_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    lines = [
        "# Train/Validation Sequence Statistics Plot Record",
        "",
        "## Runtime",
        f"- Timestamp: {now}",
        f"- Host: {socket.gethostname()}",
        f"- Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', sys.prefix)}",
        f"- Python executable: {sys.executable}",
        "",
        "## Inputs",
        f"- Training root: {args.train_root}",
        f"- Validation CSV: {args.validation_csv}",
        f"- Include validation linear: {args.include_linear_validation}",
        f"- Minimum plotted peptide length: {args.min_length if args.min_length is not None else 'none'}",
        f"- Max rows per group: {args.max_rows_per_group if args.max_rows_per_group else 'full'}",
        "",
        "## Non-natural residue definition",
        "- D/L stereochemistry is not treated as non-natural.",
        "- N-methyl residues, unknown #hash residues, and non-standard residue names are treated as non-natural.",
        "- Fractions are computed as non-natural residue count divided by peptide_length.",
        "",
        "## Outputs",
    ]
    lines.extend([f"- {path}" for path in figure_paths])
    lines.extend(
        [
            f"- {output_dir / 'tables' / 'plotting_table.csv'}",
            f"- {output_dir / 'tables' / 'summary_by_dataset_topology.csv'}",
            "",
            "## Summary",
            dataframe_to_markdown(summary),
            "",
            "## Command",
            "```bash",
            command,
            "```",
        ]
    )
    (output_dir / "run_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    figure_dir = args.output_dir / "figures"
    table_dir = args.output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    plotting, full_table = build_plotting_table(args)
    summary = summarize(full_table, plotting)
    plotting.to_csv(table_dir / "plotting_table.csv", index=False)
    summary.to_csv(table_dir / "summary_by_dataset_topology.csv", index=False)

    figure_paths: list[Path] = []
    figure_paths.extend(
        make_violin_plot(
            plotting,
            value_column="peptide_length",
            ylabel="Peptide length (aa)",
            output_prefix=figure_dir / "sequence_length_distribution",
            formats=args.formats,
            integer_y=True,
        )
    )
    figure_paths.extend(
        make_violin_plot(
            plotting,
            value_column="nonnatural_fraction",
            ylabel="Per-sequence non-natural fraction",
            output_prefix=figure_dir / "nonnatural_fraction_distribution",
            formats=args.formats,
            ylim=(-0.02, 1.02),
        )
    )
    figure_paths.extend(
        make_fraction_scatter_plot(
            plotting,
            output_prefix=figure_dir / "nonnatural_fraction_scatter",
            formats=args.formats,
            seed=args.seed,
        )
    )
    write_run_record(args.output_dir, args, summary, figure_paths)

    print(f"Wrote plotting table: {table_dir / 'plotting_table.csv'}")
    print(f"Wrote summary table: {table_dir / 'summary_by_dataset_topology.csv'}")
    print(f"Wrote run record: {args.output_dir / 'run_record.md'}")
    for path in figure_paths:
        print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
