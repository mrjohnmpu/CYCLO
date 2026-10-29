#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MM_PER_INCH = 25.4
DOUBLE_COLUMN_WIDTH = 180.0 / MM_PER_INCH
FONT_PATH = Path("/usr/share/fonts/arial.ttf")
TOPOLOGY_LABELS = {
    "head_to_tail": "Head-to-tail",
    "sc_to_tail": "Side-chain-to-tail",
    "disulfide_bridge": "Disulfide-bridged",
}
COLORS = {
    "AAE": "#4C78A8",
    "VAE": "#72B7B2",
    "CharRNN": "#59A14F",
    "PepINVENT-30": "#E3BF3B",
    "PepINVENT-50": "#F28E2B",
    "PepINVENT-70": "#B86B32",
    "TopoCycGPT": "#6F4BB2",
}
LINE_STYLES = {
    "AAE": "-",
    "VAE": "-",
    "CharRNN": "-",
    "PepINVENT-30": (0, (1.5, 1.2)),
    "PepINVENT-50": (0, (3.0, 1.4)),
    "PepINVENT-70": (0, (5.0, 1.5)),
    "TopoCycGPT": "-",
}
RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "axes.grid": False,
    "savefig.dpi": 600,
}


def apply_style() -> None:
    if FONT_PATH.is_file():
        font_manager.fontManager.addfont(str(FONT_PATH))
        name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
        RCPARAMS["font.sans-serif"] = [
            name,
            "Helvetica",
            "Nimbus Sans",
            "Liberation Sans",
            "DejaVu Sans",
        ]
    else:
        warnings.warn("Arial font file was not found; using a sans-serif fallback.", RuntimeWarning)
    plt.rcParams.update(RCPARAMS)


def save_figure(figure: plt.Figure, stem: Path, formats: list[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    if "svg" in formats:
        figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    if "pdf" in formats:
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    if "png" in formats:
        figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    if "tiff" in formats:
        figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def shared_legend(figure: plt.Figure, model_order: list[str]) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model],
            linestyle=LINE_STYLES[model],
            linewidth=1.4,
            label=model,
        )
        for model in model_order
    ]
    figure.legend(
        handles=handles,
        labels=model_order,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(model_order),
        handlelength=2.0,
        columnspacing=1.1,
    )


def plot_ecdf(
    ecdf: pd.DataFrame,
    output_root: Path,
    model_order: list[str],
    topology_order: list[str],
    formats: list[str],
) -> None:
    figure, axes = plt.subplots(
        1,
        len(topology_order),
        figsize=(DOUBLE_COLUMN_WIDTH, 54.0 / MM_PER_INCH),
        sharex=True,
        sharey=True,
    )
    for axis, topology in zip(np.atleast_1d(axes), topology_order):
        subset = ecdf.loc[ecdf["topology"].eq(topology)]
        for model in model_order:
            values = subset.loc[subset["model"].eq(model)]
            axis.plot(
                values["nearest_train_similarity"],
                values["ecdf"],
                color=COLORS[model],
                linestyle=LINE_STYLES[model],
                linewidth=1.1 if model != "TopoCycGPT" else 1.5,
                alpha=0.95,
            )
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xlabel("Nearest-training-neighbor Tanimoto similarity")
        axis.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("ECDF")
    shared_legend(figure, model_order)
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.80, wspace=0.28)
    save_figure(
        figure,
        output_root / "figures" / "nearest_training_similarity_ecdf",
        formats,
    )


def plot_coverage(
    coverage: pd.DataFrame,
    output_root: Path,
    model_order: list[str],
    topology_order: list[str],
    formats: list[str],
) -> None:
    figure, axes = plt.subplots(
        1,
        len(topology_order),
        figsize=(DOUBLE_COLUMN_WIDTH, 54.0 / MM_PER_INCH),
        sharex=True,
        sharey=True,
    )
    for axis, topology in zip(np.atleast_1d(axes), topology_order):
        subset = coverage.loc[coverage["topology"].eq(topology)]
        for model in model_order:
            values = subset.loc[subset["model"].eq(model)]
            axis.plot(
                values["threshold"],
                values["coverage"],
                color=COLORS[model],
                linestyle=LINE_STYLES[model],
                linewidth=1.1 if model != "TopoCycGPT" else 1.5,
                alpha=0.95,
            )
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xlabel("Tanimoto threshold")
        axis.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("Reference coverage")
    shared_legend(figure, model_order)
    figure.text(
        0.5,
        0.075,
        "Direction: topology-matched reference to generated set",
        ha="center",
        va="center",
        fontsize=6,
    )
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.25, top=0.80, wspace=0.28)
    save_figure(
        figure,
        output_root / "figures" / "reference_coverage_curve",
        formats,
    )


def plot_scaffold_novelty(
    summary: pd.DataFrame,
    output_root: Path,
    model_order: list[str],
    topology_order: list[str],
    formats: list[str],
) -> None:
    figure, axes = plt.subplots(
        1,
        len(topology_order),
        figsize=(DOUBLE_COLUMN_WIDTH, 62.0 / MM_PER_INCH),
        sharey=True,
    )
    x = np.arange(len(model_order), dtype=float)
    width = 0.34
    for axis, topology in zip(np.atleast_1d(axes), topology_order):
        subset = summary.loc[summary["topology"].eq(topology)].set_index("model")
        subset = subset.reindex(model_order)
        murcko = subset["Murcko_novelty"].to_numpy(dtype=float)
        generic = subset["generic_scaffold_novelty"].to_numpy(dtype=float)
        axis.bar(
            x - width / 2,
            murcko,
            width,
            color=[COLORS[model] for model in model_order],
            edgecolor="black",
            linewidth=0.35,
            label="Bemis-Murcko",
        )
        axis.bar(
            x + width / 2,
            generic,
            width,
            color=[COLORS[model] for model in model_order],
            edgecolor="black",
            linewidth=0.45,
            hatch="///",
            alpha=0.62,
            label="Generic",
        )
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.set_xticks(x)
        axis.set_xticklabels(model_order, rotation=52, ha="right", rotation_mode="anchor")
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("Scaffold novelty")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#8F8F8F", edgecolor="black", linewidth=0.4),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#BDBDBD",
            edgecolor="black",
            linewidth=0.4,
            hatch="///",
        ),
    ]
    figure.legend(
        handles,
        ["Bemis-Murcko", "Generic scaffold"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.34, top=0.82, wspace=0.25)
    save_figure(
        figure,
        output_root / "figures" / "scaffold_novelty",
        formats,
    )


def generate_all_figures(
    *,
    output_root: Path,
    model_order: list[str],
    topology_order: list[str],
    formats: list[str],
) -> None:
    apply_style()
    summary_path = output_root / "tables" / "chemical_space_coverage_summary.csv"
    ecdf_path = output_root / "curves" / "nearest_training_similarity_ecdf.csv"
    coverage_path = output_root / "curves" / "reference_coverage_curve_grid.csv"
    for path in (summary_path, ecdf_path, coverage_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = pd.read_csv(summary_path)
    ecdf = pd.read_csv(ecdf_path)
    coverage = pd.read_csv(coverage_path)
    plot_ecdf(ecdf, output_root, model_order, topology_order, formats)
    plot_coverage(coverage, output_root, model_order, topology_order, formats)
    plot_scaffold_novelty(summary, output_root, model_order, topology_order, formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot memorization and chemical-space coverage results.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-order", nargs="+", required=True)
    parser.add_argument("--topology-order", nargs="+", required=True)
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png", "tiff"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_all_figures(
        output_root=args.output_root,
        model_order=args.model_order,
        topology_order=args.topology_order,
        formats=args.formats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
