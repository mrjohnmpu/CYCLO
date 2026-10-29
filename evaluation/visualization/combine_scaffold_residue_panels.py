#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


DEFAULT_SCAFFOLD_ROOT = Path('results/head_to_tail_scaffold_statistics_20260412')
DEFAULT_RESIDUE_ROOT = Path('results/head_to_tail_residue_composition_20260526')
DEFAULT_OUTPUT_ROOT = Path('results/head_to_tail_structure_diversity_20260526')
TEST2_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine scaffold diversity and residue composition panels into one manuscript figure."
    )
    parser.add_argument("--scaffold-root", type=Path, default=DEFAULT_SCAFFOLD_ROOT)
    parser.add_argument("--residue-root", type=Path, default=DEFAULT_RESIDUE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-stem", type=str, default="head_to_tail_structure_diversity")
    parser.add_argument("--width-mm", type=float, default=183.0)
    parser.add_argument("--height-mm", type=float, default=220.0)
    return parser.parse_args()


def mm_to_inch(value: float) -> float:
    return value / 25.4


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=2.5)


def load_residue_module():
    module_path = TEST2_ROOT / "evaluation" / "analysis" / "head_to_tail_residue_composition_analysis.py"
    spec = importlib.util.spec_from_file_location("head_to_tail_residue_composition_analysis", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import residue composition module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def plot_generic_occurrence(ax: plt.Axes, scaffold_root: Path) -> None:
    df = pd.read_csv(scaffold_root / "summary" / "generic_occurrence_histogram.csv")
    ax.bar(
        df["occurrence"].to_numpy(dtype=int),
        df["scaffold_count"].to_numpy(dtype=int),
        width=0.9,
        color="#0072B2",
        alpha=0.86,
        edgecolor="none",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Occurrence count")
    ax.set_ylabel("Number of generic scaffolds")
    ax.set_title("Generic scaffold occurrence")
    if len(df) > 12:
        x = df["occurrence"].to_numpy(dtype=int)
        ticks = np.unique(np.linspace(x.min(), x.max(), num=7, dtype=int))
        ax.set_xticks(ticks)
    style_axis(ax)


def plot_generic_rank_frequency(ax: plt.Axes, scaffold_root: Path) -> None:
    df = pd.read_csv(scaffold_root / "summary" / "generic_rank_frequency.csv")
    x = df["rank"].to_numpy(dtype=int)
    y = df["count"].to_numpy(dtype=int)
    ax.plot(x, y, color="#0072B2", linewidth=1.4)
    ax.set_xlabel("Generic scaffold rank")
    ax.set_ylabel("Frequency")
    ax.set_title("Rank-frequency distribution")
    top_n = min(100, len(df))
    if top_n >= 2:
        inset = inset_axes(ax, width="43%", height="50%", loc="center right", borderpad=0.9)
        inset.plot(x[:top_n], y[:top_n], color="#D55E00", linewidth=1.0)
        inset.set_xlim(1, int(x[top_n - 1]))
        inset.set_title("Top 100", fontsize=5.8)
        inset.tick_params(axis="both", labelsize=5.2, length=1.8, width=0.55)
        for spine in inset.spines.values():
            spine.set_linewidth(0.55)
    style_axis(ax)


def plot_top_generic_bar(ax: plt.Axes, scaffold_root: Path) -> None:
    df = pd.read_csv(scaffold_root / "summary" / "top_generic_scaffolds.csv").head(20)
    x = np.arange(len(df))
    ax.bar(x, df["count"].to_numpy(dtype=int), color="#0072B2", alpha=0.86, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{int(rank)}" for rank in df["rank"]], rotation=90)
    ax.set_xlabel("Generic scaffold rank")
    ax.set_ylabel("Count")
    ax.set_title("Top20 generic scaffolds")
    style_axis(ax)


def plot_generic_grid(ax: plt.Axes, scaffold_root: Path) -> None:
    image_path = scaffold_root / "figures" / "top_generic_scaffolds_grid.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing generic scaffold grid: {image_path}")
    image = mpimg.imread(image_path)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Top20 generic scaffold structures", pad=2)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    residue_module = load_residue_module()

    plotting_records = args.residue_root / "plotting_records.csv"
    if not plotting_records.exists():
        raise FileNotFoundError(f"Missing residue plotting table: {plotting_records}")
    residue_df = pd.read_csv(plotting_records)

    fig = plt.figure(
        figsize=(mm_to_inch(args.width_mm), mm_to_inch(args.height_mm)),
        constrained_layout=False,
    )
    outer = fig.add_gridspec(
        nrows=3,
        ncols=3,
        height_ratios=[1.0, 2.15, 1.1],
        hspace=0.34,
        wspace=0.34,
    )

    axes = [
        fig.add_subplot(outer[0, 0]),
        fig.add_subplot(outer[0, 1]),
        fig.add_subplot(outer[0, 2]),
        fig.add_subplot(outer[1, :]),
        fig.add_subplot(outer[2, 0]),
        fig.add_subplot(outer[2, 1]),
        fig.add_subplot(outer[2, 2]),
    ]

    plot_generic_occurrence(axes[0], args.scaffold_root)
    plot_generic_rank_frequency(axes[1], args.scaffold_root)
    plot_top_generic_bar(axes[2], args.scaffold_root)
    plot_generic_grid(axes[3], args.scaffold_root)
    residue_module.draw_panel_a(axes[4], residue_df)
    residue_module.draw_panel_b(axes[5], residue_df)
    residue_module.draw_panel_c(axes[6], residue_df)

    for ax, label in zip(axes, list("ABCDEFG")):
        add_panel_label(ax, label)

    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = args.output_root / args.output_stem
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600, facecolor="white")
    plt.close(fig)

    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".svg"))
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
