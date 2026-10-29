#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


DEFAULT_FIGURE_ROOT = Path('results/head_to_tail_scaffold_statistics_20260412/figures')
DEFAULT_OUTPUT_STEM = "head_to_tail_scaffold_summary_2x2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine scaffold summary PNG panels into one clean 2x2 canvas.")
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--output-stem", type=str, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--width-mm", type=float, default=180.0)
    parser.add_argument("--height-mm", type=float, default=160.0)
    return parser.parse_args()


def mm_to_inch(mm: float) -> float:
    return mm / 25.4


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def main() -> None:
    args = parse_args()
    figure_root = args.figure_root
    output_svg = figure_root / f"{args.output_stem}.svg"
    output_pdf = figure_root / f"{args.output_stem}.pdf"
    output_png = figure_root / f"{args.output_stem}.png"

    panels = [
        (figure_root / "generic_occurrence_histogram.png", "(A)"),
        (figure_root / "generic_rank_frequency.png", "(B)"),
        (figure_root / "top_generic_scaffolds_bar.png", "(C)"),
        (figure_root / "top_generic_scaffolds_grid.png", "(D)"),
    ]

    for path, _ in panels:
        if not path.exists():
            raise FileNotFoundError(f"Missing source figure: {path}")

    configure_matplotlib()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(mm_to_inch(args.width_mm), mm_to_inch(args.height_mm)),
        dpi=300,
        constrained_layout=False,
    )
    axes = axes.flatten()

    for ax, (path, label) in zip(axes, panels):
        image = mpimg.imread(path)
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            -0.04,
            1.04,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            family="sans-serif",
            clip_on=False,
        )

    plt.subplots_adjust(left=0.035, right=0.995, top=0.985, bottom=0.025, wspace=0.06, hspace=0.08)
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")
    fig.savefig(output_png, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)

    print(f"Wrote {output_svg}")
    print(f"Wrote {output_pdf}")
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
