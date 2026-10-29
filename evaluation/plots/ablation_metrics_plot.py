"""Descriptive comparison of retained ablations from validated metric summaries."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=["Validity", "Novelty", "IntDiv"])
    parser.add_argument("--formats", nargs="+", choices=["pdf", "svg", "png", "tiff"], default=["pdf", "svg", "png"])
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.csv], ignore_index=True)
    required = ["Method", "Cyclization", *args.metrics]
    if any(column not in frame for column in required):
        raise ValueError(f"Required columns: {required}")
    if frame.duplicated(["Method", "Cyclization"]).any():
        raise ValueError("Supply one summary per comparison, without duplicate method/topology rows")
    topologies = [value for value in ["head_to_tail", "sc_to_tail", "disulfide_bridge"] if value in set(frame.Cyclization)]
    methods = frame.Method.drop_duplicates().tolist()
    if not topologies or not methods:
        raise ValueError("No recognized cyclic topology rows")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = ["#B5BDC4", "#8EB8C7", "#E7BD89", "#A8BE9E", "#D99B95"]
    width_mm = 180.0
    fig, axes = plt.subplots(1, len(args.metrics), figsize=(width_mm / 25.4, 62 / 25.4), squeeze=False)
    x = np.arange(len(topologies))
    width = 0.8 / len(methods)
    for panel, metric in enumerate(args.metrics):
        ax = axes[0, panel]
        for index, method in enumerate(methods):
            rows = frame[frame.Method == method].set_index("Cyclization")
            if any(topology not in rows.index for topology in topologies):
                raise ValueError(f"Incomplete topology coverage for {method}")
            values = pd.to_numeric(rows.loc[topologies, metric], errors="raise").to_numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {metric} values for {method}")
            ax.bar(x - 0.4 + width / 2 + index * width, values, width, color=colors[index % len(colors)], label=method)
        ax.set_xticks(x, [value.replace("_", "\n") for value in topologies])
        ax.set_ylabel(metric)
        ax.text(-0.12, 1.05, chr(97 + panel), transform=ax.transAxes, fontweight="bold", fontsize=7)
        ax.grid(False)
        if metric in {"Validity", "Novelty", "IntDiv", "IntDiv2", "SNN", "Uniq@10k", "Uniq@20k", "Uniq@30k"}:
            ax.set_ylim(0, 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(methods)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exports = {"pdf": "ablation_metrics.pdf", "svg": "ablation_metrics.svg", "png": "ablation_metrics.png", "tiff": "ablation_metrics.tiff"}
    for suffix in args.formats:
        fig.savefig(args.out_dir / exports[suffix], dpi=600)
    frame.to_csv(args.out_dir / "source_data.csv", index=False)
    (args.out_dir / "README.txt").write_text("Descriptive metrics from supplied summaries; no replicate variance or inferential statistics are implied.\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
