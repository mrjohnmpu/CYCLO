#!/usr/bin/env python3
"""
Reference-style PPO training curves without translucent marker halos.

Required CSV columns:
step, mean_reward_total, mean_perm_raw, mean_hemo_prob, mean_kras_kd_pred

The shaded band is the within-bin ±1 SD across steps, not uncertainty across
independent random-seed training runs.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

INPUT_CSV = Path('runs/ppo_head_to_tail_perm_hemo_kras_v2_lam1_b128_full_finetune_s500_log1_20260709/reward_trace.csv')
OUTPUT_DIR = Path('results/kras_design')

N_BINS = 10
MAX_STEP = 300
FIG_WIDTH_MM = 45
FIG_HEIGHT_MM = 34.518
OUTPUT_BASENAME = "kras_ppo_trace_circle"
FONT_PATH = Path("/usr/share/fonts/arial.ttf")


def read_column(rows, name):
    try:
        return np.asarray([float(row[name]) for row in rows], dtype=float)
    except KeyError as exc:
        raise KeyError(f"Missing required column: {name}") from exc


def has_required_values(row):
    required = (
        "step",
        "mean_reward_total",
        "mean_perm_raw",
        "mean_hemo_prob",
        "mean_kras_kd_pred",
    )
    try:
        return all(np.isfinite(float(row[name])) for name in required)
    except (KeyError, TypeError, ValueError):
        return False


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if has_required_values(row)
        ]

    if not rows:
        raise ValueError("No valid rows were found.")

    step = read_column(rows, "step")
    series = {
        "reward": read_column(rows, "mean_reward_total"),
        "permeability": read_column(rows, "mean_perm_raw"),
        "hemolysis": read_column(rows, "mean_hemo_prob"),
        "kras_kd": read_column(rows, "mean_kras_kd_pred"),
    }

    order = np.argsort(step)
    step = step[order]
    series = {key: values[order] for key, values in series.items()}

    if MAX_STEP is not None:
        mask = step <= MAX_STEP
        step = step[mask]
        series = {key: values[mask] for key, values in series.items()}

    if len(step) == 0:
        raise ValueError("No rows remain after applying MAX_STEP.")

    return step, series


def binned_summary(x, y, edges):
    x_mean, y_mean, y_sd = [], [], []

    for index in range(len(edges) - 1):
        lower, upper = edges[index], edges[index + 1]

        if index < len(edges) - 2:
            mask = (x >= lower) & (x < upper)
        else:
            mask = (x >= lower) & (x <= upper)

        x_bin = x[mask]
        y_bin = y[mask]

        if len(y_bin) == 0:
            continue

        x_mean.append(np.mean(x_bin))
        y_mean.append(np.mean(y_bin))
        y_sd.append(np.std(y_bin, ddof=1) if len(y_bin) > 1 else 0.0)

    return np.asarray(x_mean), np.asarray(y_mean), np.asarray(y_sd)


def configure_style():
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Arial font file not found: {FONT_PATH}")
    font_manager.fontManager.addfont(str(FONT_PATH))
    arial_name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [arial_name],
        "font.size": 6.0,
        "axes.labelsize": 6.0,
        "axes.titlesize": 6.0,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "axes.linewidth": 0.45,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 1.8,
        "ytick.major.size": 1.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def make_figure(step, series):
    configure_style()

    specs = [
        {
            "key": "reward",
            "panel": "a",
            "title": "Total reward",
            "ylabel": "Total reward",
            "ylim": (0.0, 0.80),
            "threshold": None,
            "threshold_label": None,
            "color": "#d62728",
        },
        {
            "key": "permeability",
            "panel": "b",
            "title": "Permeability",
            "ylabel": "Permeability",
            "ylim": (-9.5, -4.5),
            "threshold": -6.0,
            "threshold_label": "p = -6",
            "color": "#2ca02c",
        },
        {
            "key": "hemolysis",
            "panel": "c",
            "title": "Hemolysis",
            "ylabel": "Hemolysis",
            "ylim": (0.0, 0.55),
            "threshold": 0.15,
            "threshold_label": "h = 0.15",
            "color": "#1f77b4",
        },
        {
            "key": "kras_kd",
            "panel": "d",
            "title": "KRAS KD",
            "ylabel": "KRAS KD",
            "ylim": (0.70, 1.40),
            "threshold": None,
            "threshold_label": None,
            "color": "#9467bd",
        },
    ]

    edges = np.linspace(step.min(), step.max(), N_BINS + 1)

    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(FIG_WIDTH_MM / 25.4, FIG_HEIGHT_MM / 25.4),
        gridspec_kw={"hspace": 0.20},
    )

    for axis, spec in zip(axes, specs):
        x_bin, y_mean, y_sd = binned_summary(
            step,
            series[spec["key"]],
            edges,
        )

        color = spec["color"]

        axis.fill_between(
            x_bin,
            y_mean - y_sd,
            y_mean + y_sd,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

        axis.plot(
            x_bin,
            y_mean,
            color=color,
            linewidth=1.0,
            zorder=3,
        )

        # Single solid marker only; no outer translucent halo.
        axis.scatter(
            x_bin,
            y_mean,
            s=12,
            color=color,
            alpha=0.90,
            linewidths=0,
            zorder=4,
        )

        if spec["threshold"] is not None:
            axis.axhline(
                spec["threshold"],
                color="0.55",
                linestyle=(0, (3, 2)),
                linewidth=0.55,
                zorder=2,
            )
            axis.text(
                0.985,
                spec["threshold"],
                spec["threshold_label"],
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=5.0,
                color="0.35",
            )

        axis.set_ylim(*spec["ylim"])
        axis.set_ylabel(spec["ylabel"], labelpad=1.2)

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", pad=1.0)
        axis.margins(x=0.02)

        axis.text(
            -0.18,
            1.02,
            spec["panel"],
            transform=axis.transAxes,
            fontsize=6.0,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    x_upper = int(np.ceil(step.max() / 100.0) * 100)
    axes[-1].set_xlim(0, x_upper)
    axes[-1].set_xticks(np.arange(0, x_upper + 1, 100))
    axes[-1].set_xlabel("Step", labelpad=1.2)

    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)

    fig.subplots_adjust(left=0.30, right=0.91, top=0.98, bottom=0.16)
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    step, series = load_data(INPUT_CSV)
    fig = make_figure(step, series)

    output_base = OUTPUT_DIR / OUTPUT_BASENAME
    fig.savefig(
        output_base.with_suffix(".pdf"),
        facecolor="white",
    )

    plt.close(fig)
    print(f"Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()