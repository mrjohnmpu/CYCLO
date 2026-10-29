import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from datetime import datetime


# =========================================================
# Part 1. Global plotting style (Nature-like)
# =========================================================
def set_nature_style():
    """
    Set a clean, publication-style matplotlib theme.
    """
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    })


# =========================================================
# Part 2. Utility functions
# =========================================================
def rolling_mean(series, window=5):
    """
    Centered rolling mean for smoothing.
    """
    return series.rolling(window=window, center=True, min_periods=1).mean()


def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


CORE_METRICS = {
    "mean_reward_total": {
        "label": "Reward",
        "color": "#2C7FB8",
        "ylabel": "Mean total reward",
        "raw": True,
        "smooth": True,
        "smooth_window": 10,
        "zero_line": True,
        "raw_alpha": 0.20,
        "raw_lw": 0.8,
        "smooth_lw": 2.0,
        "ylim": (0.0, 0.75),
    },
    "valid_rate": {
        "label": "Valid rate",
        "color": "#1B9E77",
        "ylabel": "Valid rate",
        "raw": True,
        "smooth": True,
        "smooth_window": 10,
        "ylim": (0.0, 1.0),
    },
    "mean_perm_raw": {
        "label": "Permeability",
        "color": "#D95F02",
        "ylabel": "Mean permeability",
        "raw": True,
        "smooth": True,
        "smooth_window": 10,
    },
    "mean_hemo_prob": {
        "label": "Hemo probability",
        "color": "#E66101",
        "ylabel": "Score",
        "raw": True,
        "smooth": True,
        "smooth_window": 5,
        "linestyle": "--",
        "raw_alpha": 0.20,
        "raw_lw": 0.8,
        "smooth_lw": 2.0,
        "ylim": (0.0, 1.0),
    },
    "mean_hemo_safe": {
        "label": "Hemo safe",
        "color": "#5E3C99",
        "ylabel": "Score",
        "raw": True,
        "smooth": True,
        "smooth_window": 5,
        "linestyle": "-",
        "raw_alpha": 0.20,
        "raw_lw": 0.8,
        "smooth_lw": 2.0,
        "ylim": (0.0, 1.0),
    },
}

DEFAULT_SINGLE_METRICS = [
    "mean_reward_total",
    "valid_rate",
    "mean_perm_raw",
    "mean_hemo_prob",
    "mean_hemo_safe",
]


def apply_axis_style(ax, xlabel=None, ylabel=None, title=None,
                     xlim=None, ylim=None, xticks=None, yticks=None):
    """
    Apply clean axis styling.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # No heavy grid; optionally use a very light horizontal guide
    ax.grid(False)


def plot_metric_on_axis(
    ax,
    df,
    x_col,
    y_col,
    label=None,
    color=None,
    smooth_window=5,
    raw=True,
    smooth=True,
    raw_alpha=0.25,
    raw_lw=0.8,
    smooth_lw=1.8,
    std_col=None,
    band_alpha=0.18,
    linestyle="-",
    zero_line=False,
):
    """
    Plot one metric on a given axis:
      - raw line
      - smoothed line
      - optional std band around raw mean
    """
    if y_col not in df.columns:
        print(f"[Warning] Column '{y_col}' not found. Skip.")
        return

    x = df[x_col]
    y = df[y_col]

    # Standard deviation band
    if std_col is not None and std_col in df.columns:
        std = df[std_col]
        ax.fill_between(
            x,
            y - std,
            y + std,
            alpha=band_alpha,
            color=color,
            linewidth=0
        )

    # Raw line
    if raw:
        ax.plot(
            x,
            y,
            color=color,
            alpha=raw_alpha,
            linewidth=raw_lw,
            linestyle=linestyle,
            label=f"{label} (raw)" if smooth else label
        )

    # Smoothed line
    if smooth:
        y_smooth = rolling_mean(y, window=smooth_window)
        ax.plot(
            x,
            y_smooth,
            color=color,
            linewidth=smooth_lw,
            linestyle=linestyle,
            label=label if label is not None else y_col
        )

    if zero_line:
        ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)


def plot_panels(
    df,
    panel_specs,
    x_col="step",
    fig_width_mm=180,
    fig_height_mm=80,
    sharex=False,
    out_path=None,
    dpi=600,
):
    """
    Generic multi-panel plotting function.

    Parameters
    ----------
    df : DataFrame
    panel_specs : list of dict
        Each dict defines one panel.
        Expected keys:
          - title
          - ylabel
          - xlabel
          - metrics: list of metric dicts
          - ylim / xlim / xticks / yticks (optional)
    x_col : str
        Column used as x-axis.
    fig_width_mm : float
    fig_height_mm : float
    out_path : str or None
        Save figure if provided.
    """
    set_nature_style()

    n_panels = len(panel_specs)
    fig_w = fig_width_mm / 25.4
    fig_h = fig_height_mm / 25.4

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(fig_w, fig_h),
        sharex=sharex
    )

    if n_panels == 1:
        axes = [axes]

    for i, (ax, spec) in enumerate(zip(axes, panel_specs)):
        metrics = spec.get("metrics", [])

        for m in metrics:
            plot_metric_on_axis(
                ax=ax,
                df=df,
                x_col=x_col,
                y_col=m["y_col"],
                label=m.get("label", m["y_col"]),
                color=m.get("color", None),
                smooth_window=m.get("smooth_window", 5),
                raw=m.get("raw", True),
                smooth=m.get("smooth", True),
                raw_alpha=m.get("raw_alpha", 0.25),
                raw_lw=m.get("raw_lw", 0.8),
                smooth_lw=m.get("smooth_lw", 1.8),
                std_col=m.get("std_col", None),
                band_alpha=m.get("band_alpha", 0.18),
                linestyle=m.get("linestyle", "-"),
                zero_line=m.get("zero_line", False),
            )

        apply_axis_style(
            ax=ax,
            xlabel=spec.get("xlabel", "Step"),
            ylabel=spec.get("ylabel", None),
            title=spec.get("title", None),
            xlim=spec.get("xlim", None),
            ylim=spec.get("ylim", None),
            xticks=spec.get("xticks", None),
            yticks=spec.get("yticks", None),
        )

        if spec.get("legend", True):
            ax.legend(
                frameon=False,
                loc=spec.get("legend_loc", "best"),
                handlelength=2.0
            )

        # Optional panel letters
        panel_letter = spec.get("panel_letter", None)
        if panel_letter is not None:
            ax.text(
                -0.14, 1.04, panel_letter,
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                va="bottom",
                ha="left"
            )

    plt.tight_layout()

    if out_path is not None:
        ensure_dir(out_path)
        plt.savefig(out_path, dpi=dpi)
        print(f"Figure saved to: {out_path}")

    return fig, axes


def infer_xticks(df, step_col="step", max_ticks=7):
    step_max = int(df[step_col].max())
    if step_max <= 0:
        return np.array([0])

    raw_interval = step_max / max(1, max_ticks - 1)
    magnitude = 10 ** np.floor(np.log10(raw_interval))
    normalized = raw_interval / magnitude

    if normalized <= 1:
        nice_interval = magnitude
    elif normalized <= 2:
        nice_interval = 2 * magnitude
    elif normalized <= 5:
        nice_interval = 5 * magnitude
    else:
        nice_interval = 10 * magnitude

    interval = int(max(1, nice_interval))
    return np.arange(0, step_max + interval, interval)


def validate_metric_columns(df, metric_names):
    missing = [metric_name for metric_name in metric_names if metric_name not in df.columns]
    if missing:
        raise ValueError(f"Missing required metric columns in CSV: {missing}")


def filter_dataframe_by_max_step(df, max_step):
    if max_step is None:
        return df.sort_values("step").reset_index(drop=True).copy()
    filtered = df[df["step"] <= int(max_step)].copy()
    filtered = filtered.sort_values("step").reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"No rows remain after applying --max_step={max_step}")
    return filtered


def build_single_metric_panel_spec(metric_name, df, smooth_window=10):
    if metric_name not in CORE_METRICS:
        raise ValueError(f"Unsupported metric '{metric_name}'. Supported metrics: {sorted(CORE_METRICS)}")
    spec = CORE_METRICS[metric_name].copy()
    spec["smooth_window"] = smooth_window if spec.get("smooth", False) else spec.get("smooth_window", smooth_window)
    return {
        "panel_letter": "a",
        "title": spec["label"],
        "xlabel": "Step",
        "ylabel": spec["ylabel"],
        "xlim": (0, max(df["step"]) + 20),
        "ylim": spec.get("ylim", None),
        "xticks": infer_xticks(df),
        "metrics": [
            {
                "y_col": metric_name,
                "label": spec["label"],
                "color": spec["color"],
                "raw": spec.get("raw", True),
                "smooth": spec.get("smooth", True),
                "smooth_window": spec.get("smooth_window", smooth_window),
                "zero_line": spec.get("zero_line", False),
                "raw_alpha": spec.get("raw_alpha", 0.25),
                "raw_lw": spec.get("raw_lw", 0.8),
                "smooth_lw": spec.get("smooth_lw", 1.8),
                "std_col": spec.get("std_col", None),
                "band_alpha": spec.get("band_alpha", 0.18),
                "linestyle": "-",
            }
        ],
        "legend": True,
        "legend_loc": "best",
    }


def plot_single_metric(df, metric_name, out_path, smooth_window=10, dpi=600):
    panel_specs = [build_single_metric_panel_spec(metric_name, df, smooth_window=smooth_window)]
    return plot_panels(
        df=df,
        panel_specs=panel_specs,
        x_col="step",
        fig_width_mm=90,
        fig_height_mm=65,
        out_path=out_path,
        dpi=dpi,
    )


def write_plot_record(out_dir, csv_path, single_metrics, outputs, smooth_window, dpi, prefix, max_step):
    lines = [
        "# Plot Record",
        "",
        f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Input CSV: `{csv_path}`",
        f"- Output directory: `{out_dir}`",
        f"- Prefix: `{prefix}`",
        f"- Smooth window: `{smooth_window}`",
        f"- DPI: `{dpi}`",
        f"- Max step: `{max_step}`",
        "",
        "## Single Metrics",
        "",
    ]
    for metric_name in single_metrics:
        lines.append(f"- `{metric_name}`")
    lines.extend(["", "## Outputs", ""])
    for path in outputs:
        lines.append(f"- `{path}`")
    record_path = Path(out_dir) / "plot_record.md"
    record_path.write_text("\n".join(lines) + "\n")
    return record_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot single-metric RL reward-trace curves.")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--smooth_window", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--max_step", type=int, default=1000)
    return parser.parse_args()


# =========================================================
# Part 3. Main
# =========================================================
if __name__ == "__main__":
    args = parse_args()
    csv_path = Path(args.csv_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    df = pd.read_csv(csv_path)
    df = filter_dataframe_by_max_step(df, args.max_step)

    validate_metric_columns(df, DEFAULT_SINGLE_METRICS)
    ensure_dir(out_dir / "dummy.txt")

    prefix = args.prefix or "rl_trace"
    outputs = []

    for metric_name in DEFAULT_SINGLE_METRICS:
        pdf_path = out_dir / f"single_{metric_name}.pdf"
        png_path = out_dir / f"single_{metric_name}.png"
        plot_single_metric(df, metric_name, str(pdf_path), smooth_window=args.smooth_window, dpi=args.dpi)
        plot_single_metric(df, metric_name, str(png_path), smooth_window=args.smooth_window, dpi=args.dpi)
        outputs.extend([str(pdf_path), str(png_path)])

    record_path = write_plot_record(
        out_dir=out_dir,
        csv_path=csv_path,
        single_metrics=DEFAULT_SINGLE_METRICS,
        outputs=outputs,
        smooth_window=args.smooth_window,
        dpi=args.dpi,
        prefix=prefix,
        max_step=args.max_step,
    )

    print(
        json.dumps(
            {
                "csv_path": str(csv_path),
                "out_dir": str(out_dir),
                "max_step": args.max_step,
                "outputs": outputs,
                "plot_record": str(record_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


# python model/reinforce/plot_rl.py \
#   --csv_path runs/ppo_head_to_tail_perm_hemo_v3_full_finetune_lam1_stable_b64_lr3e6_no_trainer_state_20260603/reward_trace.csv \
#   --out_dir runs/ppo_head_to_tail_perm_hemo_v3_full_finetune_lam1_stable_b64_lr3e6_no_trainer_state_20260603/reward_trace_plots \
#   --prefix ppo_head_to_tail_perm_hemo_v3_full_finetune_lam1_stable_b64_lr3e6 \
#   --smooth_window 10 \
#   --dpi 600 \
#   --max_step 3000
