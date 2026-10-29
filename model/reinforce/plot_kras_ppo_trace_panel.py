#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


DEFAULT_TRACE_CSV = Path(
    'runs/ppo_head_to_tail_perm_hemo_kras_v2_lam1_b128_full_finetune_s500_20260524/reward_trace.csv'
)
DEFAULT_OUT_DIR = Path(
    'runs/ppo_head_to_tail_perm_hemo_kras_v2_lam1_b128_full_finetune_s500_20260524/figures_kras_trace_panel_step300_20260618'
)
DEFAULT_METRICS = [
    "mean_reward_total",
    "valid_rate",
    "mean_perm_raw",
    "mean_hemo_prob",
    "mean_hemo_safe",
    "mean_kras_kd_pred",
]
DEFAULT_FORMATS = ["png", "pdf", "svg"]
MM_PER_INCH = 25.4
NATURE_COLORS = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "teal": "#42949E",
    "green": "#8BCF8B",
    "orange": "#D95F02",
    "red_strong": "#B64342",
    "violet": "#9A4D8E",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "warning_soft": "#F6CFCB",
}
METRIC_SPECS: dict[str, dict[str, Any]] = {
    "mean_reward_total": {
        "title": "Mean total reward",
        "ylabel": "mean_reward_total",
        "direction": "higher is better",
        "color": NATURE_COLORS["blue_main"],
    },
    "valid_rate": {
        "title": "Validity rate",
        "ylabel": "valid_rate",
        "direction": "higher is better",
        "color": NATURE_COLORS["teal"],
        "ylim": (0.0, 1.05),
    },
    "mean_perm_raw": {
        "title": "Mean PAMPA value",
        "ylabel": "mean_perm_raw",
        "direction": "higher is better",
        "color": NATURE_COLORS["orange"],
    },
    "mean_hemo_prob": {
        "title": "Mean hemolysis score",
        "ylabel": "mean_hemo_prob",
        "direction": "lower is better",
        "color": NATURE_COLORS["red_strong"],
        "ylim_lower": 0.0,
    },
    "mean_hemo_safe": {
        "title": "Mean hemo safety",
        "ylabel": "mean_hemo_safe",
        "direction": "higher is better",
        "color": NATURE_COLORS["blue_secondary"],
        "ylim": (0.45, 1.05),
    },
    "mean_kras_score": {
        "title": "Mean KRAS score",
        "ylabel": "mean_kras_score",
        "direction": "higher is better",
        "color": NATURE_COLORS["green"],
    },
    "mean_kras_kd_pred": {
        "title": "Mean KRAS KD pred",
        "ylabel": "mean_kras_kd_pred",
        "direction": "lower is better",
        "color": NATURE_COLORS["violet"],
    },
}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    title: str
    ylabel: str
    direction: str
    color: str
    ylim: tuple[float, float] | None = None
    ylim_lower: float | None = None
    ylim_upper: float | None = None


def apply_nature_style(*, font_size: float = 7.0, font_path: Path | None = None) -> str:
    font_family = "Arial"
    sans_serif = ["Arial", "DejaVu Sans", "Liberation Sans"]
    if font_path is not None:
        if not font_path.exists():
            raise FileNotFoundError(f"Font file not found: {font_path}")
        font_manager.fontManager.addfont(str(font_path))
        font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
        sans_serif = [font_family, *sans_serif]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans_serif,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )
    return font_family


def parse_metric_spec(metric: str) -> MetricSpec:
    base = METRIC_SPECS.get(metric, {})
    return MetricSpec(
        name=metric,
        title=str(base.get("title", metric)),
        ylabel=str(base.get("ylabel", metric)),
        direction=str(base.get("direction", "")),
        color=str(base.get("color", NATURE_COLORS["blue_main"])),
        ylim=base.get("ylim"),
        ylim_lower=base.get("ylim_lower"),
        ylim_upper=base.get("ylim_upper"),
    )


def infer_ylim(values: pd.Series, spec: MetricSpec) -> tuple[float, float] | None:
    if spec.ylim is not None:
        return spec.ylim

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None

    y_min = float(numeric.min())
    y_max = float(numeric.max())
    if y_min == y_max:
        span = max(abs(y_min) * 0.05, 0.05)
    else:
        span = y_max - y_min

    lower = y_min - span * 0.10
    upper = y_max + span * 0.10
    if spec.ylim_lower is not None:
        lower = max(float(spec.ylim_lower), lower)
    if spec.ylim_upper is not None:
        upper = min(float(spec.ylim_upper), upper)
    if lower >= upper:
        return None
    return lower, upper


def read_trace(path: Path, metrics: list[str], max_step: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trace CSV not found: {path}")
    dataframe = pd.read_csv(path)
    required = ["step", *metrics]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    dataframe = dataframe.loc[:, required].copy()
    dataframe["step"] = pd.to_numeric(dataframe["step"], errors="coerce")
    for metric in metrics:
        dataframe[metric] = pd.to_numeric(dataframe[metric], errors="coerce")
    dataframe = dataframe.dropna(subset=["step"]).sort_values("step")
    dataframe = dataframe[dataframe["step"] <= max_step].reset_index(drop=True)
    if dataframe.empty:
        raise ValueError(f"No rows remain after filtering step <= {max_step}")
    return dataframe


def add_event_marks(
    ax: plt.Axes,
    *,
    recommended_step: int | None,
    collapse_start_step: int | None,
    max_step: int,
) -> None:
    if recommended_step is not None and recommended_step <= max_step:
        ax.axvline(recommended_step, color=NATURE_COLORS["neutral_dark"], linestyle="--", linewidth=0.8, alpha=0.9)
    if collapse_start_step is not None and collapse_start_step <= max_step:
        ax.axvspan(
            collapse_start_step,
            max_step,
            color=NATURE_COLORS["warning_soft"],
            alpha=0.45,
            linewidth=0,
        )


def plot_panel(
    dataframe: pd.DataFrame,
    *,
    metrics: list[str],
    title: str,
    max_step: int,
    recommended_step: int | None,
    collapse_start_step: int | None,
    out_stem: Path,
    formats: list[str],
    dpi: int,
    show_grid: bool,
    show_direction_text: bool,
    font_size: float,
    font_path: Path | None,
    pad_inches: float,
) -> list[Path]:
    apply_nature_style(font_size=font_size, font_path=font_path)

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = int(math.ceil(n_metrics / n_cols))
    fig_w = 183 / 25.4
    fig_h = max(55 * n_rows, 95) / 25.4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, squeeze=False)
    flat_axes = axes.ravel()

    for ax, metric in zip(flat_axes, metrics):
        spec = parse_metric_spec(metric)
        x = dataframe["step"]
        y = dataframe[metric]
        ax.plot(x, y, color=spec.color, marker="o", markersize=2.5, linewidth=1.15, markeredgewidth=0)
        add_event_marks(
            ax,
            recommended_step=recommended_step,
            collapse_start_step=collapse_start_step,
            max_step=max_step,
        )
        ax.set_title(spec.title, pad=3)
        ax.set_ylabel(spec.ylabel)
        ylim = infer_ylim(y, spec)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlim(0, max_step)
        if show_direction_text and spec.direction:
            ax.text(
                0.02,
                0.92,
                spec.direction,
                transform=ax.transAxes,
                ha="left",
                va="top",
                color=NATURE_COLORS["neutral_mid"],
                fontsize=font_size,
            )
        if show_grid:
            ax.grid(True, color="#E6E6E6", linewidth=0.55, alpha=0.85)
        else:
            ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in flat_axes[n_metrics:]:
        ax.set_visible(False)

    for ax in flat_axes[:n_metrics]:
        ax.set_xlabel("PPO step")

    legend_items: list[Any] = []
    if recommended_step is not None and recommended_step <= max_step:
        legend_items.append(
            Line2D(
                [0],
                [0],
                color=NATURE_COLORS["neutral_dark"],
                linestyle="--",
                linewidth=0.8,
                label=f"recommended checkpoint: step{recommended_step}",
            )
        )
    if collapse_start_step is not None and collapse_start_step <= max_step:
        legend_items.append(
            Patch(
                facecolor=NATURE_COLORS["warning_soft"],
                edgecolor="none",
                alpha=0.45,
                label=f"collapse region: step >= {collapse_start_step}",
            )
        )
    if legend_items:
        fig.legend(
            handles=legend_items,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=len(legend_items),
            frameon=False,
            handlelength=2.6,
        )

    fig.suptitle(title, y=0.995, fontsize=font_size)
    fig.tight_layout(rect=(0, 0.035 if legend_items else 0, 1, 0.975))

    outputs: list[Path] = []
    for fmt in formats:
        out_path = out_stem.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        outputs.append(out_path)
    plt.close(fig)
    return outputs


def plot_single_metric(
    dataframe: pd.DataFrame,
    *,
    metric: str,
    max_step: int,
    recommended_step: int | None,
    collapse_start_step: int | None,
    out_stem: Path,
    formats: list[str],
    dpi: int,
    show_grid: bool,
    show_direction_text: bool,
    font_size: float,
    font_path: Path | None,
    single_width_mm: float,
    single_height_mm: float,
    pad_inches: float,
) -> list[Path]:
    apply_nature_style(font_size=font_size, font_path=font_path)

    spec = parse_metric_spec(metric)
    fig_w = single_width_mm / MM_PER_INCH
    fig_h = single_height_mm / MM_PER_INCH
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x = dataframe["step"]
    y = dataframe[metric]
    ax.plot(x, y, color=spec.color, marker="o", markersize=2.6, linewidth=1.2, markeredgewidth=0)
    add_event_marks(
        ax,
        recommended_step=recommended_step,
        collapse_start_step=collapse_start_step,
        max_step=max_step,
    )
    ax.set_title(spec.title, pad=3)
    ax.set_xlabel("PPO step")
    ax.set_ylabel(spec.ylabel)
    ax.set_xlim(0, max_step)
    ylim = infer_ylim(y, spec)
    if ylim is not None:
        ax.set_ylim(ylim)
    if show_direction_text and spec.direction:
        ax.text(
            0.02,
            0.92,
            spec.direction,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=NATURE_COLORS["neutral_mid"],
            fontsize=font_size,
        )
    if show_grid:
        ax.grid(True, color="#E6E6E6", linewidth=0.55, alpha=0.85)
    else:
        ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    outputs: list[Path] = []
    for fmt in formats:
        out_path = out_stem.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        outputs.append(out_path)
    plt.close(fig)
    return outputs


def plot_single_metrics(
    dataframe: pd.DataFrame,
    *,
    metrics: list[str],
    max_step: int,
    recommended_step: int | None,
    collapse_start_step: int | None,
    out_dir: Path,
    prefix: str,
    formats: list[str],
    dpi: int,
    show_grid: bool,
    show_direction_text: bool,
    font_size: float,
    font_path: Path | None,
    single_width_mm: float,
    single_height_mm: float,
    pad_inches: float,
) -> list[Path]:
    outputs: list[Path] = []
    for metric in metrics:
        outputs.extend(
            plot_single_metric(
                dataframe,
                metric=metric,
                max_step=max_step,
                recommended_step=recommended_step,
                collapse_start_step=collapse_start_step,
                out_stem=out_dir / f"{prefix}_{metric}",
                formats=formats,
                dpi=dpi,
                show_grid=show_grid,
                show_direction_text=show_direction_text,
                font_size=font_size,
                font_path=font_path,
                single_width_mm=single_width_mm,
                single_height_mm=single_height_mm,
                pad_inches=pad_inches,
            )
        )
    return outputs


def write_record(
    *,
    out_dir: Path,
    trace_csv: Path,
    dataframe: pd.DataFrame,
    metrics: list[str],
    outputs: list[Path],
    args: argparse.Namespace,
) -> Path:
    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": sys.argv,
        "trace_csv": str(trace_csv),
        "out_dir": str(out_dir),
        "metrics": metrics,
        "max_step": args.max_step,
        "recommended_step": args.recommended_step,
        "collapse_start_step": args.collapse_start_step,
        "formats": args.formats,
        "dpi": args.dpi,
        "single": args.single,
        "show_grid": args.show_grid,
        "show_direction_text": args.show_direction_text,
        "font_size": args.font_size,
        "font_path": str(args.font_path) if args.font_path is not None else None,
        "single_width_mm": args.single_width_mm,
        "single_height_mm": args.single_height_mm,
        "pad_inches": args.pad_inches,
        "rows": int(len(dataframe)),
        "step_min": float(dataframe["step"].min()),
        "step_max": float(dataframe["step"].max()),
        "outputs": [str(path) for path in outputs],
    }
    path = out_dir / f"{args.prefix}_plot_record.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a Nature-style multi-panel KRAS PPO reward trace.")
    parser.add_argument("--trace_csv", type=Path, default=DEFAULT_TRACE_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="kras_ppo_trace_panel_step300")
    parser.add_argument("--title", default="Head-to-tail perm + hemo + KRAS PPO v2 lam=1.0 b128 trace")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--max_step", type=int, default=300)
    parser.add_argument("--recommended_step", type=int, default=300)
    parser.add_argument("--no_recommended_step", action="store_true", help="Disable the recommended-step marker line and legend item.")
    parser.add_argument("--collapse_start_step", type=int, default=None)
    parser.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--single", action="store_true", help="Write one standalone figure per metric instead of one panel figure")
    parser.add_argument("--grid", dest="show_grid", action="store_true", default=True)
    parser.add_argument("--no-grid", dest="show_grid", action="store_false")
    parser.add_argument("--direction-text", dest="show_direction_text", action="store_true", default=True)
    parser.add_argument("--no-direction-text", dest="show_direction_text", action="store_false")
    parser.add_argument("--font_size", type=float, default=7.0)
    parser.add_argument("--font_path", type=Path, default=None)
    parser.add_argument("--single_width_mm", type=float, default=90.0)
    parser.add_argument("--single_height_mm", type=float, default=65.0)
    parser.add_argument("--pad_inches", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.formats = [fmt.lower().lstrip(".") for fmt in args.formats]
    unsupported = [fmt for fmt in args.formats if fmt not in {"png", "pdf", "svg"}]
    if unsupported:
        raise ValueError(f"Unsupported output format(s): {unsupported}")
    if args.max_step <= 0:
        raise ValueError("--max_step must be positive")
    if args.single_width_mm <= 0 or args.single_height_mm <= 0:
        raise ValueError("--single_width_mm and --single_height_mm must be positive")
    if args.font_size <= 0:
        raise ValueError("--font_size must be positive")
    if args.pad_inches < 0:
        raise ValueError("--pad_inches must be non-negative")
    if args.no_recommended_step:
        args.recommended_step = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataframe = read_trace(args.trace_csv, args.metrics, args.max_step)
    if args.single:
        outputs = plot_single_metrics(
            dataframe,
            metrics=args.metrics,
            max_step=args.max_step,
            recommended_step=args.recommended_step,
            collapse_start_step=args.collapse_start_step,
            out_dir=args.out_dir,
            prefix=args.prefix,
            formats=args.formats,
            dpi=args.dpi,
            show_grid=args.show_grid,
            show_direction_text=args.show_direction_text,
            font_size=args.font_size,
            font_path=args.font_path,
            single_width_mm=args.single_width_mm,
            single_height_mm=args.single_height_mm,
            pad_inches=args.pad_inches,
        )
    else:
        out_stem = args.out_dir / args.prefix
        outputs = plot_panel(
            dataframe,
            metrics=args.metrics,
            title=args.title,
            max_step=args.max_step,
            recommended_step=args.recommended_step,
            collapse_start_step=args.collapse_start_step,
            out_stem=out_stem,
            formats=args.formats,
            dpi=args.dpi,
            show_grid=args.show_grid,
            show_direction_text=args.show_direction_text,
            font_size=args.font_size,
            font_path=args.font_path,
            pad_inches=args.pad_inches,
        )
    record_path = write_record(
        out_dir=args.out_dir,
        trace_csv=args.trace_csv,
        dataframe=dataframe,
        metrics=args.metrics,
        outputs=outputs,
        args=args,
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "outputs": [str(path) for path in outputs],
                "plot_record": str(record_path),
                "rows": int(len(dataframe)),
                "step_min": float(dataframe["step"].min()),
                "step_max": float(dataframe["step"].max()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
