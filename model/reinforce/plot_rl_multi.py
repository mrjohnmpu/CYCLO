#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "mean_reward_total",
    "valid_rate",
    "mean_perm_raw",
    "mean_hemo_prob",
    "mean_hemo_safe",
]

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "mean_reward_total": {
        "title": "Total reward",
        "ylabel": "Mean total reward",
        "ylim": None,
        "zero_line": True,
    },
    "valid_rate": {
        "title": "Validity",
        "ylabel": "Valid rate",
        "ylim": "auto",
        "ylim_bounds": (0.0, 1.0),
    },
    "mean_perm_raw": {
        "title": "Permeability",
        "ylabel": "Mean permeability",
        "ylim": None,
    },
    "mean_hemo_prob": {
        "title": "Hemolysis probability",
        "ylabel": "Mean hemolysis probability",
        "ylim": (0.0, 1.0),
    },
    "mean_hemo_safe": {
        "title": "Hemolysis safety",
        "ylabel": "Mean hemolysis safety",
        "ylim": (0.0, 1.0),
    },
}

DEFAULT_COLORS = [
    "#2C7FB8",
    "#D95F02",
    "#1B9E77",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#E6AB02",
    "#A6761D",
]
DEFAULT_ARIAL_FONT_PATHS = [
    Path("/usr/share/fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/arial.ttf"),
]


@dataclass(frozen=True)
class TraceSpec:
    label: str
    csv_path: Path
    color: str | None = None
    linestyle: str = "-"


def mm_to_inches(value: float) -> float:
    return value / 25.4


def resolve_font(font_path: Path | None = None) -> tuple[str, str | None]:
    candidates = [font_path] if font_path is not None else DEFAULT_ARIAL_FONT_PATHS
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            font_name = font_manager.FontProperties(fname=str(candidate)).get_name()
            return font_name, str(candidate)
    return "Arial", None


def set_nature_style(font_size: float = 7.0, font_family: str = "Arial") -> None:
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": [font_family, "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def apply_axis_style(
    ax: plt.Axes,
    *,
    xlabel: str = "Step",
    ylabel: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: np.ndarray | None = None,
) -> None:
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def infer_xticks(max_step: int, max_ticks: int = 7) -> np.ndarray:
    if max_step <= 0:
        return np.array([0])

    raw_interval = max_step / max(1, max_ticks - 1)
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
    return np.arange(0, max_step + interval, interval)


def parse_trace_arg(trace_arg: str) -> TraceSpec:
    parts = trace_arg.split("=", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(f"Invalid --trace value {trace_arg!r}; expected label=/path/to/reward_trace.csv")
    return TraceSpec(label=parts[0].strip(), csv_path=Path(parts[1].strip()))


def load_manifest(path: Path) -> tuple[list[TraceSpec], list[str] | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        trace_items = payload
        manifest_metrics = None
    elif isinstance(payload, dict):
        trace_items = payload.get("traces")
        manifest_metrics = payload.get("metrics")
    else:
        raise ValueError("Manifest must be either a list of traces or an object with a 'traces' field")

    if not isinstance(trace_items, list) or not trace_items:
        raise ValueError("Manifest must contain at least one trace")

    traces: list[TraceSpec] = []
    for index, item in enumerate(trace_items):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest trace {index} must be an object")
        label = str(item.get("label", "")).strip()
        csv_value = item.get("csv_path", item.get("path", ""))
        csv_path = Path(str(csv_value).strip())
        if not label or not str(csv_path):
            raise ValueError(f"Manifest trace {index} requires 'label' and 'csv_path'")
        traces.append(
            TraceSpec(
                label=label,
                csv_path=csv_path,
                color=item.get("color"),
                linestyle=item.get("linestyle", "-"),
            )
        )

    if manifest_metrics is not None:
        if not isinstance(manifest_metrics, list) or not all(isinstance(metric, str) for metric in manifest_metrics):
            raise ValueError("Manifest 'metrics' must be a list of metric names")
    return traces, manifest_metrics


def collect_traces(args: argparse.Namespace) -> list[TraceSpec]:
    traces: list[TraceSpec] = []
    if args.manifest is not None:
        manifest_traces, manifest_metrics = load_manifest(args.manifest)
        traces.extend(manifest_traces)
        if args.metrics is None and manifest_metrics is not None:
            args.metrics = manifest_metrics
    if args.trace:
        traces.extend(parse_trace_arg(trace_arg) for trace_arg in args.trace)
    if not traces:
        raise ValueError("Provide at least one --trace label=path or --manifest JSON file")
    return traces


def read_trace(trace: TraceSpec, metrics: list[str], max_step: int | None) -> pd.DataFrame:
    if not trace.csv_path.exists():
        raise FileNotFoundError(f"Reward trace CSV not found for {trace.label!r}: {trace.csv_path}")

    dataframe = pd.read_csv(trace.csv_path)
    required = ["step", *metrics]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing columns in {trace.csv_path}: {missing}")

    dataframe = dataframe.loc[:, required].copy()
    dataframe["step"] = pd.to_numeric(dataframe["step"], errors="coerce")
    for metric in metrics:
        dataframe[metric] = pd.to_numeric(dataframe[metric], errors="coerce")
    dataframe = dataframe.dropna(subset=["step"]).sort_values("step").reset_index(drop=True)
    if max_step is not None:
        dataframe = dataframe[dataframe["step"] <= int(max_step)].copy()
    if dataframe.empty:
        raise ValueError(f"No rows remain for {trace.label!r} after filtering")
    return dataframe


def assign_colors(traces: list[TraceSpec]) -> dict[str, str]:
    colors: dict[str, str] = {}
    for index, trace in enumerate(traces):
        colors[trace.label] = trace.color or DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
    return colors


def resolve_ylim(
    spec: dict[str, Any],
    trace_data: dict[str, pd.DataFrame],
    metric: str,
) -> tuple[float, float] | None:
    ylim = spec.get("ylim")
    if ylim != "auto":
        return ylim

    values = pd.concat(
        [dataframe[metric] for dataframe in trace_data.values()],
        ignore_index=True,
    )
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None

    y_min = float(values.min())
    y_max = float(values.max())
    if y_min == y_max:
        span = max(abs(y_min) * 0.02, 0.02)
    else:
        span = y_max - y_min

    lower = y_min - span * 0.12
    upper = y_max + span * 0.12
    bounds = spec.get("ylim_bounds")
    if bounds is not None:
        lower = max(float(bounds[0]), lower)
        upper = min(float(bounds[1]), upper)

    if lower >= upper:
        return None
    return lower, upper


def should_draw_zero_line(
    spec: dict[str, Any],
    trace_data: dict[str, pd.DataFrame],
    metric: str,
) -> bool:
    if not spec.get("zero_line", False):
        return False

    values = pd.concat(
        [dataframe[metric] for dataframe in trace_data.values()],
        ignore_index=True,
    )
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return False
    return float(values.min()) < 0 < float(values.max())


def plot_metric(
    *,
    metric: str,
    trace_data: dict[str, pd.DataFrame],
    traces: list[TraceSpec],
    colors: dict[str, str],
    out_dir: Path,
    prefix: str,
    formats: list[str],
    smooth_window: int,
    show_raw: bool,
    show_smooth: bool,
    draw_zero_line: bool,
    single_width_mm: float,
    single_height_mm: float,
    max_xticks: int,
    pad_inches: float,
    dpi: int,
) -> list[Path]:
    spec = METRIC_SPECS.get(metric, {"title": metric, "ylabel": metric, "ylim": None})
    max_step = int(max(dataframe["step"].max() for dataframe in trace_data.values()))
    xticks = infer_xticks(max_step, max_ticks=max_xticks)
    ylim = resolve_ylim(spec, trace_data, metric)

    fig_w = mm_to_inches(single_width_mm)
    fig_h = mm_to_inches(single_height_mm)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    for trace in traces:
        dataframe = trace_data[trace.label]
        x = dataframe["step"]
        y = dataframe[metric]
        color = colors[trace.label]
        if show_raw:
            ax.plot(
                x,
                y,
                color=color,
                alpha=0.16 if show_smooth else 0.9,
                linewidth=0.8,
                linestyle=trace.linestyle,
                label=trace.label if not show_smooth else None,
            )
        if show_smooth:
            ax.plot(
                x,
                rolling_mean(y, smooth_window),
                color=color,
                linewidth=1.9,
                linestyle=trace.linestyle,
                label=trace.label,
            )

    if draw_zero_line and should_draw_zero_line(spec, trace_data, metric):
        ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)

    apply_axis_style(
        ax,
        xlabel="Step",
        ylabel=spec.get("ylabel", metric),
        title=spec.get("title", metric),
        xlim=(0, max_step),
        ylim=ylim,
        xticks=xticks,
    )
    ax.legend(frameon=False, loc="best", handlelength=2.0)
    fig.tight_layout()

    outputs: list[Path] = []
    for fmt in formats:
        out_path = out_dir / f"{prefix}_{metric}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        outputs.append(out_path)
    plt.close(fig)
    return outputs


def plot_composite(
    *,
    metrics: list[str],
    trace_data: dict[str, pd.DataFrame],
    traces: list[TraceSpec],
    colors: dict[str, str],
    out_dir: Path,
    prefix: str,
    formats: list[str],
    smooth_window: int,
    show_raw: bool,
    show_smooth: bool,
    draw_zero_line: bool,
    panel_width_mm: float,
    panel_height_mm: float,
    max_xticks: int,
    pad_inches: float,
    dpi: int,
) -> list[Path]:
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    max_step = int(max(dataframe["step"].max() for dataframe in trace_data.values()))
    xticks = infer_xticks(max_step, max_ticks=max_xticks)

    legend_height_mm = 10 if len(traces) > 1 else 0
    fig_w = mm_to_inches(panel_width_mm * n_cols)
    fig_h = mm_to_inches(panel_height_mm * n_rows + legend_height_mm)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    flat_axes = axes.ravel()

    for ax, metric in zip(flat_axes, metrics):
        spec = METRIC_SPECS.get(metric, {"title": metric, "ylabel": metric, "ylim": None})
        ylim = resolve_ylim(spec, trace_data, metric)
        for trace in traces:
            dataframe = trace_data[trace.label]
            x = dataframe["step"]
            y = dataframe[metric]
            color = colors[trace.label]
            if show_raw:
                ax.plot(
                    x,
                    y,
                    color=color,
                    alpha=0.12 if show_smooth else 0.85,
                    linewidth=0.6,
                    linestyle=trace.linestyle,
                )
            if show_smooth:
                ax.plot(
                    x,
                    rolling_mean(y, smooth_window),
                    color=color,
                    linewidth=1.5,
                    linestyle=trace.linestyle,
                    label=trace.label,
                )
        if draw_zero_line and should_draw_zero_line(spec, trace_data, metric):
            ax.axhline(0, color="0.5", linestyle="--", linewidth=0.7)
        apply_axis_style(
            ax,
            xlabel="Step",
            ylabel=spec.get("ylabel", metric),
            title=spec.get("title", metric),
            xlim=(0, max_step),
            ylim=ylim,
            xticks=xticks,
        )

    for ax in flat_axes[n_metrics:]:
        ax.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(len(labels), 4))
    fig.tight_layout(rect=(0, 0, 1, 0.94 if handles else 1))

    outputs: list[Path] = []
    for fmt in formats:
        out_path = out_dir / f"{prefix}_composite.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        outputs.append(out_path)
    plt.close(fig)
    return outputs


def write_record(
    *,
    out_dir: Path,
    traces: list[TraceSpec],
    trace_data: dict[str, pd.DataFrame],
    metrics: list[str],
    outputs: list[Path],
    args: argparse.Namespace,
) -> Path:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "out_dir": str(out_dir),
        "metrics": metrics,
        "smooth_window": args.smooth_window,
        "max_step": args.max_step,
        "show_raw": args.raw,
        "show_smooth": args.smooth,
        "draw_zero_line": not args.no_zero_line,
        "font_size": args.font_size,
        "single_width_mm": args.single_width_mm,
        "single_height_mm": args.single_height_mm,
        "composite_panel_width_mm": args.composite_panel_width_mm,
        "composite_panel_height_mm": args.composite_panel_height_mm,
        "max_xticks": args.max_xticks,
        "pad_inches": args.pad_inches,
        "font_family": args.resolved_font_family,
        "font_path": str(args.resolved_font_path) if args.resolved_font_path else None,
        "traces": [
            {
                "label": trace.label,
                "csv_path": str(trace.csv_path),
                "rows": int(len(trace_data[trace.label])),
                "step_min": float(trace_data[trace.label]["step"].min()),
                "step_max": float(trace_data[trace.label]["step"].max()),
            }
            for trace in traces
        ],
        "outputs": [str(path) for path in outputs],
    }
    record_path = out_dir / f"{args.prefix}_plot_record.json"
    record_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot multi-topology RL reward-trace curves from multiple reward_trace.csv files."
    )
    parser.add_argument(
        "--trace",
        action="append",
        default=[],
        help="Trace definition in the form label=/path/to/reward_trace.csv. Can be repeated.",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSON manifest with traces.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="multi_rl_trace")
    parser.add_argument("--metrics", nargs="+", default=None, help="Metrics to plot. Defaults to core RL metrics.")
    parser.add_argument("--smooth_window", type=int, default=10)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Output formats, e.g. pdf png svg.")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--composite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--font_size", type=float, default=7.0, help="Base matplotlib font size in points.")
    parser.add_argument(
        "--font_path",
        type=Path,
        default=None,
        help="Optional path to a TrueType/OpenType font file. Defaults to common Arial paths when available.",
    )
    parser.add_argument("--single_width_mm", type=float, default=90.0, help="Single-metric figure width in mm.")
    parser.add_argument("--single_height_mm", type=float, default=65.0, help="Single-metric figure height in mm.")
    parser.add_argument(
        "--composite_panel_width_mm",
        type=float,
        default=60.0,
        help="Approximate width of each composite subplot panel in mm.",
    )
    parser.add_argument(
        "--composite_panel_height_mm",
        type=float,
        default=58.0,
        help="Approximate height of each composite subplot panel in mm.",
    )
    parser.add_argument("--no_zero_line", action="store_true", help="Disable the horizontal y=0 reference line.")
    parser.add_argument("--max_xticks", type=int, default=7, help="Maximum target number of x-axis ticks.")
    parser.add_argument("--pad_inches", type=float, default=0.1, help="Padding around tight bounding boxes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smooth_window <= 0:
        raise ValueError("--smooth_window must be positive")
    if not args.raw and not args.smooth:
        raise ValueError("At least one of --raw or --smooth must be enabled")
    positive_size_args = {
        "--font_size": args.font_size,
        "--single_width_mm": args.single_width_mm,
        "--single_height_mm": args.single_height_mm,
        "--composite_panel_width_mm": args.composite_panel_width_mm,
        "--composite_panel_height_mm": args.composite_panel_height_mm,
    }
    invalid_sizes = {name: value for name, value in positive_size_args.items() if value <= 0}
    if invalid_sizes:
        raise ValueError(f"Figure size and font arguments must be positive: {invalid_sizes}")
    if args.font_path is not None and not args.font_path.exists():
        raise FileNotFoundError(f"Font file not found: {args.font_path}")
    if args.max_xticks <= 1:
        raise ValueError("--max_xticks must be greater than 1")
    if args.pad_inches < 0:
        raise ValueError("--pad_inches must be non-negative")

    traces = collect_traces(args)
    metrics = args.metrics or DEFAULT_METRICS
    trace_data = {trace.label: read_trace(trace, metrics, args.max_step) for trace in traces}

    font_family, font_path = resolve_font(args.font_path)
    args.resolved_font_family = font_family
    args.resolved_font_path = font_path
    set_nature_style(font_size=args.font_size, font_family=font_family)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    colors = assign_colors(traces)
    draw_zero_line = not args.no_zero_line

    outputs: list[Path] = []
    for metric in metrics:
        outputs.extend(
            plot_metric(
                metric=metric,
                trace_data=trace_data,
                traces=traces,
                colors=colors,
                out_dir=args.out_dir,
                prefix=args.prefix,
                formats=args.formats,
                smooth_window=args.smooth_window,
                show_raw=args.raw,
                show_smooth=args.smooth,
                draw_zero_line=draw_zero_line,
                single_width_mm=args.single_width_mm,
                single_height_mm=args.single_height_mm,
                max_xticks=args.max_xticks,
                pad_inches=args.pad_inches,
                dpi=args.dpi,
            )
        )
    if args.composite and len(metrics) > 1:
        outputs.extend(
            plot_composite(
                metrics=metrics,
                trace_data=trace_data,
                traces=traces,
                colors=colors,
                out_dir=args.out_dir,
                prefix=args.prefix,
                formats=args.formats,
                smooth_window=args.smooth_window,
                show_raw=args.raw,
                show_smooth=args.smooth,
                draw_zero_line=draw_zero_line,
                panel_width_mm=args.composite_panel_width_mm,
                panel_height_mm=args.composite_panel_height_mm,
                max_xticks=args.max_xticks,
                pad_inches=args.pad_inches,
                dpi=args.dpi,
            )
        )

    record_path = write_record(
        out_dir=args.out_dir,
        traces=traces,
        trace_data=trace_data,
        metrics=metrics,
        outputs=outputs,
        args=args,
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "metrics": metrics,
                "traces": [trace.label for trace in traces],
                "outputs": [str(path) for path in outputs],
                "plot_record": str(record_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()


# Example:
# python ./model/reinforce/plot_rl_multi.py \
#   --trace head-to-tail=runs/ppo_head_to_tail_perm_hemo_v3_full_finetune_lam1_stable_b64_lr3e6_no_trainer_state_20260603/reward_trace.csv \
#   --trace sc-to-tail=runs/ppo_sc_to_tail_perm_hemo_v3_full_finetune_lam1_stable_b64_lr3e6_no_trainer_state_20260603/reward_trace.csv \
#   --trace disulfide-bridge=runs/ppo_disulfide_bridge_perm_hemo_v3_full_finetune_len512_b64_mini8_lr3e6_kl007_tkl003_20260612/reward_trace.csv \
#   --out_dir runs/rl_multi_trace_plots_20260612 \
#   --prefix topology_comparison \
#   --smooth_window 10 \
#   --metrics mean_reward_total valid_rate mean_perm_raw mean_hemo_prob mean_hemo_safe
