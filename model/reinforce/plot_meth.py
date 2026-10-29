#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import sys
import warnings
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

MM_PER_INCH = 25.4
PUBLICATION_FONT_NAME = "Arial"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "methylation_shift.json"
DEFAULT_OUTPUT_DIR = Path('results/figures/methylation_shift_20260618')
DEFAULT_FORMATS = ["png", "pdf", "svg"]
DEFAULT_DPI = 600
DEFAULT_STAGE_ORDER = ["before", "after"]

TOPOLOGY_LABELS = {
    "head_to_tail": "Head-to-tail",
    "head-to-tail": "Head-to-tail",
    "sc_to_tail": "SC-to-tail",
    "sc-to-tail": "SC-to-tail",
    "disulfide_bridge": "Disulfide bridge",
    "disulfide-bridge": "Disulfide bridge",
    "disulfidge-bridge": "Disulfide bridge",
}

STAGE_LABELS = {
    "before": "Before RL",
    "after": "After RL",
}

STAGE_COLORS = {
    "before": "#8E9AAF",
    "after": "#D98973",
}

PUBLICATION_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": [PUBLICATION_FONT_NAME, "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": DEFAULT_DPI,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-sequence N-methylation fraction shifts before and after reinforcement learning."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--formats", nargs="+", default=None, choices=["png", "pdf", "svg"])
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=None)
    return parser.parse_args()


def apply_publication_style() -> None:
    plt.rcParams.update(PUBLICATION_RCPARAMS)
    try:
        font_manager.findfont(font_manager.FontProperties(family=PUBLICATION_FONT_NAME), fallback_to_default=False)
    except ValueError:
        warnings.warn(f"{PUBLICATION_FONT_NAME} is not available; using the closest sans-serif fallback.", RuntimeWarning)


def normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def default_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def topology_label(topology_id: str) -> str:
    return TOPOLOGY_LABELS.get(topology_id, default_label(topology_id))


def stage_label(stage_id: str) -> str:
    return STAGE_LABELS.get(stage_id, default_label(stage_id))


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config.get("datasets"), list) or not config["datasets"]:
        raise ValueError("Config must contain a non-empty 'datasets' list.")
    return config


def prepare_output_layout(out_dir: Path) -> dict[str, Path]:
    layout = {
        "root": out_dir,
        "figures": out_dir / "figures",
        "tables": out_dir / "tables",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def split_residue_sequence(sequence: object) -> list[str]:
    if sequence is None or pd.isna(sequence):
        return []
    return [token.strip() for token in str(sequence).split("|") if token.strip()]


def is_n_methyl_residue(token: str) -> bool:
    lowered = token.strip().lower()
    if not lowered:
        return False
    return "[nmet]" in lowered or "n-me" in lowered or "nme" in lowered or "n-methyl" in lowered


def valid_length(value: object) -> float:
    length = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(length) or float(length) <= 0:
        return math.nan
    return float(length)


def compute_methylation(sequence: object, peptide_length: object) -> dict[str, float | int | str]:
    residues = split_residue_sequence(sequence)
    length = valid_length(peptide_length)
    if not np.isfinite(length):
        return {
            "residue_token_count": len(residues),
            "peptide_length_numeric": math.nan,
            "n_methyl_residue_count": math.nan,
            "n_methyl_residue_fraction": math.nan,
            "n_methyl_residue_tokens": "",
        }
    methyl_tokens = [token for token in residues if is_n_methyl_residue(token)]
    return {
        "residue_token_count": len(residues),
        "peptide_length_numeric": length,
        "n_methyl_residue_count": len(methyl_tokens),
        "n_methyl_residue_fraction": len(methyl_tokens) / length,
        "n_methyl_residue_tokens": "|".join(methyl_tokens),
    }


def read_dataset(dataset: dict[str, Any], max_rows_per_dataset: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    topology_id = normalize_key(str(dataset.get("topology") or dataset.get("id") or ""))
    stage_id = normalize_key(str(dataset.get("stage") or ""))
    if not topology_id or not stage_id:
        raise ValueError(f"Each dataset must define topology/id and stage: {dataset!r}")
    path = Path(str(dataset.get("path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV for {topology_id}/{stage_id}: {path}")

    frame = pd.read_csv(path)
    raw_rows = len(frame)
    if max_rows_per_dataset is not None and max_rows_per_dataset > 0:
        frame = frame.head(max_rows_per_dataset).copy()

    missing_columns = [column for column in ("peptide_sequence", "peptide_length") if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    status_filter = str(dataset.get("parse_status") or "success").strip().lower()
    if status_filter and "parse_status" in frame.columns:
        frame = frame[frame["parse_status"].astype(str).str.lower() == status_filter].copy()
        filter_mode = f"parse_status == {status_filter}"
    else:
        filter_mode = "kept all rows"

    methylation_rows = [
        compute_methylation(sequence, length)
        for sequence, length in zip(frame["peptide_sequence"], frame["peptide_length"], strict=False)
    ]
    methylation = pd.DataFrame(methylation_rows, index=frame.index)
    result = pd.concat([frame.reset_index(drop=True), methylation.reset_index(drop=True)], axis=1)
    result["topology"] = topology_id
    result["topology_label"] = str(dataset.get("topology_label") or dataset.get("label") or topology_label(topology_id))
    result["stage"] = stage_id
    result["stage_label"] = str(dataset.get("stage_label") or stage_label(stage_id))
    result["source_path"] = str(path)

    summary = {
        "topology": topology_id,
        "topology_label": result["topology_label"].iloc[0] if len(result) else topology_label(topology_id),
        "stage": stage_id,
        "stage_label": result["stage_label"].iloc[0] if len(result) else stage_label(stage_id),
        "path": str(path),
        "raw_rows": raw_rows,
        "rows_after_max_rows": raw_rows if max_rows_per_dataset is None else min(raw_rows, max_rows_per_dataset),
        "rows_after_filter": len(result),
        "filter_mode": filter_mode,
        "missing_required_columns": ";".join(missing_columns),
        "valid_methyl_fraction_rows": int(result["n_methyl_residue_fraction"].notna().sum()),
        "token_length_mismatch_rows": int(
            (
                result["peptide_length_numeric"].notna()
                & (result["residue_token_count"] != result["peptide_length_numeric"].round().astype("Int64"))
            ).sum()
        ),
    }
    return result, summary


def load_datasets(
    datasets: list[dict[str, Any]],
    max_rows_per_dataset: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for dataset in datasets:
        topology_id = normalize_key(str(dataset.get("topology") or dataset.get("id") or ""))
        stage_id = normalize_key(str(dataset.get("stage") or ""))
        key = (topology_id, stage_id)
        if key in seen:
            raise ValueError(f"Duplicate topology/stage dataset: {key}")
        seen.add(key)
        frame, summary = read_dataset(dataset, max_rows_per_dataset)
        frames.append(frame)
        summaries.append(summary)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return data, pd.DataFrame(summaries)


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_numeric(group["n_methyl_residue_fraction"], errors="coerce").dropna().to_numpy(dtype=float)
    counts = pd.to_numeric(group["n_methyl_residue_count"], errors="coerce").dropna().to_numpy(dtype=float)
    lengths = pd.to_numeric(group["peptide_length_numeric"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "iqr": math.nan,
            "min": math.nan,
            "max": math.nan,
            "zero_fraction": math.nan,
            "sequence_fraction_with_n_methyl": math.nan,
            "total_methylated_residues": math.nan,
            "total_residues": math.nan,
            "overall_methylated_residue_fraction": math.nan,
        }
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    total_methylated = float(np.nansum(counts))
    total_residues = float(np.nansum(lengths))
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "zero_fraction": float(np.mean(values == 0)),
        "sequence_fraction_with_n_methyl": float(np.mean(values > 0)),
        "total_methylated_residues": int(total_methylated),
        "total_residues": int(total_residues),
        "overall_methylated_residue_fraction": float(total_methylated / total_residues) if total_residues > 0 else math.nan,
    }


def build_summary_tables(data: pd.DataFrame, topology_order: list[str], stage_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    for topology in topology_order:
        for stage in stage_order:
            group = data[(data["topology"] == topology) & (data["stage"] == stage)]
            if group.empty:
                continue
            row = {
                "topology": topology,
                "topology_label": group["topology_label"].iloc[0],
                "stage": stage,
                "stage_label": group["stage_label"].iloc[0],
            }
            row.update(summarize_group(group))
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    shift_rows: list[dict[str, Any]] = []
    for topology in topology_order:
        subset = summary[summary["topology"] == topology].set_index("stage")
        if not {"before", "after"}.issubset(set(subset.index)):
            continue
        before = subset.loc["before"]
        after = subset.loc["after"]
        shift_rows.append({
            "topology": topology,
            "topology_label": before["topology_label"],
            "before_n": int(before["n"]),
            "after_n": int(after["n"]),
            "mean_before": before["mean"],
            "mean_after": after["mean"],
            "mean_delta_after_minus_before": after["mean"] - before["mean"],
            "median_before": before["median"],
            "median_after": after["median"],
            "median_delta_after_minus_before": after["median"] - before["median"],
            "zero_fraction_before": before["zero_fraction"],
            "zero_fraction_after": after["zero_fraction"],
            "zero_fraction_delta_after_minus_before": after["zero_fraction"] - before["zero_fraction"],
            "sequence_fraction_with_n_methyl_before": before["sequence_fraction_with_n_methyl"],
            "sequence_fraction_with_n_methyl_after": after["sequence_fraction_with_n_methyl"],
            "sequence_fraction_with_n_methyl_delta_after_minus_before": (
                after["sequence_fraction_with_n_methyl"] - before["sequence_fraction_with_n_methyl"]
            ),
            "overall_methylated_residue_fraction_before": before["overall_methylated_residue_fraction"],
            "overall_methylated_residue_fraction_after": after["overall_methylated_residue_fraction"],
            "overall_methylated_residue_fraction_delta_after_minus_before": (
                after["overall_methylated_residue_fraction"] - before["overall_methylated_residue_fraction"]
            ),
        })
    shift = pd.DataFrame(shift_rows)
    return summary, shift


def build_length_bin_tables(
    data: pd.DataFrame,
    topology_order: list[str],
    stage_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = data.copy()
    work["peptide_length_bin"] = pd.to_numeric(work["peptide_length_numeric"], errors="coerce").round().astype("Int64")
    work = work[work["peptide_length_bin"].notna()].copy()

    summary_rows: list[dict[str, Any]] = []
    for topology in topology_order:
        topology_data = work[work["topology"] == topology]
        length_bins = sorted(int(value) for value in topology_data["peptide_length_bin"].dropna().unique())
        for length_bin in length_bins:
            for stage in stage_order:
                group = topology_data[
                    (topology_data["peptide_length_bin"] == length_bin)
                    & (topology_data["stage"] == stage)
                ]
                if group.empty:
                    continue
                row = {
                    "topology": topology,
                    "topology_label": group["topology_label"].iloc[0],
                    "peptide_length_bin": int(length_bin),
                    "stage": stage,
                    "stage_label": group["stage_label"].iloc[0],
                }
                row.update(summarize_group(group))
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    shift_rows: list[dict[str, Any]] = []
    if not summary.empty:
        for topology in topology_order:
            topology_summary = summary[summary["topology"] == topology]
            length_bins = sorted(int(value) for value in topology_summary["peptide_length_bin"].dropna().unique())
            for length_bin in length_bins:
                subset = topology_summary[topology_summary["peptide_length_bin"] == length_bin].set_index("stage")
                if not {"before", "after"}.issubset(set(subset.index)):
                    continue
                before = subset.loc["before"]
                after = subset.loc["after"]
                shift_rows.append({
                    "topology": topology,
                    "topology_label": before["topology_label"],
                    "peptide_length_bin": int(length_bin),
                    "before_n": int(before["n"]),
                    "after_n": int(after["n"]),
                    "mean_before": before["mean"],
                    "mean_after": after["mean"],
                    "mean_delta_after_minus_before": after["mean"] - before["mean"],
                    "median_before": before["median"],
                    "median_after": after["median"],
                    "median_delta_after_minus_before": after["median"] - before["median"],
                    "sequence_fraction_with_n_methyl_before": before["sequence_fraction_with_n_methyl"],
                    "sequence_fraction_with_n_methyl_after": after["sequence_fraction_with_n_methyl"],
                    "sequence_fraction_with_n_methyl_delta_after_minus_before": (
                        after["sequence_fraction_with_n_methyl"] - before["sequence_fraction_with_n_methyl"]
                    ),
                    "overall_methylated_residue_fraction_before": before["overall_methylated_residue_fraction"],
                    "overall_methylated_residue_fraction_after": after["overall_methylated_residue_fraction"],
                    "overall_methylated_residue_fraction_delta_after_minus_before": (
                        after["overall_methylated_residue_fraction"] - before["overall_methylated_residue_fraction"]
                    ),
                })
    shift = pd.DataFrame(shift_rows)
    return summary, shift


def downsample_for_scatter(values: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) <= max_points:
        return np.arange(len(values))
    return np.sort(rng.choice(np.arange(len(values)), size=max_points, replace=False))


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    topology: str,
    stage_order: list[str],
    rng: np.random.Generator,
    *,
    scatter_max_points: int,
) -> None:
    subset = data[data["topology"] == topology]
    topology_name = subset["topology_label"].iloc[0] if len(subset) else topology_label(topology)
    positions = np.arange(len(stage_order), dtype=float)

    for idx, stage in enumerate(stage_order):
        group = subset[subset["stage"] == stage]
        values = (
            pd.to_numeric(group["n_methyl_residue_fraction"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
        color = STAGE_COLORS.get(stage, "#999999")
        if len(values) == 0:
            continue

        violin = ax.violinplot(
            values,
            positions=[positions[idx]],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.45)
            body.set_linewidth(0.55)

        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        whisker_low, whisker_high = np.quantile(values, [0.05, 0.95])
        ax.plot([positions[idx], positions[idx]], [whisker_low, whisker_high], color="black", lw=0.65, zorder=4)
        ax.add_patch(
            plt.Rectangle(
                (positions[idx] - 0.16, q25),
                0.32,
                max(q75 - q25, 0.002),
                facecolor="white",
                edgecolor="black",
                lw=0.65,
                zorder=5,
            )
        )
        ax.plot([positions[idx] - 0.17, positions[idx] + 0.17], [median, median], color="black", lw=0.9, zorder=6)

        sample_idx = downsample_for_scatter(values, scatter_max_points, rng)
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(sample_idx))
        ax.scatter(
            np.full(len(sample_idx), positions[idx]) + jitter,
            values[sample_idx],
            s=3.0,
            color=color,
            alpha=0.13,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )

        row = summary[(summary["topology"] == topology) & (summary["stage"] == stage)]
        if not row.empty:
            ax.text(
                positions[idx],
                1.03,
                f"n={int(row['n'].iloc[0]):,}",
                ha="center",
                va="bottom",
                fontsize=5.7,
                color="#333333",
                transform=ax.get_xaxis_transform(),
            )

    shift_row = summary[summary["topology"] == topology].set_index("stage")
    title = topology_name
    if {"before", "after"}.issubset(set(shift_row.index)):
        median_delta = shift_row.loc["after", "median"] - shift_row.loc["before", "median"]
        mean_delta = shift_row.loc["after", "mean"] - shift_row.loc["before", "mean"]
        ax.text(
            0.5,
            0.955,
            f"\u0394median={median_delta:+.3f}; \u0394mean={mean_delta:+.3f}",
            ha="center",
            va="top",
            fontsize=5.9,
            color="#333333",
            transform=ax.transAxes,
        )
    ax.set_title(title, pad=13)
    ax.set_xticks(positions)
    ax.set_xticklabels([stage_label(stage) for stage in stage_order], rotation=0)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlim(-0.55, len(stage_order) - 0.45)
    ax.grid(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


def plot_methylation_shift(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    topology_order: list[str],
    stage_order: list[str],
    out_prefix: Path,
    formats: Iterable[str],
    dpi: int,
    scatter_max_points: int,
) -> list[str]:
    apply_publication_style()
    rng = np.random.default_rng(20260618)
    fig_w = 183 / MM_PER_INCH
    fig_h = 82 / MM_PER_INCH
    fig, axes = plt.subplots(1, len(topology_order), figsize=(fig_w, fig_h), sharey=True)
    if len(topology_order) == 1:
        axes = np.array([axes])
    fig.patch.set_facecolor("white")

    for idx, (ax, topology) in enumerate(zip(axes, topology_order, strict=True)):
        draw_panel(
            ax,
            data,
            summary,
            topology,
            stage_order,
            rng,
            scatter_max_points=scatter_max_points,
        )
        if idx == 0:
            ax.set_ylabel("Per-sequence N-methylation fraction")
        else:
            ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=STAGE_COLORS.get(stage, "#999999"),
                   markeredgecolor="black", markeredgewidth=0.5, markersize=6, alpha=0.75, label=stage_label(stage))
        for stage in stage_order
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.02), ncol=len(handles), frameon=False)
    fig.text(0.012, 0.985, "a", fontsize=13, fontweight="bold", ha="left", va="top")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.82, bottom=0.18, wspace=0.25)

    outputs: list[str] = []
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_prefix.with_suffix(f".{fmt}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def marker_size_from_n(n: float, max_n: float) -> float:
    if max_n <= 0 or not np.isfinite(n):
        return 18.0
    return 18.0 + 72.0 * np.sqrt(max(float(n), 0.0) / max_n)


def plot_methylation_by_length_shift(
    length_summary: pd.DataFrame,
    length_shift: pd.DataFrame,
    *,
    topology_order: list[str],
    stage_order: list[str],
    out_prefix: Path,
    formats: Iterable[str],
    dpi: int,
    min_n_per_stage: int,
) -> list[str]:
    apply_publication_style()
    plot_shift = length_shift[
        (length_shift["before_n"] >= min_n_per_stage)
        & (length_shift["after_n"] >= min_n_per_stage)
    ].copy()
    fig_w = 183 / MM_PER_INCH
    fig_h = 88 / MM_PER_INCH
    fig, axes = plt.subplots(1, len(topology_order), figsize=(fig_w, fig_h), sharey=True)
    if len(topology_order) == 1:
        axes = np.array([axes])
    fig.patch.set_facecolor("white")
    max_n = float(length_summary["n"].max()) if not length_summary.empty else 1.0

    for idx, (ax, topology) in enumerate(zip(axes, topology_order, strict=True)):
        topology_summary = length_summary[length_summary["topology"] == topology].copy()
        topology_plot_bins = set(plot_shift[plot_shift["topology"] == topology]["peptide_length_bin"].astype(int).tolist())
        topology_summary = topology_summary[topology_summary["peptide_length_bin"].astype(int).isin(topology_plot_bins)]
        topology_name = topology_label(topology)
        if not topology_summary.empty:
            topology_name = topology_summary["topology_label"].iloc[0]

        for stage in stage_order:
            stage_data = topology_summary[topology_summary["stage"] == stage].sort_values("peptide_length_bin")
            if stage_data.empty:
                continue
            color = STAGE_COLORS.get(stage, "#999999")
            x = stage_data["peptide_length_bin"].to_numpy(dtype=float)
            y = stage_data["mean"].to_numpy(dtype=float)
            q25 = stage_data["q25"].to_numpy(dtype=float)
            q75 = stage_data["q75"].to_numpy(dtype=float)
            ax.plot(x, y, color=color, lw=1.0, alpha=0.95, zorder=3)
            for x_value, low, high in zip(x, q25, q75, strict=False):
                ax.vlines(x_value, low, high, color=color, lw=0.65, alpha=0.78, zorder=2)
                ax.hlines([low, high], x_value - 0.08, x_value + 0.08, color=color, lw=0.65, alpha=0.78, zorder=2)
            sizes = [marker_size_from_n(value, max_n) for value in stage_data["n"]]
            ax.scatter(
                x,
                y,
                s=sizes,
                color=color,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.88,
                label=stage_label(stage),
                zorder=4,
            )

        topology_shift = plot_shift[plot_shift["topology"] == topology].sort_values("peptide_length_bin")
        if not topology_shift.empty:
            x = topology_shift["peptide_length_bin"].to_numpy(dtype=float)
            delta = topology_shift["mean_delta_after_minus_before"].to_numpy(dtype=float)
            y_base = np.full_like(x, -0.012)
            scale = 0.075 / max(np.nanmax(np.abs(delta)), 1e-9)
            ax.axhline(0, color="#B5B5B5", lw=0.55, zorder=1)
            ax.plot(x, y_base + delta * scale, color="#4A4A4A", lw=0.8, zorder=2)
            ax.fill_between(
                x,
                y_base,
                y_base + delta * scale,
                where=delta >= 0,
                color="#4DAF4A",
                alpha=0.18,
                interpolate=True,
                zorder=1,
            )
            ax.fill_between(
                x,
                y_base,
                y_base + delta * scale,
                where=delta < 0,
                color="#C44E52",
                alpha=0.18,
                interpolate=True,
                zorder=1,
            )
            mean_delta = float(np.nanmean(delta))
            ax.text(
                0.98,
                0.95,
                f"mean \u0394={mean_delta:+.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.9,
                color="#333333",
            )

        if not topology_summary.empty:
            xmin = int(topology_summary["peptide_length_bin"].min())
            xmax = int(topology_summary["peptide_length_bin"].max())
            ax.set_xticks(range(xmin, xmax + 1))
        ax.set_title(topology_name, pad=8)
        ax.set_xlabel("Peptide length")
        ax.set_ylim(-0.035, 0.38)
        ax.grid(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
        if idx == 0:
            ax.set_ylabel("Mean per-sequence N-methylation fraction")
        else:
            ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([0], [0], marker="o", color=STAGE_COLORS.get(stage, "#999999"),
                   markerfacecolor=STAGE_COLORS.get(stage, "#999999"), markeredgecolor="black",
                   markeredgewidth=0.45, markersize=5, lw=1.0, label=stage_label(stage))
        for stage in stage_order
    ]
    handles.append(plt.Line2D([0], [0], color="#4A4A4A", lw=0.8, label="After - before mean shift"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.02), ncol=3, frameon=False)
    fig.text(0.012, 0.985, "b", fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(
        0.995,
        0.02,
        f"Only length bins with both stages n\u2265{min_n_per_stage} are plotted; all bins are retained in tables.",
        ha="right",
        va="bottom",
        fontsize=5.6,
        color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.82, bottom=0.18, wspace=0.25)

    outputs: list[str] = []
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_prefix.with_suffix(f".{fmt}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def write_run_record(
    path: Path,
    *,
    config_path: Path,
    command: str,
    config: dict[str, Any],
    output_files: list[str],
    source_summary: pd.DataFrame,
    summary: pd.DataFrame,
    shift: pd.DataFrame,
    length_summary: pd.DataFrame | None = None,
    length_shift: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# N-methylation Shift Plot Record",
        "",
        f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Command: `{command}`",
        f"- Config: `{config_path}`",
        f"- Output directory: `{config.get('output_dir', DEFAULT_OUTPUT_DIR)}`",
        f"- Definition: `n_methyl_residue_fraction = explicit N-methyl residue count / peptide_length`",
        f"- Detection rule: `{config.get('n_methyl_rule', '[Nmet], N-Me, NMe, or N-methyl in peptide_sequence token')}`",
        "",
        "## Outputs",
        "",
    ]
    lines.extend(f"- `{item}`" for item in output_files)
    lines.extend([
        "",
        "## Sources",
        "",
    ])
    for _, row in source_summary.iterrows():
        lines.append(
            "- "
            f"`{row['topology']}` / `{row['stage']}`: `{row['path']}` "
            f"(raw={row['raw_rows']}, used={row['rows_after_filter']}, valid={row['valid_methyl_fraction_rows']})"
        )
    lines.extend([
        "",
        "## Summary",
        "",
    ])
    for _, row in summary.iterrows():
        lines.append(
            "- "
            f"{row['topology_label']} {row['stage_label']}: "
            f"median={row['median']:.4f}, mean={row['mean']:.4f}, "
            f"overall residue fraction={row['overall_methylated_residue_fraction']:.4f}, n={int(row['n'])}"
        )
    if not shift.empty:
        lines.extend([
            "",
            "## After - Before Shift",
            "",
        ])
        for _, row in shift.iterrows():
            lines.append(
                "- "
                f"{row['topology_label']}: "
                f"\u0394median={row['median_delta_after_minus_before']:+.4f}, "
                f"\u0394mean={row['mean_delta_after_minus_before']:+.4f}, "
                f"\u0394overall residue fraction={row['overall_methylated_residue_fraction_delta_after_minus_before']:+.4f}"
            )
    if length_shift is not None and not length_shift.empty:
        min_n = int(config.get("length_bin_min_n_per_stage", 50))
        plotted = length_shift[
            (length_shift["before_n"] >= min_n)
            & (length_shift["after_n"] >= min_n)
        ]
        lines.extend([
            "",
            "## Length-Binned Shift",
            "",
            f"- Plotted length bins require both stages `n >= {min_n}`.",
            f"- Length-bin rows written: `{0 if length_summary is None else len(length_summary)}`.",
            f"- Comparable length-bin shift rows written: `{len(length_shift)}`.",
            f"- Length-bin shift rows plotted: `{len(plotted)}`.",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out_dir = args.out_dir or Path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    formats = args.formats or config.get("formats", DEFAULT_FORMATS)
    dpi = int(args.dpi or config.get("dpi", DEFAULT_DPI))
    scatter_max_points = int(config.get("scatter_max_points", 800))
    length_bin_min_n_per_stage = int(config.get("length_bin_min_n_per_stage", 50))
    plot_length_bins = bool(config.get("plot_length_bins", True))
    topology_order = [normalize_key(item) for item in config.get("topology_order", [])]
    stage_order = [normalize_key(item) for item in config.get("stage_order", DEFAULT_STAGE_ORDER)]
    if not stage_order:
        stage_order = DEFAULT_STAGE_ORDER

    layout = prepare_output_layout(out_dir)
    data, source_summary = load_datasets(config["datasets"], args.max_rows_per_dataset)
    if not topology_order:
        topology_order = list(dict.fromkeys(data["topology"].tolist()))
    data = data[data["topology"].isin(topology_order) & data["stage"].isin(stage_order)].copy()
    summary, shift = build_summary_tables(data, topology_order, stage_order)
    length_summary, length_shift = build_length_bin_tables(data, topology_order, stage_order)

    per_sequence_columns = [
        "topology",
        "topology_label",
        "stage",
        "stage_label",
        "mol_id",
        "source_index",
        "parse_status",
        "peptide_sequence",
        "peptide_length",
        "residue_token_count",
        "peptide_length_numeric",
        "n_methyl_residue_count",
        "n_methyl_residue_fraction",
        "n_methyl_residue_tokens",
        "source_path",
    ]
    available_per_sequence_columns = [column for column in per_sequence_columns if column in data.columns]
    per_sequence_path = layout["tables"] / "per_sequence_methylation.csv"
    summary_path = layout["tables"] / "methylation_summary.csv"
    shift_path = layout["tables"] / "methylation_shift_summary.csv"
    source_summary_path = layout["tables"] / "source_summary.csv"
    length_summary_path = layout["tables"] / "methylation_by_length_summary.csv"
    length_shift_path = layout["tables"] / "methylation_by_length_shift_summary.csv"
    data[available_per_sequence_columns].to_csv(per_sequence_path, index=False)
    summary.to_csv(summary_path, index=False)
    shift.to_csv(shift_path, index=False)
    source_summary.to_csv(source_summary_path, index=False)
    length_summary.to_csv(length_summary_path, index=False)
    length_shift.to_csv(length_shift_path, index=False)

    output_files = [
        str(per_sequence_path),
        str(summary_path),
        str(shift_path),
        str(source_summary_path),
        str(length_summary_path),
        str(length_shift_path),
    ]
    output_files.extend(
        plot_methylation_shift(
            data,
            summary,
            topology_order=topology_order,
            stage_order=stage_order,
            out_prefix=layout["figures"] / "methylation_fraction_shift",
            formats=formats,
            dpi=dpi,
            scatter_max_points=scatter_max_points,
        )
    )
    if plot_length_bins:
        output_files.extend(
            plot_methylation_by_length_shift(
                length_summary,
                length_shift,
                topology_order=topology_order,
                stage_order=stage_order,
                out_prefix=layout["figures"] / "methylation_by_length_shift",
                formats=formats,
                dpi=dpi,
                min_n_per_stage=length_bin_min_n_per_stage,
            )
        )
    write_run_record(
        layout["root"] / "run_record.md",
        config_path=args.config,
        command=shlex.join(sys.argv),
        config=config,
        output_files=output_files,
        source_summary=source_summary,
        summary=summary,
        shift=shift,
        length_summary=length_summary,
        length_shift=length_shift,
    )
    print(f"Wrote {len(output_files)} outputs to {layout['root']}")


if __name__ == "__main__":
    main()
