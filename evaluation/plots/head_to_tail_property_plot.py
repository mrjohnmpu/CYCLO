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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

MM_PER_INCH = 25.4
NATURE_DOUBLE_COLUMN_WIDTH = 180.0 / MM_PER_INCH
PUBLICATION_FONT_NAME = "Arial"
PUBLICATION_FONT_PATH = Path("/usr/share/fonts/arial.ttf")
DEFAULT_TOPOLOGY = "head_to_tail"
DEFAULT_OUTPUT_DIR = Path('results/figures/head_to_tail_property_distribution_20260610')
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "configs" / "head_to_tail_property_sources.json"
DEFAULT_PROPERTIES = [
    "MW",
    "TPSA",
    "LogP",
    "HBA",
    "HBD",
    "RotBonds",
    "PAMPA_value",
    "hemolysis_score",
    "half_life_value",
]
DEFAULT_JOINT_PAIRS = ["PAMPA_value:TPSA", "PAMPA_value:LogP"]
REFERENCE_DATASET_ID = "reference"
REFERENCE_LINE_COLOR = "#737373"
REFERENCE_LINE_STYLE = (0, (2.7, 2.0))
REFERENCE_LINE_WIDTH = 0.9
REFERENCE_IQR_COLOR = "#D9D9D9"
REFERENCE_IQR_ALPHA = 0.45
DEFAULT_METHOD_ORDER = [
    "reference",
    "aae",
    "vae",
    "charrnn",
    "pepinvent_mask_30",
    "pepinvent_mask_50",
    "pepinvent_mask_70",
    "ours",
]
PROPERTY_METADATA = {
    "MW": {"label": "MW", "stem": "mw", "kind": "continuous"},
    "TPSA": {"label": "TPSA", "stem": "tpsa", "kind": "continuous"},
    "PAMPA_value": {"label": "PAMPA", "stem": "pampa", "kind": "continuous"},
    "half_life_value": {"label": "Half-life", "stem": "half_life", "kind": "continuous"},
    "LogP": {"label": "LogP", "stem": "logp", "kind": "continuous"},
    "HBA": {"label": "HBA", "stem": "hba", "kind": "discrete"},
    "HBD": {"label": "HBD", "stem": "hbd", "kind": "discrete"},
    "RotBonds": {"label": "Rotatable bonds", "stem": "rotbonds", "kind": "discrete"},
    "hemolysis_score": {"label": "Hemolysis", "stem": "hemolysis", "kind": "continuous"},
}
NUMERIC_COLUMNS = sorted(PROPERTY_METADATA)
PALETTE = {
    "reference": "#B8B8B8",
    "aae": "#4C78A8",
    "vae": "#72B7B2",
    "charrnn": "#59A14F",
    "pepinvent_mask_30": "#F2CF5B",
    "pepinvent_mask_50": "#F28E2B",
    "pepinvent_mask_70": "#C17A36",
    "ours": "#6F4BB2",
}
PUBLICATION_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": [PUBLICATION_FONT_NAME, "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 6,
    "axes.titlesize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.unicode_minus": False,
}
OUTPUT_SUFFIXES = {
    "pdf": ".pdf",
    "svg": ".svg",
    "png": ".png",
    "tiff": ".tiff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Nature-style property distributions for head-to-tail peptide datasets.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    parser.add_argument(
        "--reference-style",
        choices=["distribution", "median-line", "median-iqr"],
        default="distribution",
        help="Draw Reference as a full distribution, a median line, or a median line with an IQR band. "
        "Joint plots are only generated in distribution mode.",
    )
    parser.add_argument("--properties", nargs="+", default=DEFAULT_PROPERTIES)
    parser.add_argument("--joint-pairs", nargs="+", default=DEFAULT_JOINT_PAIRS)
    parser.add_argument("--joint-background", default="reference")
    parser.add_argument("--joint-baseline", default="pepinvent_mask_70")
    parser.add_argument("--joint-highlight", default="ours")
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--joint-sample-size", type=int, default=1200)
    parser.add_argument("--scatter-sample-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--formats", nargs="+", default=list(OUTPUT_SUFFIXES), choices=OUTPUT_SUFFIXES)
    return parser.parse_args()


def apply_publication_style() -> None:
    if PUBLICATION_FONT_PATH.is_file():
        font_manager.fontManager.addfont(str(PUBLICATION_FONT_PATH))
        registered_name = font_manager.FontProperties(fname=str(PUBLICATION_FONT_PATH)).get_name()
        PUBLICATION_RCPARAMS["font.sans-serif"] = [
            registered_name,
            "Helvetica",
            "Nimbus Sans",
            "Liberation Sans",
            "DejaVu Sans",
        ]
    plt.rcParams.update(PUBLICATION_RCPARAMS)
    try:
        font_manager.findfont(font_manager.FontProperties(family=PUBLICATION_FONT_NAME), fallback_to_default=False)
    except ValueError:
        warnings.warn(f"{PUBLICATION_FONT_NAME} is not available; using the closest sans-serif fallback.", RuntimeWarning)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", payload if isinstance(payload, list) else None)
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Manifest must contain a non-empty 'datasets' list.")
    seen: set[str] = set()
    for entry in datasets:
        dataset_id = entry.get("id")
        if not dataset_id or dataset_id in seen:
            raise ValueError(f"Invalid or duplicate dataset id: {dataset_id!r}")
        seen.add(dataset_id)
        if not entry.get("label") or not entry.get("path"):
            raise ValueError(f"Dataset entry {dataset_id!r} must include label and path.")
        source = Path(entry["path"]).expanduser()
        if not source.is_absolute():
            entry["path"] = str((Path(__file__).resolve().parents[2] / source).resolve())
    return datasets


def ensure_output_layout(out_dir: Path) -> dict[str, Path]:
    layout = {
        "root": out_dir,
        "overview": out_dir / "figures" / "overview",
        "single_property": out_dir / "figures" / "single_property",
        "joint": out_dir / "figures" / "joint",
        "tables": out_dir / "tables",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def filter_topology(frame: pd.DataFrame, *, topology: str, dataset_id: str, source_path: Path) -> tuple[pd.DataFrame, str]:
    if "Cyclization" in frame.columns:
        filtered = frame[frame["Cyclization"].astype(str).str.lower() == topology.lower()].copy()
        mode = f"Cyclization == {topology}"
        if dataset_id == "reference" and filtered.empty:
            raise ValueError(f"Reference dataset {source_path} has no rows for topology {topology!r}.")
        if not filtered.empty:
            return filtered.reset_index(drop=True), mode
    if dataset_id == "reference":
        raise ValueError(f"Reference dataset {source_path} is missing usable Cyclization rows for {topology!r}.")
    return frame.copy().reset_index(drop=True), "no topology column or no matching rows; kept all rows for generated dataset"


def coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            raise ValueError(f"Missing required property column: {column}")
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def sample_frame(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed).copy()


def load_datasets(
    datasets: list[dict[str, Any]],
    *,
    topology: str,
    properties: list[str],
    max_rows: int | None,
    seed: int,
    reference_style: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str], dict[str, str], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    summary: list[dict[str, Any]] = []
    order: list[str] = []
    label_map: dict[str, str] = {}
    reference_statistics: list[dict[str, Any]] = []
    for index, entry in enumerate(datasets):
        dataset_id = entry["id"]
        label = entry["label"]
        path = Path(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing CSV for {dataset_id}: {path}")
        raw = pd.read_csv(path)
        raw_rows = len(raw)
        filtered, filter_mode = filter_topology(raw, topology=topology, dataset_id=dataset_id, source_path=path)
        topology_rows = len(filtered)
        filtered = coerce_numeric(filtered, properties)
        valid_mask = filtered[properties].notna().any(axis=1)
        filtered = filtered[valid_mask].copy()
        if dataset_id == REFERENCE_DATASET_ID:
            for property_name in properties:
                values = filtered[property_name].dropna().to_numpy(dtype=float)
                q1, median, q3 = np.nanpercentile(values, [25, 50, 75]) if values.size else (np.nan, np.nan, np.nan)
                reference_statistics.append({
                    "property": property_name,
                    "n": int(values.size),
                    "q1": float(q1),
                    "median": float(median),
                    "q3": float(q3),
                })
        if dataset_id == REFERENCE_DATASET_ID and reference_style != "distribution":
            plot_frame = filtered.iloc[0:0].copy()
        else:
            plot_frame = sample_frame(filtered, max_rows, seed + index)
        plot_frame["dataset_id"] = dataset_id
        plot_frame["dataset_label"] = label
        frames.append(plot_frame)
        order.append(dataset_id)
        label_map[dataset_id] = label
        summary.append({
            "dataset_id": dataset_id,
            "dataset_label": label,
            "path": str(path),
            "filter_mode": filter_mode,
            "raw_rows": raw_rows,
            "rows_after_topology_filter": topology_rows,
            "rows_after_numeric_filter": int(valid_mask.sum()),
            "plotted_rows": len(plot_frame),
        })
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return merged, summary, order, label_map, reference_statistics


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str], *, tight: bool = True) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    outputs: list[str] = []
    for fmt in formats:
        path = out_dir / f"{stem}{OUTPUT_SUFFIXES[fmt]}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"} if tight else {}
        if fmt in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def dataset_color(dataset_id: str) -> str:
    return PALETTE.get(dataset_id, "#4C78A8")


def method_values(data: pd.DataFrame, dataset_id: str, property_name: str) -> np.ndarray:
    values = pd.to_numeric(data.loc[data["dataset_id"] == dataset_id, property_name], errors="coerce").dropna()
    return values.to_numpy(dtype=float)


def reference_line_handle() -> Line2D:
    return Line2D(
        [0],
        [0],
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_STYLE,
        linewidth=REFERENCE_LINE_WIDTH,
        label="Reference median",
    )


def reference_legend_handles(reference_style: str) -> list[Patch | Line2D]:
    handles: list[Patch | Line2D] = []
    if reference_style == "median-iqr":
        handles.append(Patch(
            facecolor=REFERENCE_IQR_COLOR,
            edgecolor="none",
            alpha=REFERENCE_IQR_ALPHA,
            label="Reference IQR",
        ))
    handles.append(reference_line_handle())
    return handles


def reference_style_suffix(reference_style: str) -> str:
    if reference_style == "median-line":
        return "_reference_median"
    if reference_style == "median-iqr":
        return "_reference_median_iqr"
    return ""


def draw_half_violin_property(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    property_name: str,
    dataset_order: list[str],
    label_map: dict[str, str],
    reference_style: str,
    reference_statistics: dict[str, dict[str, Any]],
    show_xlabel: bool = True,
) -> None:
    meta = PROPERTY_METADATA[property_name]
    series: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    active_ids: list[str] = []
    plot_order = [
        dataset_id
        for dataset_id in dataset_order
        if reference_style == "distribution" or dataset_id != REFERENCE_DATASET_ID
    ]
    for dataset_id in plot_order:
        values = method_values(data, dataset_id, property_name)
        if values.size == 0:
            continue
        series.append(values)
        labels.append(label_map[dataset_id])
        colors.append(dataset_color(dataset_id))
        active_ids.append(dataset_id)
    if not series:
        raise ValueError(f"No plottable values for property {property_name!r}.")

    positions = np.arange(len(series), dtype=float)
    for idx, values in enumerate(series):
        violin = ax.violinplot(values, positions=[positions[idx]], widths=0.82, showmeans=False, showmedians=False, showextrema=False)
        for body in violin["bodies"]:
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.clip(vertices[:, 0], positions[idx], np.inf)
            body.set_facecolor(colors[idx])
            body.set_edgecolor("black")
            body.set_linewidth(0.45)
            body.set_alpha(0.58 if active_ids[idx] != "reference" else 0.36)

    box = ax.boxplot(
        series,
        positions=positions - 0.10,
        widths=0.16,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.75},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.55},
        whiskerprops={"color": "black", "linewidth": 0.45},
        capprops={"color": "black", "linewidth": 0.45},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.88)

    rng = np.random.default_rng(42)
    for idx, values in enumerate(series):
        sample_n = min(values.size, 180)
        sampled = values if values.size <= sample_n else rng.choice(values, size=sample_n, replace=False)
        x = rng.normal(loc=positions[idx] - 0.23, scale=0.025, size=sampled.size)
        ax.scatter(x, sampled, s=1.8, alpha=0.16, color="black", linewidths=0, rasterized=True)

    if reference_style != "distribution":
        property_statistics = reference_statistics.get(property_name, {})
        reference_median = property_statistics.get("median", np.nan)
        if not np.isfinite(reference_median):
            raise ValueError(f"Reference has no valid values for property {property_name!r}.")
        if reference_style == "median-iqr":
            reference_q1 = property_statistics.get("q1", np.nan)
            reference_q3 = property_statistics.get("q3", np.nan)
            if not np.isfinite(reference_q1) or not np.isfinite(reference_q3) or reference_q1 > reference_q3:
                raise ValueError(f"Reference has invalid IQR values for property {property_name!r}.")
            ax.axhspan(
                reference_q1,
                reference_q3,
                facecolor=REFERENCE_IQR_COLOR,
                edgecolor="none",
                alpha=REFERENCE_IQR_ALPHA,
                zorder=0,
            )
        ax.hlines(
            reference_median,
            positions[0] - 0.41,
            positions[-1] + 0.41,
            colors=REFERENCE_LINE_COLOR,
            linestyles=REFERENCE_LINE_STYLE,
            linewidth=REFERENCE_LINE_WIDTH,
            zorder=1,
        )
        ax.set_xlim(positions[0] - 0.45, positions[-1] + 0.45)

    ax.set_ylabel(meta["label"])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    if not show_xlabel:
        ax.set_xticklabels([])
    ax.tick_params(axis="both", length=2.5, width=0.6)
    ax.grid(False)


def plot_single_properties(
    data: pd.DataFrame,
    *,
    properties: list[str],
    dataset_order: list[str],
    label_map: dict[str, str],
    reference_style: str,
    reference_statistics: dict[str, dict[str, Any]],
    out_dir: Path,
    formats: list[str],
) -> list[str]:
    outputs: list[str] = []
    stem_suffix = reference_style_suffix(reference_style)
    for property_name in properties:
        fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_COLUMN_WIDTH * 0.72, 62.0 / MM_PER_INCH))
        draw_half_violin_property(
            ax,
            data,
            property_name=property_name,
            dataset_order=dataset_order,
            label_map=label_map,
            reference_style=reference_style,
            reference_statistics=reference_statistics,
        )
        ax.set_title(f"{PROPERTY_METADATA[property_name]['label']} distribution")
        if reference_style != "distribution":
            ax.legend(
                handles=reference_legend_handles(reference_style),
                loc="upper right",
                frameon=False,
                handlelength=1.8,
                borderpad=0.2,
                handletextpad=0.5,
            )
        stem = f"single_{PROPERTY_METADATA[property_name]['stem']}{stem_suffix}"
        outputs.extend(save_figure(fig, out_dir, stem, formats))
    return outputs


def plot_overview(
    data: pd.DataFrame,
    *,
    properties: list[str],
    dataset_order: list[str],
    label_map: dict[str, str],
    reference_style: str,
    reference_statistics: dict[str, dict[str, Any]],
    out_dir: Path,
    formats: list[str],
) -> list[str]:
    ncols = 3
    nrows = math.ceil(len(properties) / ncols)
    fig_w = NATURE_DOUBLE_COLUMN_WIDTH
    fig_h = min(165.0 / MM_PER_INCH, max(2.4, nrows * 1.62))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
    axes_array = np.array(axes).reshape(-1)
    for idx, property_name in enumerate(properties):
        ax = axes_array[idx]
        draw_half_violin_property(
            ax,
            data,
            property_name=property_name,
            dataset_order=dataset_order,
            label_map=label_map,
            reference_style=reference_style,
            reference_statistics=reference_statistics,
            show_xlabel=idx >= (nrows - 1) * ncols,
        )
        ax.text(-0.13, 1.03, chr(ord("a") + idx), transform=ax.transAxes, fontsize=7, fontweight="bold", va="bottom")
    for ax in axes_array[len(properties):]:
        ax.axis("off")
    stem = f"head_to_tail_property_distribution_overview{reference_style_suffix(reference_style)}"
    if reference_style != "distribution":
        legend_handles = reference_legend_handles(reference_style)
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(legend_handles),
            frameon=False,
            handlelength=1.8,
            handletextpad=0.5,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return save_figure(fig, out_dir, stem, formats, tight=False)
    return save_figure(fig, out_dir, stem, formats)


def parse_joint_pair(pair: str) -> tuple[str, str]:
    if ":" not in pair:
        raise ValueError(f"Invalid joint pair {pair!r}; expected format x:y")
    x_col, y_col = [item.strip() for item in pair.split(":", 1)]
    if not x_col or not y_col:
        raise ValueError(f"Invalid joint pair {pair!r}; expected non-empty x:y")
    return x_col, y_col


def sample_joint(data: pd.DataFrame, dataset_id: str, columns: tuple[str, str], n: int, seed: int) -> pd.DataFrame:
    subset = data[data["dataset_id"] == dataset_id].dropna(subset=list(columns)).copy()
    if subset.empty:
        raise ValueError(f"Dataset {dataset_id!r} has no rows for joint columns {columns}.")
    if n and len(subset) > n:
        subset = subset.sample(n=n, random_state=seed)
    return subset


def plot_joint_distributions(data: pd.DataFrame, *, joint_pairs: list[str], label_map: dict[str, str], background_id: str, baseline_id: str, highlight_id: str, out_dir: Path, formats: list[str], sample_size: int, seed: int) -> list[str]:
    outputs: list[str] = []
    for idx, pair in enumerate(joint_pairs):
        x_col, y_col = parse_joint_pair(pair)
        for col in (x_col, y_col):
            if col not in PROPERTY_METADATA:
                raise ValueError(f"Unsupported joint property {col!r}. Supported: {sorted(PROPERTY_METADATA)}")
        bg = sample_joint(data, background_id, (x_col, y_col), sample_size, seed + idx * 10)
        base = sample_joint(data, baseline_id, (x_col, y_col), sample_size, seed + idx * 10 + 1)
        high = sample_joint(data, highlight_id, (x_col, y_col), sample_size, seed + idx * 10 + 2)
        fig, ax = plt.subplots(figsize=(86.0 / MM_PER_INCH, 72.0 / MM_PER_INCH))
        ax.scatter(bg[x_col], bg[y_col], s=5, alpha=0.13, color="#BDBDBD", linewidths=0, label=f"{label_map[background_id]} background", rasterized=True)
        ax.scatter(base[x_col], base[y_col], s=7, alpha=0.30, color=dataset_color(baseline_id), linewidths=0, label=label_map[baseline_id], rasterized=True)
        ax.scatter(high[x_col], high[y_col], s=8, alpha=0.42, color=dataset_color(highlight_id), linewidths=0, label=label_map[highlight_id], rasterized=True)
        ax.set_xlabel(PROPERTY_METADATA[x_col]["label"])
        ax.set_ylabel(PROPERTY_METADATA[y_col]["label"])
        ax.legend(frameon=False, loc="best", handletextpad=0.3, borderpad=0.2)
        ax.tick_params(axis="both", length=2.5, width=0.6)
        ax.grid(False)
        stem = f"joint_{PROPERTY_METADATA[x_col]['stem']}_vs_{PROPERTY_METADATA[y_col]['stem']}"
        outputs.extend(save_figure(fig, out_dir, stem, formats))
    return outputs


def write_run_record(
    path: Path,
    *,
    command: str,
    manifest_path: Path,
    output_dir: Path,
    properties: list[str],
    joint_pairs: list[str],
    outputs: list[str],
    summary: list[dict[str, Any]],
    reference_statistics: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Head-to-tail Property Distribution Plot Record",
        "",
        f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Command: `{command}`",
        f"- Manifest: `{manifest_path}`",
        f"- Output directory: `{output_dir}`",
        f"- Topology: `{args.topology}`",
        f"- Reference style: `{args.reference_style}`",
        f"- Properties: {', '.join(f'`{p}`' for p in properties)}",
        f"- Max rows per dataset: `{args.max_rows_per_dataset}`",
    ]
    if args.reference_style == "distribution":
        lines.extend([
            f"- Joint pairs: {', '.join(f'`{p}`' for p in joint_pairs)}",
            f"- Joint background: `{args.joint_background}`",
            f"- Joint baseline: `{args.joint_baseline}`",
            f"- Joint highlight: `{args.joint_highlight}`",
        ])
    else:
        statistic_description = "median" if args.reference_style == "median-line" else "median and IQR (Q1-Q3)"
        lines.extend([
            f"- Reference statistic: `{statistic_description}`",
            f"- Joint plots: `not generated in {args.reference_style} mode`",
        ])
    lines.extend([
        "",
        "## Inputs",
        "",
        "| dataset_id | label | raw_rows | rows_after_topology_filter | rows_after_numeric_filter | plotted_rows | filter_mode | path |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for item in summary:
        lines.append(
            "| {dataset_id} | {dataset_label} | {raw_rows} | {rows_after_topology_filter} | "
            "{rows_after_numeric_filter} | {plotted_rows} | {filter_mode} | `{path}` |".format(**item)
        )
    if args.reference_style != "distribution":
        include_iqr = args.reference_style == "median-iqr"
        lines.extend([
            "",
            "## Reference statistics",
            "",
            "| property | n | q1 | median | q3 |" if include_iqr else "| property | n | median |",
            "|---|---:|---:|---:|---:|" if include_iqr else "|---|---:|---:|",
        ])
        for item in reference_statistics:
            if include_iqr:
                lines.append(
                    f"| {item['property']} | {item['n']} | {item['q1']:.8g} | "
                    f"{item['median']:.8g} | {item['q3']:.8g} |"
                )
            else:
                lines.append(f"| {item['property']} | {item['n']} | {item['median']:.8g} |")
    lines.extend(["", "## Outputs", ""])
    for output in outputs:
        lines.append(f"- `{output}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_command() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def main() -> None:
    args = parse_args()
    apply_publication_style()
    properties = list(args.properties)
    unsupported = sorted(set(properties) - set(PROPERTY_METADATA))
    if unsupported:
        raise ValueError(f"Unsupported properties: {unsupported}. Supported: {sorted(PROPERTY_METADATA)}")
    if args.reference_style == "distribution":
        for pair in args.joint_pairs:
            parse_joint_pair(pair)
    layout = ensure_output_layout(args.out_dir)
    datasets = load_manifest(args.manifest)
    merged, summary, dataset_order, label_map, reference_statistics = load_datasets(
        datasets,
        topology=args.topology,
        properties=properties,
        max_rows=args.max_rows_per_dataset,
        seed=args.seed,
        reference_style=args.reference_style,
    )
    reference_statistics_by_property = {item["property"]: item for item in reference_statistics}
    if args.reference_style != "distribution":
        if REFERENCE_DATASET_ID not in label_map:
            raise ValueError(f"Manifest must contain a 'reference' dataset for {args.reference_style} mode.")
        missing_reference_properties = [
            property_name
            for property_name in properties
            if property_name not in reference_statistics_by_property
            or not np.isfinite(reference_statistics_by_property[property_name].get("median", np.nan))
        ]
        if missing_reference_properties:
            raise ValueError(f"Reference has no valid values for properties: {missing_reference_properties}")
        if args.reference_style == "median-iqr":
            invalid_iqr_properties = [
                property_name
                for property_name in properties
                if not np.isfinite(reference_statistics_by_property[property_name].get("q1", np.nan))
                or not np.isfinite(reference_statistics_by_property[property_name].get("q3", np.nan))
                or reference_statistics_by_property[property_name]["q1"]
                > reference_statistics_by_property[property_name]["q3"]
            ]
            if invalid_iqr_properties:
                raise ValueError(f"Reference has invalid IQR values for properties: {invalid_iqr_properties}")
    else:
        required_joint_ids = [args.joint_background, args.joint_baseline, args.joint_highlight]
        missing_joint_ids = [dataset_id for dataset_id in required_joint_ids if dataset_id not in label_map]
        if missing_joint_ids:
            raise ValueError(f"Joint dataset ids not found in manifest: {missing_joint_ids}")

    outputs: list[str] = []
    outputs.extend(plot_single_properties(
        merged,
        properties=properties,
        dataset_order=dataset_order,
        label_map=label_map,
        reference_style=args.reference_style,
        reference_statistics=reference_statistics_by_property,
        out_dir=layout["single_property"],
        formats=args.formats,
    ))
    outputs.extend(plot_overview(
        merged,
        properties=properties,
        dataset_order=dataset_order,
        label_map=label_map,
        reference_style=args.reference_style,
        reference_statistics=reference_statistics_by_property,
        out_dir=layout["overview"],
        formats=args.formats,
    ))
    if args.reference_style == "distribution":
        outputs.extend(plot_joint_distributions(
            merged,
            joint_pairs=args.joint_pairs,
            label_map=label_map,
            background_id=args.joint_background,
            baseline_id=args.joint_baseline,
            highlight_id=args.joint_highlight,
            out_dir=layout["joint"],
            formats=args.formats,
            sample_size=args.joint_sample_size,
            seed=args.seed,
        ))

    record_suffix = reference_style_suffix(args.reference_style)
    summary_path = layout["tables"] / f"source_summary{record_suffix}.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    outputs.append(str(summary_path))
    if args.reference_style != "distribution":
        if args.reference_style == "median-iqr":
            reference_path = layout["tables"] / "reference_iqr.csv"
            reference_columns = ["property", "n", "q1", "median", "q3"]
        else:
            reference_path = layout["tables"] / "reference_medians.csv"
            reference_columns = ["property", "n", "median"]
        pd.DataFrame(reference_statistics)[reference_columns].to_csv(reference_path, index=False)
        outputs.append(str(reference_path))
    run_record_path = layout["root"] / f"run_record{record_suffix}.md"
    write_run_record(
        run_record_path,
        command=build_command(),
        manifest_path=args.manifest,
        output_dir=args.out_dir,
        properties=properties,
        joint_pairs=args.joint_pairs,
        outputs=outputs,
        summary=summary,
        reference_statistics=reference_statistics,
        args=args,
    )
    print(json.dumps({"output_dir": str(args.out_dir), "outputs": outputs, "run_record": str(run_record_path)}, indent=2))


if __name__ == "__main__":
    main()
