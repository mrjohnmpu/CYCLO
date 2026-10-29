#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys
import warnings
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

MM_PER_INCH = 25.4
PUBLICATION_FONT_NAME = "Arial"
PUBLICATION_FONT_PATH = Path("/usr/share/fonts/arial.ttf")
FIGURE_WIDTH = 172.0 / MM_PER_INCH
MULTI_TOPOLOGY_PANEL_HEIGHT = 18.6 / MM_PER_INCH
SINGLE_TOPOLOGY_PANEL_HEIGHT = 18.0 / MM_PER_INCH
DEFAULT_TOPOLOGY = "sc_to_tail"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "configs" / "sc_to_tail_property_sources.json"
DEFAULT_DISULFIDE_MANIFEST = Path(__file__).resolve().parent / "configs" / "disulfide_bridge_property_sources.json"
DEFAULT_OUTPUT_DIR = Path('results/figures/sc_disulfide_property_distribution_20260610')
DEFAULT_TOPOLOGY_CONFIGS = [
    f"sc_to_tail:{DEFAULT_MANIFEST}",
    f"disulfide_bridge:{DEFAULT_DISULFIDE_MANIFEST}",
]
DEFAULT_PROPERTIES = [
    "MW",
    "HBD",
    "TPSA",
    "LogP",
    "PAMPA_value",
    "hemolysis_score",
]
PROPERTY_METADATA = {
    "MW": {"label": "MW", "stem": "mw"},
    "TPSA": {"label": "TPSA", "stem": "tpsa"},
    "LogP": {"label": "LogP", "stem": "logp"},
    "HBA": {"label": "HBA", "stem": "hba"},
    "HBD": {"label": "HBD", "stem": "hbd"},
    "RotBonds": {"label": "Rotatable bonds", "stem": "rotbonds"},
    "PAMPA_value": {"label": "PAMPA", "stem": "pampa"},
    "hemolysis_score": {"label": "Hemolysis", "stem": "hemolysis"},
    "half_life_value": {"label": "Half-life", "stem": "half_life"},
}
TOPOLOGY_LABELS = {
    "head_to_tail": "head-to-tail",
    "sc_to_tail": "sc-to-tail",
    "disulfide_bridge": "disulfide",
}
PUBLICATION_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": [PUBLICATION_FONT_NAME, "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "font.size": 6.0,
    "axes.labelsize": 6.0,
    "axes.titlesize": 6.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "axes.unicode_minus": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw sc-to-tail/disulfide property distributions with the stacked violin template."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY)
    parser.add_argument(
        "--topology-config",
        nargs="+",
        default=DEFAULT_TOPOLOGY_CONFIGS,
        help="Topology and manifest pairs as topology:/path/to/manifest.json. "
        "The default draws sc_to_tail and disulfide_bridge in one figure.",
    )
    parser.add_argument(
        "--single-topology",
        action="store_true",
        help="Use --manifest and --topology to draw one topology only.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--properties", nargs="+", default=DEFAULT_PROPERTIES)
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png", "tiff"], choices=["pdf", "svg", "png", "tiff"])
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-label", default="")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def apply_publication_style() -> None:
    if PUBLICATION_FONT_PATH.is_file():
        font_manager.fontManager.addfont(str(PUBLICATION_FONT_PATH))
    plt.rcParams.update(PUBLICATION_RCPARAMS)
    try:
        font_manager.findfont(font_manager.FontProperties(family=PUBLICATION_FONT_NAME), fallback_to_default=False)
    except ValueError:
        warnings.warn(f"{PUBLICATION_FONT_NAME} is not available; using the closest sans-serif fallback.", RuntimeWarning)


def topology_label(topology: str) -> str:
    return TOPOLOGY_LABELS.get(topology, topology.replace("_", "-"))


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
    return datasets


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


def coerce_numeric(frame: pd.DataFrame, properties: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for property_name in properties:
        if property_name not in result.columns:
            raise ValueError(f"Missing required property column: {property_name}")
        result[property_name] = pd.to_numeric(result[property_name], errors="coerce")
    return result


def sample_frame(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed).copy()


def to_long(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_label: str,
    topology: str,
    properties: list[str],
) -> pd.DataFrame:
    long = frame.melt(
        value_vars=properties,
        var_name="property",
        value_name="value",
    ).dropna(subset=["value"])
    long["dataset_id"] = dataset_id
    long["model"] = dataset_label
    long["topology"] = topology_label(topology)
    return long


def load_property_data(
    datasets: list[dict[str, Any]],
    *,
    topology: str,
    properties: list[str],
    max_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str], list[dict[str, Any]]]:
    generated_frames: list[pd.DataFrame] = []
    reference_frames: list[pd.DataFrame] = []
    model_order: list[str] = []
    label_map: dict[str, str] = {}
    summary: list[dict[str, Any]] = []

    for index, entry in enumerate(datasets):
        dataset_id = entry["id"]
        label = entry["label"]
        source_path = Path(entry["path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing CSV for {dataset_id}: {source_path}")

        raw = pd.read_csv(source_path)
        filtered, filter_mode = filter_topology(raw, topology=topology, dataset_id=dataset_id, source_path=source_path)
        filtered = coerce_numeric(filtered, properties)
        valid_mask = filtered[properties].notna().any(axis=1)
        numeric_filtered = filtered[valid_mask].copy()
        sampled = sample_frame(numeric_filtered, max_rows, seed + index)
        long = to_long(sampled, dataset_id=dataset_id, dataset_label=label, topology=topology, properties=properties)

        if dataset_id == "reference" or entry.get("role") == "reference":
            reference_frames.append(long)
        else:
            generated_frames.append(long)
            model_order.append(dataset_id)
            label_map[dataset_id] = label

        summary.append({
            "dataset_id": dataset_id,
            "dataset_label": label,
            "path": str(source_path),
            "filter_mode": filter_mode,
            "raw_rows": len(raw),
            "rows_after_topology_filter": len(filtered),
            "rows_after_numeric_filter": int(valid_mask.sum()),
            "plotted_rows": len(sampled),
        })

    generated = pd.concat(generated_frames, ignore_index=True) if generated_frames else pd.DataFrame()
    reference = pd.concat(reference_frames, ignore_index=True) if reference_frames else pd.DataFrame()
    if generated.empty:
        raise ValueError("No generated datasets were loaded.")
    if reference.empty:
        raise ValueError("No reference dataset was loaded.")
    return generated, reference, model_order, label_map, summary


def parse_topology_configs(configs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in configs:
        if ":" not in item:
            raise ValueError(f"Invalid --topology-config {item!r}; expected topology:/path/to/manifest.json")
        topology, manifest = item.split(":", 1)
        topology = topology.strip()
        manifest_path = Path(manifest.strip())
        if not topology or not manifest:
            raise ValueError(f"Invalid --topology-config {item!r}; topology and manifest must be non-empty.")
        if topology in seen:
            raise ValueError(f"Duplicate topology in --topology-config: {topology}")
        seen.add(topology)
        parsed.append((topology, manifest_path))
    if not parsed:
        raise ValueError("At least one topology config is required.")
    return parsed


def load_multi_topology_data(
    configs: list[tuple[str, Path]],
    *,
    properties: list[str],
    max_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str], list[dict[str, Any]]]:
    generated_frames: list[pd.DataFrame] = []
    reference_frames: list[pd.DataFrame] = []
    summary: list[dict[str, Any]] = []
    model_order: list[str] = []
    label_map: dict[str, str] = {}

    for topology_index, (topology, manifest_path) in enumerate(configs):
        datasets = load_manifest(manifest_path)
        generated, reference, topology_model_order, topology_label_map, topology_summary = load_property_data(
            datasets,
            topology=topology,
            properties=properties,
            max_rows=max_rows,
            seed=seed + topology_index * 1000,
        )
        generated_frames.append(generated)
        reference_frames.append(reference)
        for dataset_id in topology_model_order:
            if dataset_id not in model_order:
                model_order.append(dataset_id)
            previous = label_map.get(dataset_id)
            current = topology_label_map[dataset_id]
            if previous is not None and previous != current:
                raise ValueError(f"Dataset label mismatch for {dataset_id!r}: {previous!r} vs {current!r}")
            label_map[dataset_id] = current
        for item in topology_summary:
            item = dict(item)
            item["topology"] = topology
            item["manifest"] = str(manifest_path)
            summary.append(item)

    return (
        pd.concat(generated_frames, ignore_index=True),
        pd.concat(reference_frames, ignore_index=True),
        model_order,
        label_map,
        summary,
    )


def prepare_output_dirs(out_dir: Path) -> dict[str, Path]:
    layout = {
        "root": out_dir,
        "figures": out_dir / "figures",
        "tables": out_dir / "tables",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> list[str]:
    outputs: list[str] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if fmt in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def panel_letter(index: int) -> str:
    return chr(ord("a") + index)


def property_values(frame: pd.DataFrame, *, property_name: str, dataset_id: str | None = None) -> np.ndarray:
    subset = frame[frame["property"] == property_name]
    if dataset_id is not None:
        subset = subset[subset["dataset_id"] == dataset_id]
    return pd.to_numeric(subset["value"], errors="coerce").dropna().to_numpy(dtype=float)


def plot_property_violin_template(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    *,
    properties: list[str],
    model_order: list[str],
    label_map: dict[str, str],
    topology: str,
    output_prefix: Path,
    formats: list[str],
    panel_label: str,
) -> list[str]:
    baseline_color = (90 / 255, 113 / 255, 146 / 255)
    our_color = (197 / 255, 94 / 255, 96 / 255)
    reference_color = (0.45, 0.45, 0.45)
    topo_label = topology_label(topology)

    labels = [label_map[dataset_id] for dataset_id in model_order]
    positions = np.arange(len(model_order), dtype=float)
    x0 = positions[0] - 0.34
    x1 = positions[-1] + 0.34

    height = max(88.0 / MM_PER_INCH, SINGLE_TOPOLOGY_PANEL_HEIGHT * len(properties))
    fig, axes = plt.subplots(
        len(properties),
        1,
        figsize=(FIGURE_WIDTH, height),
        sharex=True,
    )
    axes_array = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for property_index, (ax, property_name) in enumerate(zip(axes_array, properties)):
        ax.set_facecolor("white")
        ax.text(
            0.008,
            0.94,
            panel_letter(property_index),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.0,
            fontweight="bold",
        )
        combined = np.concatenate([
            property_values(df, property_name=property_name),
            property_values(ref_df, property_name=property_name),
        ])
        if combined.size == 0:
            raise ValueError(f"No plottable values for property {property_name!r}.")
        y_low, y_high = np.nanpercentile(combined, [0.6, 99.4])
        pad = (y_high - y_low) * 0.16 if y_high > y_low else 1.0

        for xpos, dataset_id in zip(positions, model_order):
            values = property_values(df, property_name=property_name, dataset_id=dataset_id)
            if values.size == 0:
                continue
            if values.size >= 2:
                parts = ax.violinplot(
                    values,
                    positions=[xpos],
                    widths=0.68,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                fill = our_color if dataset_id == "ours" else baseline_color
                for body in parts["bodies"]:
                    body.set_facecolor(fill)
                    body.set_edgecolor(fill)
                    body.set_alpha(0.88 if dataset_id == "ours" else 0.68)
                    body.set_linewidth(0.4)

            q1, med, q3 = np.percentile(values, [25, 50, 75])
            ax.plot([xpos, xpos], [q1, q3], color="black", linewidth=0.75, zorder=4)
            ax.plot(xpos, med, marker="o", color="black", markersize=1.9, zorder=5)

        ref_values = property_values(ref_df, property_name=property_name)
        if ref_values.size == 0:
            raise ValueError(f"Reference has no plottable values for property {property_name!r}.")
        ax.hlines(
            np.nanmedian(ref_values),
            x0,
            x1,
            colors=reference_color,
            linestyles=(0, (2.7, 2.0)),
            linewidth=0.9,
            zorder=2,
        )

        ax.set_ylim(y_low - pad, y_high + pad)
        ax.set_ylabel(PROPERTY_METADATA[property_name]["label"], fontsize=6.0)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.tick_params(axis="y", labelsize=6.0, length=2.2, pad=2)
        ax.tick_params(axis="x", length=2.2, pad=1.5)

    top_ax = axes_array[0]
    top_ax.text(
        float(np.mean(positions)),
        1.10,
        topo_label,
        transform=top_ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.0,
        weight="bold",
    )

    axes_array[-1].set_xticks(positions)
    axes_array[-1].set_xticklabels(labels, rotation=65, ha="right", fontsize=6.0)
    for ax in axes_array[:-1]:
        ax.tick_params(labelbottom=False)

    if panel_label:
        fig.text(0.015, 0.985, panel_label, ha="left", va="top", fontsize=6.0, weight="bold")

    legend_elements = [
        Line2D([0], [0], color=baseline_color, lw=6, solid_capstyle="butt", label="Baselines"),
        Line2D([0], [0], color=our_color, lw=6, solid_capstyle="butt", label="TopoCycGPT"),
        Line2D([0], [0], color=reference_color, lw=1.0, linestyle=(0, (2.7, 2.0)), label="Reference median"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.987),
        ncol=3,
        frameon=False,
        fontsize=6.0,
        handlelength=1.8,
        columnspacing=1.5,
        handletextpad=0.5,
    )

    plt.subplots_adjust(left=0.09, right=0.995, top=0.91, bottom=0.15, hspace=0.18)
    return save_figure(fig, output_prefix.parent, output_prefix.name, formats)


def plot_multi_topology_violin_template(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    *,
    properties: list[str],
    model_order: list[str],
    label_map: dict[str, str],
    topologies: list[str],
    output_prefix: Path,
    formats: list[str],
    panel_label: str,
) -> list[str]:
    baseline_color = (90 / 255, 113 / 255, 146 / 255)
    our_color = (197 / 255, 94 / 255, 96 / 255)
    reference_color = (0.45, 0.45, 0.45)
    gap = 0.8

    positions: dict[tuple[str, str], float] = {}
    xticks: list[float] = []
    xticklabels: list[str] = []
    block_centers: list[tuple[str, float]] = []
    block_ranges: dict[str, tuple[float, float]] = {}
    x = 0.0

    for topology in topologies:
        block_start = x
        for dataset_id in model_order:
            positions[(topology, dataset_id)] = x
            xticks.append(x)
            xticklabels.append(label_map[dataset_id])
            x += 1.0
        block_end = x - 1.0
        block_centers.append((topology, (block_start + block_end) / 2.0))
        block_ranges[topology] = (block_start - 0.34, block_end + 0.34)
        x += gap

    height = max(105.0 / MM_PER_INCH, MULTI_TOPOLOGY_PANEL_HEIGHT * len(properties))
    fig, axes = plt.subplots(
        len(properties),
        1,
        figsize=(FIGURE_WIDTH, height),
        sharex=True,
    )
    axes_array = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for property_index, (ax, property_name) in enumerate(zip(axes_array, properties)):
        ax.set_facecolor("white")
        ax.text(
            0.008,
            0.94,
            panel_letter(property_index),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.0,
            fontweight="bold",
        )
        combined = np.concatenate([
            property_values(df, property_name=property_name),
            property_values(ref_df, property_name=property_name),
        ])
        if combined.size == 0:
            raise ValueError(f"No plottable values for property {property_name!r}.")
        y_low, y_high = np.nanpercentile(combined, [0.6, 99.4])
        pad = (y_high - y_low) * 0.16 if y_high > y_low else 1.0

        for topology in topologies:
            topo_df = df[df["topology"] == topology_label(topology)]
            for dataset_id in model_order:
                values = property_values(topo_df, property_name=property_name, dataset_id=dataset_id)
                if values.size == 0:
                    continue
                xpos = positions[(topology, dataset_id)]
                if values.size >= 2:
                    parts = ax.violinplot(
                        values,
                        positions=[xpos],
                        widths=0.68,
                        showmeans=False,
                        showmedians=False,
                        showextrema=False,
                    )
                    fill = our_color if dataset_id == "ours" else baseline_color
                    for body in parts["bodies"]:
                        body.set_facecolor(fill)
                        body.set_edgecolor(fill)
                        body.set_alpha(0.88 if dataset_id == "ours" else 0.68)
                        body.set_linewidth(0.4)
                q1, med, q3 = np.percentile(values, [25, 50, 75])
                ax.plot([xpos, xpos], [q1, q3], color="black", linewidth=0.75, zorder=4)
                ax.plot(xpos, med, marker="o", color="black", markersize=1.9, zorder=5)

            topo_ref = ref_df[ref_df["topology"] == topology_label(topology)]
            ref_values = property_values(topo_ref, property_name=property_name)
            if ref_values.size == 0:
                raise ValueError(f"Reference has no plottable values for {topology!r} / {property_name!r}.")
            x0, x1 = block_ranges[topology]
            ax.hlines(
                np.nanmedian(ref_values),
                x0,
                x1,
                colors=reference_color,
                linestyles=(0, (2.7, 2.0)),
                linewidth=0.9,
                zorder=2,
            )

        ax.set_ylim(y_low - pad, y_high + pad)
        ax.set_ylabel(PROPERTY_METADATA[property_name]["label"], fontsize=6.0)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.tick_params(axis="y", labelsize=6.0, length=2.2, pad=2)
        ax.tick_params(axis="x", length=2.2, pad=1.5)

    top_ax = axes_array[0]
    for topology, center in block_centers:
        top_ax.text(
            center,
            1.10,
            topology_label(topology),
            transform=top_ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.0,
            weight="bold",
        )

    axes_array[-1].set_xticks(xticks)
    axes_array[-1].set_xticklabels(xticklabels, rotation=65, ha="right", fontsize=6.0)
    for ax in axes_array[:-1]:
        ax.tick_params(labelbottom=False)

    if panel_label:
        fig.text(0.015, 0.985, panel_label, ha="left", va="top", fontsize=6.0, weight="bold")

    legend_elements = [
        Line2D([0], [0], color=baseline_color, lw=6, solid_capstyle="butt", label="Baselines"),
        Line2D([0], [0], color=our_color, lw=6, solid_capstyle="butt", label="TopoCycGPT"),
        Line2D([0], [0], color=reference_color, lw=1.0, linestyle=(0, (2.7, 2.0)), label="Reference median"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.987),
        ncol=3,
        frameon=False,
        fontsize=6.0,
        handlelength=1.8,
        columnspacing=1.5,
        handletextpad=0.5,
    )

    plt.subplots_adjust(left=0.09, right=0.995, top=0.91, bottom=0.15, hspace=0.18)
    return save_figure(fig, output_prefix.parent, output_prefix.name, formats)


def write_run_record(
    path: Path,
    *,
    command: str,
    manifest_path: Path | None,
    topology_configs: list[tuple[str, Path]] | None,
    output_dir: Path,
    topology: str | None,
    properties: list[str],
    outputs: list[str],
    summary: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Sc-to-tail Property Distribution Plot Record",
        "",
        f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Command: `{command}`",
        f"- Output directory: `{output_dir}`",
        f"- Properties: {', '.join(f'`{p}`' for p in properties)}",
        f"- Max rows per dataset: `{args.max_rows_per_dataset}`",
        "",
        "## Inputs",
        "",
    ]
    if topology_configs is not None:
        lines.append("- Topology configs:")
        for topology_name, manifest in topology_configs:
            lines.append(f"  - `{topology_name}`: `{manifest}`")
    else:
        lines.append(f"- Manifest: `{manifest_path}`")
        lines.append(f"- Topology: `{topology}`")
    lines.extend([
        "",
        "| topology | dataset_id | label | raw_rows | rows_after_topology_filter | rows_after_numeric_filter | plotted_rows | filter_mode | path |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ])
    for item in summary:
        row = dict(item)
        row.setdefault("topology", topology or "")
        lines.append(
            "| {topology} | {dataset_id} | {dataset_label} | {raw_rows} | {rows_after_topology_filter} | "
            "{rows_after_numeric_filter} | {plotted_rows} | {filter_mode} | `{path}` |".format(**row)
        )
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

    layout = prepare_output_dirs(args.out_dir)
    if args.single_topology:
        datasets = load_manifest(args.manifest)
        generated, reference, model_order, label_map, summary = load_property_data(
            datasets,
            topology=args.topology,
            properties=properties,
            max_rows=args.max_rows_per_dataset,
            seed=args.seed,
        )
        output_stem = args.output_prefix or f"{args.topology}_property_distribution_violin"
        outputs = plot_property_violin_template(
            generated,
            reference,
            properties=properties,
            model_order=model_order,
            label_map=label_map,
            topology=args.topology,
            output_prefix=layout["figures"] / output_stem,
            formats=args.formats,
            panel_label=args.panel_label,
        )
        topology_configs = None
        manifest_path: Path | None = args.manifest
        topology_name: str | None = args.topology
    else:
        topology_configs = parse_topology_configs(args.topology_config)
        generated, reference, model_order, label_map, summary = load_multi_topology_data(
            topology_configs,
            properties=properties,
            max_rows=args.max_rows_per_dataset,
            seed=args.seed,
        )
        topologies = [topology for topology, _ in topology_configs]
        output_stem = args.output_prefix or "sc_disulfide_property_distribution_violin"
        outputs = plot_multi_topology_violin_template(
            generated,
            reference,
            properties=properties,
            model_order=model_order,
            label_map=label_map,
            topologies=topologies,
            output_prefix=layout["figures"] / output_stem,
            formats=args.formats,
            panel_label=args.panel_label,
        )
        manifest_path = None
        topology_name = None

    summary_path = layout["tables"] / "source_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    outputs.append(str(summary_path))

    run_record = layout["root"] / "run_record.md"
    write_run_record(
        run_record,
        command=build_command(),
        manifest_path=manifest_path,
        topology_configs=topology_configs,
        output_dir=args.out_dir,
        topology=topology_name,
        properties=properties,
        outputs=outputs,
        summary=summary,
        args=args,
    )
    print(json.dumps({"output_dir": str(args.out_dir), "outputs": outputs, "run_record": str(run_record)}, indent=2))


if __name__ == "__main__":
    main()
