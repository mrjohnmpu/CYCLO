#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


DEFAULT_PROPERTIES = ["permeability", "hemo"]
DEFAULT_FORMATS = ["png", "pdf", "svg"]
DEFAULT_STAGE_COLORS = {
    "before_fill": "#D7D7D7",
    "before_line": "#606060",
}
DEFAULT_TOPOLOGY_COLORS = {
    "head-to-tail": "#2C7FB8",
    "head_to_tail": "#2C7FB8",
    "sc-to-tail": "#D95F02",
    "sc_to_tail": "#D95F02",
    "disulfide-bridge": "#1B9E77",
    "disulfide_bridge": "#1B9E77",
}
FALLBACK_COLORS = [
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
class ComparisonSpec:
    topology: str
    label: str
    before_csv: Path
    after_csv: Path
    color: str


@dataclass(frozen=True)
class PropertySpec:
    name: str
    source_column: str
    output_name: str
    xlabel: str
    transform: str | None = None
    xlim: tuple[float, float] | None = None
    color: str | None = None


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


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "value"


def apply_nature_style(font_size: float = 7.0, font_family: str = "Arial") -> None:
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": [font_family, "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )


def parse_xlim(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("xlim must be a two-item list, for example [-10, -4]")
    low = float(value[0])
    high = float(value[1])
    if not low < high:
        raise ValueError(f"Invalid xlim {value!r}: lower bound must be smaller than upper bound")
    return low, high


def property_from_name(name: str, overrides: dict[str, Any] | None = None) -> PropertySpec:
    normalized = name.strip()
    lower = normalized.lower()
    overrides = overrides or {}
    property_color = str(overrides.get("color", "")).strip() or None

    if lower in {"permeability", "pampa", "pampa_value"}:
        spec = PropertySpec(
            name=normalized,
            source_column="PAMPA_value",
            output_name="permeability",
            xlabel="PAMPA value",
            xlim=parse_xlim(overrides.get("xlim")),
        )
    elif lower in {"hemo", "hemo_safe", "non_hemolytic", "non_hemolytic_score"}:
        spec = PropertySpec(
            name=normalized,
            source_column="hemolysis_score",
            output_name="hemo_safe",
            xlabel="Non-hemolytic score (1 - hemolysis score)",
            transform="one_minus",
            xlim=parse_xlim(overrides.get("xlim")),
        )
    elif lower in {"hemo_raw", "hemolysis", "hemolysis_score"}:
        spec = PropertySpec(
            name=normalized,
            source_column="hemolysis_score",
            output_name="hemolysis_score",
            xlabel="Hemolysis score",
            xlim=parse_xlim(overrides.get("xlim")),
        )
    else:
        source_column = str(overrides.get("source_column", overrides.get("column", normalized))).strip()
        spec = PropertySpec(
            name=normalized,
            source_column=source_column,
            output_name=slugify(str(overrides.get("output_name", source_column))),
            xlabel=str(overrides.get("xlabel", source_column)),
            transform=overrides.get("transform"),
            xlim=parse_xlim(overrides.get("xlim")),
        )

    if property_color is not None:
        spec = PropertySpec(
            name=spec.name,
            source_column=spec.source_column,
            output_name=spec.output_name,
            xlabel=spec.xlabel,
            transform=spec.transform,
            xlim=spec.xlim,
            color=property_color,
        )

    if "xlabel" in overrides:
        spec = PropertySpec(
            name=spec.name,
            source_column=spec.source_column,
            output_name=spec.output_name,
            xlabel=str(overrides["xlabel"]),
            transform=spec.transform,
            xlim=spec.xlim,
            color=spec.color,
        )
    if "output_name" in overrides:
        spec = PropertySpec(
            name=spec.name,
            source_column=spec.source_column,
            output_name=slugify(str(overrides["output_name"])),
            xlabel=spec.xlabel,
            transform=spec.transform,
            xlim=spec.xlim,
            color=spec.color,
        )
    return spec


def load_manifest(path: Path) -> tuple[list[ComparisonSpec], list[PropertySpec] | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object")

    items = payload.get("comparisons")
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a non-empty 'comparisons' list")

    comparisons: list[ComparisonSpec] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Comparison {index} must be an object")
        topology = str(item.get("topology", "")).strip()
        if not topology:
            raise ValueError(f"Comparison {index} requires 'topology'")
        before_csv = Path(str(item.get("before_csv", "")).strip())
        after_csv = Path(str(item.get("after_csv", "")).strip())
        if not str(before_csv) or not str(after_csv):
            raise ValueError(f"Comparison {topology!r} requires before_csv and after_csv")
        label = str(item.get("label", topology)).strip()
        color = str(
            item.get(
                "color",
                DEFAULT_TOPOLOGY_COLORS.get(topology, DEFAULT_TOPOLOGY_COLORS.get(slugify(topology), "")),
            )
        ).strip()
        if not color:
            color = FALLBACK_COLORS[len(comparisons) % len(FALLBACK_COLORS)]
        comparisons.append(
            ComparisonSpec(
                topology=topology,
                label=label,
                before_csv=before_csv,
                after_csv=after_csv,
                color=color,
            )
        )

    manifest_properties = payload.get("properties")
    if manifest_properties is None:
        return comparisons, None

    property_overrides = payload.get("property_specs", {})
    if property_overrides is None:
        property_overrides = {}
    if not isinstance(property_overrides, dict):
        raise ValueError("Manifest 'property_specs' must be an object when provided")

    properties = parse_properties(manifest_properties, property_overrides=property_overrides)
    return comparisons, properties


def parse_properties(
    values: list[str] | list[Any] | Any,
    *,
    property_overrides: dict[str, Any] | None = None,
) -> list[PropertySpec]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not values:
        raise ValueError("Properties must be a non-empty list")

    property_overrides = property_overrides or {}
    specs: list[PropertySpec] = []
    for value in values:
        if isinstance(value, str):
            overrides = property_overrides.get(value, {})
            if overrides is None:
                overrides = {}
            if not isinstance(overrides, dict):
                raise ValueError(f"Property override for {value!r} must be an object")
            specs.append(property_from_name(value, overrides))
        elif isinstance(value, dict):
            name = str(value.get("name", value.get("column", ""))).strip()
            if not name:
                raise ValueError("Property objects require a 'name' or 'column'")
            specs.append(property_from_name(name, value))
        else:
            raise ValueError(f"Invalid property value: {value!r}")
    return specs


def read_property_values(path: Path, spec: PropertySpec) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    dataframe = pd.read_csv(path, usecols=lambda column: column == spec.source_column)
    if spec.source_column not in dataframe.columns:
        header = pd.read_csv(path, nrows=0)
        raise ValueError(
            f"Column {spec.source_column!r} not found in {path}. "
            f"Available columns: {', '.join(header.columns)}"
        )

    raw = pd.to_numeric(dataframe[spec.source_column], errors="coerce")
    values = raw.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if spec.transform == "one_minus":
        finite = 1.0 - finite
    elif spec.transform in {None, ""}:
        pass
    else:
        raise ValueError(f"Unsupported transform for {spec.name!r}: {spec.transform!r}")

    summary = summarize_values(finite)
    summary.update(
        {
            "path": str(path),
            "column": spec.source_column,
            "transform": spec.transform or "",
            "rows": int(len(values)),
            "valid_n": int(finite.size),
            "missing_n": int(len(values) - finite.size),
        }
    )
    return finite, summary


def summarize_values(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def resolve_xlim(property_values: dict[str, dict[str, dict[str, np.ndarray]]], spec: PropertySpec) -> tuple[float, float] | None:
    if spec.xlim is not None:
        return spec.xlim

    values = [
        array
        for topology_values in property_values.values()
        for array in topology_values.values()
        if array.size
    ]
    if not values:
        return None

    combined = np.concatenate(values)
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.05)
    else:
        padding = (upper - lower) * 0.05
    return lower - padding, upper + padding


def evaluate_kde(values: np.ndarray, xlim: tuple[float, float] | None) -> tuple[np.ndarray, np.ndarray] | None:
    values = values[np.isfinite(values)]
    if values.size < 2 or np.unique(values).size < 2:
        return None

    if xlim is None:
        lower = float(np.min(values))
        upper = float(np.max(values))
        padding = (upper - lower) * 0.05 if upper > lower else max(abs(lower) * 0.05, 0.05)
        xlim = (lower - padding, upper + padding)

    grid = np.linspace(xlim[0], xlim[1], 512)
    kde = stats.gaussian_kde(values)
    density = kde(grid)
    return grid, density


def format_property_title(spec: PropertySpec) -> str:
    if spec.output_name == "permeability":
        return "Permeability"
    if spec.output_name == "hemolysis_score":
        return "Hemolysis"
    if spec.output_name == "hemo_safe":
        return "Hemolysis safety"
    return spec.output_name.replace("_", " ").title()


def format_property_xlabel(spec: PropertySpec) -> str:
    if spec.output_name == "permeability":
        return "Permeability"
    if spec.output_name == "hemolysis_score":
        return "Hemolysis"
    if spec.output_name == "hemo_safe":
        return "Hemolysis safety"
    return spec.xlabel


def add_stats_annotation(
    ax: plt.Axes,
    *,
    before_values: np.ndarray,
    after_values: np.ndarray,
    before_label: str,
    after_label: str,
    font_size: float,
) -> None:
    before_summary = summarize_values(before_values)
    after_summary = summarize_values(after_values)
    text = (
        "{before_label}: mean {before_mean:.2f}, median {before_median:.2f}\n"
        "{after_label}: mean {after_mean:.2f}, median {after_median:.2f}"
    ).format(
        before_label=before_label,
        after_label=after_label,
        before_mean=before_summary["mean"],
        before_median=before_summary["median"],
        after_mean=after_summary["mean"],
        after_median=after_summary["median"],
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="0.2",
        fontsize=font_size,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )


def add_curve_stats_annotations(
    ax: plt.Axes,
    *,
    fig: plt.Figure,
    curve_items: list[dict[str, Any]],
    font_size: float,
) -> None:
    prepared_items = []
    for item in curve_items:
        values = item["values"]
        grid = item["grid"]
        density = item["density"]
        if values.size == 0 or grid.size == 0 or density.size == 0:
            continue

        peak_index = int(np.argmax(density))
        anchor = (float(grid[peak_index]), float(density[peak_index]))
        summary = summarize_values(values)
        prepared_items.append(
            {
                **item,
                "anchor": anchor,
                "text": "Mean {mean:.2f}\nMedian {median:.2f}".format(
                    mean=summary["mean"],
                    median=summary["median"],
                ),
            }
        )

    if not prepared_items:
        return

    annotations = []
    for item in prepared_items:
        anchor_axes = ax.transAxes.inverted().transform(ax.transData.transform(item["anchor"]))
        anchor_x = float(anchor_axes[0])
        anchor_y = float(anchor_axes[1])
        if anchor_x < 0.5:
            text_x = min(anchor_x + 0.25, 0.72)
            ha = "left"
        else:
            text_x = max(anchor_x - 0.25, 0.08)
            ha = "right"
        if anchor_y > 0.70:
            text_y = max(anchor_y - 0.20, 0.58)
            va = "top"
        else:
            text_y = min(anchor_y + 0.16, 0.62)
            va = "bottom"
        annotation = ax.annotate(
            item["text"],
            xy=item["anchor"],
            xytext=(text_x, text_y),
            textcoords=ax.transAxes,
            ha=ha,
            va=va,
            color=item["line_color"],
            fontsize=font_size,
            arrowprops={
                "arrowstyle": "-",
                "color": item["line_color"],
                "linewidth": 0.6,
                "shrinkA": 0,
                "shrinkB": 2,
            },
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
        )
        annotations.append(annotation)

    if len(annotations) < 2:
        return

    def annotations_overlap() -> bool:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        first_box = annotations[0].get_window_extent(renderer).expanded(1.05, 1.10)
        second_box = annotations[1].get_window_extent(renderer).expanded(1.05, 1.10)
        return first_box.overlaps(second_box)

    if not annotations_overlap():
        return

    fallback_positions = [
        (0.70, 0.82, "left", "top"),
        (0.70, 0.58, "left", "center"),
        (0.30, 0.82, "right", "top"),
        (0.30, 0.58, "right", "center"),
        (0.58, 0.34, "left", "center"),
        (0.42, 0.34, "right", "center"),
    ]
    after_annotation = annotations[1]
    for text_x, text_y, ha, va in fallback_positions:
        after_annotation.set_position((text_x, text_y))
        after_annotation.set_ha(ha)
        after_annotation.set_va(va)
        if not annotations_overlap():
            break


def add_direction_arrow(ax: plt.Axes, spec: PropertySpec) -> None:
    arrow_specs = {
        "permeability": {
            "text": "higher permeability",
            "xy": (0.88, 0.83),
            "xytext": (0.58, 0.83),
            "color": "#2C7FB8",
        },
        "hemolysis_score": {
            "text": "lower hemolysis",
            "xy": (0.12, 0.83),
            "xytext": (0.42, 0.83),
            "color": "#A23B19",
        },
    }
    arrow_spec = arrow_specs.get(spec.output_name)
    if arrow_spec is None:
        return

    ax.annotate(
        arrow_spec["text"],
        xy=arrow_spec["xy"],
        xytext=arrow_spec["xytext"],
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        color=arrow_spec["color"],
        arrowprops={
            "arrowstyle": "Simple,tail_width=0.45,head_width=4.0,head_length=5.0",
            "facecolor": arrow_spec["color"],
            "edgecolor": arrow_spec["color"],
            "alpha": 0.88,
            "mutation_scale": 1.0,
        },
    )


def plot_distribution_shift(
    *,
    comparison: ComparisonSpec,
    spec: PropertySpec,
    before_values: np.ndarray,
    after_values: np.ndarray,
    xlim: tuple[float, float] | None,
    out_dir: Path,
    prefix: str,
    formats: list[str],
    dpi: int,
    font_size: float,
    font_family: str,
    single_width_mm: float,
    single_height_mm: float,
    pad_inches: float,
    annotate_stats: bool,
    annotate_curve_stats: bool,
    direction_arrows: bool,
    stats_font_size: float,
    before_label: str,
    after_label: str,
    legend_outside: bool,
) -> list[Path]:
    apply_nature_style(font_size=font_size, font_family=font_family)
    fig_w = mm_to_inches(single_width_mm)
    fig_h = mm_to_inches(single_height_mm)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    after_color = spec.color or comparison.color
    plot_items = [
        (before_label, before_values, DEFAULT_STAGE_COLORS["before_fill"], DEFAULT_STAGE_COLORS["before_line"], 0.36),
        (after_label, after_values, after_color, after_color, 0.26),
    ]

    curve_items: list[dict[str, Any]] = []
    for label, values, fill_color, line_color, alpha in plot_items:
        kde = evaluate_kde(values, xlim)
        if kde is None:
            continue
        grid, density = kde
        ax.fill_between(grid, density, color=fill_color, alpha=alpha, linewidth=0, label=label)
        ax.plot(grid, density, color=line_color, linewidth=1.4)
        curve_items.append(
            {
                "label": label,
                "values": values,
                "grid": grid,
                "density": density,
                "line_color": line_color,
            }
        )

    ax.set_xlabel(format_property_xlabel(spec))
    ax.set_ylabel("Density")
    ax.set_title(comparison.label)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylim(bottom=0)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if legend_outside:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False, handlelength=1.6, borderaxespad=0)
    else:
        ax.legend(loc="best", frameon=False, handlelength=1.6)
    if annotate_stats:
        add_stats_annotation(
            ax,
            before_values=before_values,
            after_values=after_values,
            before_label=before_label,
            after_label=after_label,
            font_size=stats_font_size,
        )
    if annotate_curve_stats:
        add_curve_stats_annotations(
            ax,
            fig=fig,
            curve_items=curve_items,
            font_size=stats_font_size,
        )
    if direction_arrows:
        add_direction_arrow(ax, spec)
    fig.tight_layout()

    topology_slug = slugify(comparison.topology)
    property_slug = slugify(spec.output_name)
    outputs: list[Path] = []
    for fmt in formats:
        out_path = out_dir / f"{prefix}_{topology_slug}_{property_slug}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        outputs.append(out_path)
    plt.close(fig)
    return outputs


def collect_data(
    comparisons: list[ComparisonSpec],
    properties: list[PropertySpec],
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    list[dict[str, Any]],
]:
    all_values: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    summary_rows: list[dict[str, Any]] = []

    for spec in properties:
        all_values[spec.output_name] = {}
        for comparison in comparisons:
            before_values, before_summary = read_property_values(comparison.before_csv, spec)
            after_values, after_summary = read_property_values(comparison.after_csv, spec)
            all_values[spec.output_name][comparison.topology] = {
                "before": before_values,
                "after": after_values,
            }
            for stage, summary in [("before", before_summary), ("after", after_summary)]:
                row = {
                    "topology": comparison.topology,
                    "label": comparison.label,
                    "stage": stage,
                    "property": spec.name,
                    "output_property": spec.output_name,
                    "xlabel": spec.xlabel,
                    **summary,
                }
                summary_rows.append(row)
    return all_values, summary_rows


def write_record(
    *,
    out_dir: Path,
    manifest: Path,
    comparisons: list[ComparisonSpec],
    properties: list[PropertySpec],
    formats: list[str],
    dpi: int,
    outputs: list[Path],
    summary_rows: list[dict[str, Any]],
    xlims: dict[str, tuple[float, float] | None],
    args: argparse.Namespace,
    font_family: str,
    font_path: str | None,
) -> Path:
    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": sys.argv,
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "formats": formats,
        "dpi": dpi,
        "font_size": args.font_size,
        "font_family": font_family,
        "font_path": font_path,
        "single_width_mm": args.single_width_mm,
        "single_height_mm": args.single_height_mm,
        "pad_inches": args.pad_inches,
        "annotate_stats": args.annotate_stats,
        "annotate_curve_stats": args.annotate_curve_stats,
        "stats_font_size": args.stats_font_size,
        "direction_arrows": args.direction_arrows,
        "before_label": args.before_label,
        "after_label": args.after_label,
        "legend_outside": args.legend_outside,
        "comparisons": [
            {
                "topology": comparison.topology,
                "label": comparison.label,
                "before_csv": str(comparison.before_csv),
                "after_csv": str(comparison.after_csv),
                "color": comparison.color,
            }
            for comparison in comparisons
        ],
        "properties": [
            {
                "name": spec.name,
                "source_column": spec.source_column,
                "output_name": spec.output_name,
                "xlabel": spec.xlabel,
                "transform": spec.transform,
                "xlim": xlims.get(spec.output_name),
                "color": spec.color,
            }
            for spec in properties
        ],
        "outputs": [str(path) for path in outputs],
        "summary_rows": summary_rows,
    }
    path = out_dir / "plot_record.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Nature-style before/after RL KDE distribution shifts for cyclic peptide properties."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSON file with topology before/after CSV pairs")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory for figures and records")
    parser.add_argument("--properties", nargs="+", help="Properties to plot; overrides manifest properties")
    parser.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS, help="Output formats, default: png pdf svg")
    parser.add_argument("--dpi", type=int, default=600, help="DPI for raster outputs")
    parser.add_argument("--prefix", default="rl_distribution_shift", help="Output filename prefix")
    parser.add_argument("--before_label", default="Before RL", help="Legend label for the before distribution")
    parser.add_argument("--after_label", default="After RL", help="Legend label for the after distribution")
    parser.add_argument("--legend_outside", action="store_true", help="Place the legend above the plot area")
    parser.add_argument("--font_size", type=float, default=7.0, help="Base matplotlib font size in points")
    parser.add_argument(
        "--font_path",
        type=Path,
        default=None,
        help="Optional path to a TrueType/OpenType font file. Defaults to common Arial paths when available.",
    )
    parser.add_argument("--single_width_mm", type=float, default=90.0, help="Single-panel figure width in mm")
    parser.add_argument("--single_height_mm", type=float, default=65.0, help="Single-panel figure height in mm")
    parser.add_argument("--pad_inches", type=float, default=0.1, help="Padding around tight bounding boxes")
    parser.add_argument("--annotate_stats", action="store_true", help="Annotate before/after means and medians")
    parser.add_argument(
        "--annotate_curve_stats",
        action="store_true",
        help="Annotate before/after means and medians next to their KDE curves",
    )
    parser.add_argument(
        "--stats_font_size",
        type=float,
        default=None,
        help="Font size for the mean/median annotation. Defaults to --font_size.",
    )
    parser.add_argument("--direction_arrows", action="store_true", help="Add directional improvement arrows")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.font_size <= 0:
        raise ValueError("--font_size must be positive")
    if args.stats_font_size is None:
        args.stats_font_size = args.font_size
    if args.stats_font_size <= 0:
        raise ValueError("--stats_font_size must be positive")
    if args.annotate_stats and args.annotate_curve_stats:
        raise ValueError("--annotate_stats and --annotate_curve_stats cannot be used together")
    if args.single_width_mm <= 0 or args.single_height_mm <= 0:
        raise ValueError("--single_width_mm and --single_height_mm must be positive")
    if args.pad_inches < 0:
        raise ValueError("--pad_inches must be non-negative")
    if not args.before_label.strip() or not args.after_label.strip():
        raise ValueError("Stage labels must be non-empty")
    if args.font_path is not None and not args.font_path.exists():
        raise FileNotFoundError(f"Font file not found: {args.font_path}")

    font_family, font_path = resolve_font(args.font_path)
    comparisons, manifest_properties = load_manifest(args.manifest)
    if args.properties:
        properties = parse_properties(args.properties)
    elif manifest_properties is not None:
        properties = manifest_properties
    else:
        properties = parse_properties(DEFAULT_PROPERTIES)

    formats = [fmt.lower().lstrip(".") for fmt in args.formats]
    unsupported = [fmt for fmt in formats if fmt not in {"png", "pdf", "svg"}]
    if unsupported:
        raise ValueError(f"Unsupported output format(s): {unsupported}. Use png, pdf, and/or svg.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_values, summary_rows = collect_data(comparisons, properties)

    xlims: dict[str, tuple[float, float] | None] = {}
    outputs: list[Path] = []
    for spec in properties:
        property_values = all_values[spec.output_name]
        xlim = resolve_xlim(property_values, spec)
        xlims[spec.output_name] = xlim
        for comparison in comparisons:
            topology_values = property_values[comparison.topology]
            outputs.extend(
                plot_distribution_shift(
                    comparison=comparison,
                    spec=spec,
                    before_values=topology_values["before"],
                    after_values=topology_values["after"],
                    xlim=xlim,
                    out_dir=args.out_dir,
                    prefix=args.prefix,
                    formats=formats,
                    dpi=args.dpi,
                    font_size=args.font_size,
                    font_family=font_family,
                    single_width_mm=args.single_width_mm,
                    single_height_mm=args.single_height_mm,
                    pad_inches=args.pad_inches,
                    annotate_stats=args.annotate_stats,
                    annotate_curve_stats=args.annotate_curve_stats,
                    direction_arrows=args.direction_arrows,
                    stats_font_size=args.stats_font_size,
                    before_label=args.before_label,
                    after_label=args.after_label,
                    legend_outside=args.legend_outside,
                )
            )

    summary_path = args.out_dir / "plot_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    record_path = write_record(
        out_dir=args.out_dir,
        manifest=args.manifest,
        comparisons=comparisons,
        properties=properties,
        formats=formats,
        dpi=args.dpi,
        outputs=outputs,
        summary_rows=summary_rows,
        xlims=xlims,
        args=args,
        font_family=font_family,
        font_path=font_path,
    )

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "summary_csv": str(summary_path),
                "plot_record": str(record_path),
                "n_outputs": len(outputs),
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
