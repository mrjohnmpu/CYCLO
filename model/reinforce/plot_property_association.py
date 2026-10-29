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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MM_PER_INCH = 25.4
PUBLICATION_FONT_NAME = "Arial"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "property_association_bubble.json"
DEFAULT_OUTPUT_DIR = Path('results/figures/property_association_bubble_20260616')
DEFAULT_FORMATS = ["png", "pdf", "svg"]
DEFAULT_DPI = 600
DEFAULT_RHO_LIMIT = 0.7
DEFAULT_Q_CUTOFF = 0.05
DEFAULT_MIN_N = 5
CORRPLOT_NEGATIVE_COLOR = "#73B6D5"
CORRPLOT_CENTER_COLOR = "#FFFFFF"
CORRPLOT_POSITIVE_COLOR = "#DD7B71"

TOPOLOGY_LABELS = {
    "head_to_tail": "Head-to-tail",
    "head-to-tail": "Head-to-tail",
    "sc_to_tail": "SC-to-tail",
    "sc-to-tail": "SC-to-tail",
    "disulfide_bridge": "Disulfide bridge",
    "disulfide-bridge": "Disulfide bridge",
}

PROPERTY_ALIASES = {
    "permeability": {"column": "PAMPA_value", "label": "Permeability", "stem": "permeability"},
    "pampa": {"column": "PAMPA_value", "label": "Permeability", "stem": "permeability"},
    "pampa_value": {"column": "PAMPA_value", "label": "Permeability", "stem": "permeability"},
    "hemolysis": {"column": "hemolysis_score", "label": "Hemolysis", "stem": "hemolysis"},
    "hemolysis_score": {"column": "hemolysis_score", "label": "Hemolysis", "stem": "hemolysis"},
    "non_natural_fraction": {
        "column": "non_natural_fraction",
        "label": "NCAA fraction",
        "stem": "non_natural_fraction",
        "derived": "non_natural_fraction",
    },
    "nnaa_fraction": {
        "column": "non_natural_fraction",
        "label": "NCAA fraction",
        "stem": "non_natural_fraction",
        "derived": "non_natural_fraction",
    },
    "frac_non_natural_resolved": {
        "column": "frac_non_natural_resolved",
        "label": "NCAA fraction",
        "stem": "non_natural_fraction",
    },
    "mw": {"column": "MW", "label": "MW", "stem": "mw"},
    "molecular_weight": {"column": "MW", "label": "MW", "stem": "mw"},
    "tpsa": {"column": "TPSA", "label": "TPSA", "stem": "tpsa"},
    "hbd": {"column": "HBD", "label": "HBD", "stem": "hbd"},
    "hba": {"column": "HBA", "label": "HBA", "stem": "hba"},
    "logp": {"column": "LogP", "label": "LogP", "stem": "logp"},
    "clogp": {"column": "LogP", "label": "LogP", "stem": "logp"},
    "rotbonds": {"column": "RotBonds", "label": "Rotatable bonds", "stem": "rotbonds"},
    "rotatable_bonds": {"column": "RotBonds", "label": "Rotatable bonds", "stem": "rotbonds"},
    "peptide_length": {"column": "peptide_length", "label": "Peptide length", "stem": "peptide_length"},
}

NATURAL_RESIDUE_NAMES = {
    "alanine",
    "arginine",
    "asparagine",
    "aspartic acid",
    "cysteine",
    "glutamic acid",
    "glutamine",
    "glycine",
    "histidine",
    "isoleucine",
    "leucine",
    "lysine",
    "methionine",
    "phenylalanine",
    "proline",
    "serine",
    "threonine",
    "tryptophan",
    "tyrosine",
    "valine",
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
    parser = argparse.ArgumentParser(description="Draw Nature-style property association bubble heatmaps.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--formats", nargs="+", default=None, choices=["png", "pdf", "svg"])
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--plot-style", choices=["bubble", "corrplot"], default="bubble")
    parser.add_argument(
        "--corrplot-orientation",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Corrplot matrix orientation. vertical: topology columns x feature rows; horizontal: feature columns x topology rows.",
    )
    parser.add_argument(
        "--rho-text-color",
        choices=["auto", "black"],
        default="auto",
        help="Color for rho labels in corrplot cells.",
    )
    parser.add_argument("--font-size", type=float, default=7.0, help="Base font size in points.")
    parser.add_argument("--font-path", type=Path, default=None, help="Optional font file to load explicitly.")
    parser.add_argument("--figure-width-mm", type=float, default=None, help="Override figure width in millimetres.")
    parser.add_argument("--figure-height-mm", type=float, default=None, help="Override figure height in millimetres.")
    parser.add_argument("--pad-inches", type=float, default=0.1, help="Padding around tight saved figures.")
    parser.add_argument("--corrplot-min-bubble-size", type=float, default=40.0)
    parser.add_argument("--corrplot-max-bubble-size", type=float, default=850.0)
    parser.add_argument(
        "--min-abs-rho-circle",
        type=float,
        default=0.0,
        help="Hide circles with abs(rho) below this threshold; rho values can still be shown.",
    )
    parser.add_argument("--show-rho-values", action="store_true", help="Print rho values inside corrplot cells.")
    parser.add_argument("--hide-significance-stars", action="store_true", help="Do not mark q-value significance stars.")
    args = parser.parse_args()
    if args.font_size <= 0:
        parser.error("--font-size must be positive")
    if args.figure_width_mm is not None and args.figure_width_mm <= 0:
        parser.error("--figure-width-mm must be positive")
    if args.figure_height_mm is not None and args.figure_height_mm <= 0:
        parser.error("--figure-height-mm must be positive")
    if args.pad_inches < 0:
        parser.error("--pad-inches must be non-negative")
    if args.corrplot_min_bubble_size < 0:
        parser.error("--corrplot-min-bubble-size must be non-negative")
    if args.corrplot_max_bubble_size <= args.corrplot_min_bubble_size:
        parser.error("--corrplot-max-bubble-size must be greater than --corrplot-min-bubble-size")
    if args.min_abs_rho_circle < 0:
        parser.error("--min-abs-rho-circle must be non-negative")
    return args


def apply_publication_style(font_size: float, font_path: Path | None = None) -> str:
    font_name = PUBLICATION_FONT_NAME
    if font_path is not None:
        if not font_path.is_file():
            raise FileNotFoundError(f"Font file not found: {font_path}")
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    rcparams = {
        **PUBLICATION_RCPARAMS,
        "font.sans-serif": [font_name, PUBLICATION_FONT_NAME, "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    }
    plt.rcParams.update(rcparams)
    try:
        font_manager.findfont(font_manager.FontProperties(family=font_name), fallback_to_default=False)
    except ValueError:
        warnings.warn(f"{font_name} is not available; using the closest sans-serif fallback.", RuntimeWarning)
    return font_name


def normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def safe_stem(value: str) -> str:
    stem = normalize_key(value)
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem).strip("_")


def default_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def resolve_property(spec: str | dict[str, Any]) -> dict[str, str]:
    if isinstance(spec, str):
        key = normalize_key(spec)
        alias = PROPERTY_ALIASES.get(key)
        if alias is not None:
            return {"id": key, **alias}
        return {"id": key, "column": spec, "label": default_label(spec), "stem": safe_stem(spec)}
    if not isinstance(spec, dict):
        raise TypeError(f"Property spec must be a string or object, got {type(spec)!r}")
    name = str(spec.get("id") or spec.get("name") or spec.get("column") or "").strip()
    if not name:
        raise ValueError(f"Property object is missing id/name/column: {spec!r}")
    resolved = resolve_property(name)
    if spec.get("column"):
        resolved["column"] = str(spec["column"])
    if spec.get("label"):
        resolved["label"] = str(spec["label"])
    if spec.get("stem"):
        resolved["stem"] = safe_stem(str(spec["stem"]))
    if spec.get("derived"):
        resolved["derived"] = str(spec["derived"])
    resolved["id"] = normalize_key(str(spec.get("id") or name))
    return resolved


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config.get("datasets"), list) or not config["datasets"]:
        raise ValueError("Config must contain a non-empty 'datasets' list.")
    if not isinstance(config.get("targets"), list) or not config["targets"]:
        raise ValueError("Config must contain a non-empty 'targets' list.")
    if not isinstance(config.get("features"), list) or not config["features"]:
        raise ValueError("Config must contain a non-empty 'features' list.")
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


def topology_label(topology_id: str) -> str:
    return TOPOLOGY_LABELS.get(topology_id, default_label(topology_id))


def filter_topology(frame: pd.DataFrame, dataset: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    topology_id = str(dataset["id"])
    filter_value = str(dataset.get("filter_value") or topology_id)
    if dataset.get("filter_topology", True) and "Cyclization" in frame.columns:
        filtered = frame[frame["Cyclization"].astype(str).str.lower() == filter_value.lower()].copy()
        mode = f"Cyclization == {filter_value}"
        if not filtered.empty:
            return filtered.reset_index(drop=True), mode
    return frame.copy().reset_index(drop=True), "kept all rows"


def split_residue_sequence(sequence: object) -> list[str]:
    if sequence is None or pd.isna(sequence):
        return []
    return [token.strip() for token in str(sequence).split("|") if token.strip()]


def is_natural_residue(residue: str) -> bool:
    token = residue.strip()
    lower = token.lower()
    if not token:
        return False
    if token.startswith("#") or "[nmet]" in lower or "n-methyl" in lower or "n-me" in lower:
        return False
    if "[" in token or "]" in token:
        return False
    if token.startswith("L-") or token.startswith("D-"):
        token = token[2:]
    token = " ".join(token.strip().lower().split())
    return token in NATURAL_RESIDUE_NAMES


def compute_non_natural_fraction(sequence: object, peptide_length: object) -> float:
    residues = split_residue_sequence(sequence)
    length = pd.to_numeric(pd.Series([peptide_length]), errors="coerce").iloc[0]
    if pd.isna(length) or length <= 0:
        return float("nan")
    length_int = int(length)
    non_natural_count = sum(not is_natural_residue(residue) for residue in residues)
    return float(non_natural_count / length_int)


def derived_required_columns(properties: list[dict[str, str]]) -> list[str]:
    required: list[str] = []
    for item in properties:
        if item.get("derived") == "non_natural_fraction":
            required.extend(["peptide_sequence", "peptide_length"])
    return required


def add_derived_columns(frame: pd.DataFrame, properties: list[dict[str, str]]) -> pd.DataFrame:
    result = frame.copy()
    if any(item.get("derived") == "non_natural_fraction" for item in properties):
        missing = [column for column in ("peptide_sequence", "peptide_length") if column not in result.columns]
        if missing:
            raise ValueError(f"Cannot derive non_natural_fraction; missing columns: {missing}")
        result["non_natural_fraction"] = [
            compute_non_natural_fraction(sequence, length)
            for sequence, length in zip(result["peptide_sequence"], result["peptide_length"], strict=False)
        ]
    return result


def required_columns(targets: list[dict[str, str]], features: list[dict[str, str]]) -> list[str]:
    properties = [*targets, *features]
    base_columns = [item["column"] for item in properties if not item.get("derived")]
    return list(dict.fromkeys([*base_columns, *derived_required_columns(properties)]))


def analysis_columns(targets: list[dict[str, str]], features: list[dict[str, str]]) -> list[str]:
    return list(dict.fromkeys([item["column"] for item in [*targets, *features]]))


def load_datasets(
    datasets: list[dict[str, Any]],
    *,
    required_input_columns: list[str],
    required_analysis_columns: list[str],
    properties: list[dict[str, str]],
    max_rows_per_dataset: int | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    summary: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for dataset in datasets:
        topology_id = str(dataset.get("id") or "").strip()
        if not topology_id or topology_id in seen_ids:
            raise ValueError(f"Invalid or duplicate dataset id: {topology_id!r}")
        seen_ids.add(topology_id)
        label = str(dataset.get("label") or topology_label(topology_id))
        path = Path(str(dataset.get("path") or ""))
        if not path.is_file():
            raise FileNotFoundError(f"Missing CSV for {topology_id}: {path}")
        raw = pd.read_csv(path)
        filtered, filter_mode = filter_topology(raw, dataset)
        missing_columns = [column for column in required_input_columns if column not in filtered.columns]
        if missing_columns:
            raise ValueError(f"{path} is missing required columns: {missing_columns}")
        use = add_derived_columns(filtered, properties)
        if max_rows_per_dataset is not None and max_rows_per_dataset > 0:
            use = use.head(max_rows_per_dataset).copy()
        for column in required_analysis_columns:
            use[column] = pd.to_numeric(use[column], errors="coerce")
        missing_values = {column: int(use[column].isna().sum()) for column in required_analysis_columns}
        complete_rows = int(use[required_analysis_columns].notna().all(axis=1).sum())
        use["topology"] = topology_id
        use["topology_label"] = label
        frames.append(use)
        summary.append({
            "topology": topology_id,
            "topology_label": label,
            "path": str(path),
            "filter_mode": filter_mode,
            "raw_rows": len(raw),
            "rows_after_topology_filter": len(filtered),
            "plotted_rows": len(use),
            "required_input_columns": ";".join(required_input_columns),
            "required_analysis_columns": ";".join(required_analysis_columns),
            "missing_columns": ";".join(missing_columns),
            "missing_values": json.dumps(missing_values, sort_keys=True),
            "complete_required_rows": complete_rows,
        })
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return merged, summary


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    q_values = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.notna()
    if not valid.any():
        return q_values
    values = p_values.loc[valid].to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    q_values.loc[valid] = out
    return q_values


def calculate_property_association(
    data: pd.DataFrame,
    *,
    target: dict[str, str],
    features: list[dict[str, str]],
    datasets: list[dict[str, Any]],
    min_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        topology_id = str(dataset["id"])
        topology_name = str(dataset.get("label") or topology_label(topology_id))
        subset = data[data["topology"] == topology_id].copy()
        y = subset[target["column"]]
        for feature in features:
            x = subset[feature["column"]]
            valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
            x_valid = x.to_numpy(dtype=float)[valid]
            y_valid = y.to_numpy(dtype=float)[valid]
            if len(x_valid) < min_n or np.nanstd(x_valid) == 0 or np.nanstd(y_valid) == 0:
                rho = np.nan
                pval = np.nan
            else:
                rho, pval = spearmanr(x_valid, y_valid)
            rows.append({
                "target": target["id"],
                "target_column": target["column"],
                "target_label": target["label"],
                "feature": feature["id"],
                "feature_column": feature["column"],
                "feature_label": feature["label"],
                "topology": topology_id,
                "topology_label": topology_name,
                "rho": rho,
                "pval": pval,
                "n": int(len(x_valid)),
            })
    result = pd.DataFrame(rows)
    result["qval"] = benjamini_hochberg(result["pval"])
    return result


def bubble_size(rho: float, rho_limit: float, *, min_size: float = 18.0, max_size: float = 520.0) -> float:
    scaled = min(abs(float(rho)), rho_limit) / rho_limit if rho_limit > 0 else 0.0
    return min_size + (max_size - min_size) * (scaled ** 1.45)


def corrplot_bubble_size(rho: float, rho_limit: float, *, min_size: float, max_size: float) -> float:
    return bubble_size(rho, rho_limit, min_size=min_size, max_size=max_size)


def corrplot_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "corrplot_blue_white_red",
        [CORRPLOT_NEGATIVE_COLOR, CORRPLOT_CENTER_COLOR, CORRPLOT_POSITIVE_COLOR],
        N=256,
    )


def display_feature_label(label: str) -> str:
    return "NCAA" if label == "NCAA fraction" else label


def plot_bubble_heatmap(
    assoc_df: pd.DataFrame,
    *,
    target: dict[str, str],
    features: list[dict[str, str]],
    datasets: list[dict[str, Any]],
    out_prefix: Path,
    formats: Iterable[str],
    dpi: int,
    rho_limit: float,
    q_cutoff: float,
    panel_label: str | None,
    hide_significance_stars: bool,
    font_size: float,
    figure_width_mm: float | None,
    figure_height_mm: float | None,
    pad_inches: float,
) -> list[str]:
    topology_order = [str(dataset["id"]) for dataset in datasets]
    topology_labels = [str(dataset.get("label") or topology_label(str(dataset["id"]))) for dataset in datasets]
    feature_ids = [feature["id"] for feature in features]
    feature_labels = [display_feature_label(feature["label"]) for feature in features]
    x_positions = {topology_id: idx * 0.78 for idx, topology_id in enumerate(topology_order)}
    y_positions = {feature_id: idx for idx, feature_id in enumerate(feature_ids)}
    fig_w_mm = figure_width_mm or max(105.0, min(135.0, 66.0 + 23.0 * len(topology_order)))
    fig_h_mm = figure_height_mm or max(68.0, min(130.0, 34.0 + 10.0 * len(features)))
    fig_w = fig_w_mm / MM_PER_INCH
    fig_h = fig_h_mm / MM_PER_INCH
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(vmin=-rho_limit, vmax=rho_limit)

    for _, row in assoc_df.iterrows():
        rho = row["rho"]
        if not np.isfinite(rho):
            continue
        x = x_positions[row["topology"]]
        y = y_positions[row["feature"]]
        ax.scatter(
            x,
            y,
            s=bubble_size(rho, rho_limit),
            c=[rho],
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.92,
            zorder=3,
        )
        if not hide_significance_stars and np.isfinite(row["qval"]) and row["qval"] < q_cutoff:
            ax.text(x, y, "*", ha="center", va="center", fontsize=font_size, fontweight="bold", color="black", zorder=4)

    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(feature_labels)
    ax.invert_yaxis()
    ax.set_xticks([x_positions[topology_id] for topology_id in topology_order])
    ax.set_xticklabels(topology_labels)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=3, pad=3)
    ax.set_xlim(min(x_positions.values()) - 0.38, max(x_positions.values()) + 0.38)
    ax.set_ylim(len(feature_labels) - 0.35, -0.65)
    ax.set_title(f"{target['label']}-property association", fontsize=font_size, pad=7)
    ax.grid(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)

    fig.subplots_adjust(left=0.25, right=0.68, top=0.88, bottom=0.16)

    colorbar_ax = fig.add_axes([0.735, 0.47, 0.035, 0.34])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    colorbar.set_label("Spearman \u03c1", fontsize=font_size, rotation=90, labelpad=5)
    colorbar.ax.tick_params(labelsize=font_size, width=0.6, length=2.5)
    colorbar.outline.set_linewidth(0.6)

    legend_rhos = [value for value in (0.1, 0.3, 0.5, rho_limit) if value <= rho_limit + 1e-9]
    legend_ax = fig.add_axes([0.725, 0.10, 0.24, 0.30])
    legend_ax.axis("off")
    legend_ax.text(0.0, 1.02, r"$|\rho|$", ha="left", va="bottom", fontsize=font_size)
    y0 = 0.82
    for idx, rho in enumerate(legend_rhos):
        y = y0 - idx * 0.20
        legend_ax.scatter(
            [0.16],
            [y],
            s=bubble_size(rho, rho_limit),
            facecolor="white",
            edgecolor="black",
            linewidth=0.45,
            clip_on=False,
        )
        legend_ax.text(0.42, y, f"{rho:.1f}", ha="left", va="center", fontsize=font_size)
    legend_ax.set_xlim(0.0, 1.0)
    legend_ax.set_ylim(0.0, 1.05)

    if panel_label:
        fig.text(0.015, 0.985, panel_label, fontsize=font_size + 5, fontweight="bold", va="top", ha="left")

    outputs: list[str] = []
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_prefix.with_suffix(f".{fmt}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": pad_inches}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def plot_corrplot_heatmap(
    assoc_df: pd.DataFrame,
    *,
    target: dict[str, str],
    features: list[dict[str, str]],
    datasets: list[dict[str, Any]],
    out_prefix: Path,
    formats: Iterable[str],
    dpi: int,
    rho_limit: float,
    q_cutoff: float,
    panel_label: str | None,
    min_abs_rho_circle: float,
    show_rho_values: bool,
    hide_significance_stars: bool,
    font_size: float,
    figure_width_mm: float | None,
    figure_height_mm: float | None,
    pad_inches: float,
    corrplot_min_bubble_size: float,
    corrplot_max_bubble_size: float,
    corrplot_orientation: str,
    rho_text_color: str,
) -> list[str]:
    topology_order = [str(dataset["id"]) for dataset in datasets]
    topology_labels = [str(dataset.get("label") or topology_label(str(dataset["id"]))) for dataset in datasets]
    feature_ids = [feature["id"] for feature in features]
    feature_labels = [display_feature_label(feature["label"]) for feature in features]
    if corrplot_orientation == "horizontal":
        x_items = feature_ids
        x_labels = feature_labels
        y_items = topology_order
        y_labels = topology_labels
        fig_w_mm = figure_width_mm or max(92.0, 35.0 + 11.0 * len(feature_ids))
        fig_h_mm = figure_height_mm or max(42.0, 24.0 + 6.0 * len(topology_order))
    else:
        x_items = topology_order
        x_labels = topology_labels
        y_items = feature_ids
        y_labels = feature_labels
        fig_w_mm = figure_width_mm or max(92.0, 58.0 + 19.0 * len(topology_order))
        fig_h_mm = figure_height_mm or max(66.0, 30.0 + 8.0 * len(features))
    x_positions = {item_id: idx for idx, item_id in enumerate(x_items)}
    y_positions = {item_id: idx for idx, item_id in enumerate(y_items)}
    fig_w = fig_w_mm / MM_PER_INCH
    fig_h = fig_h_mm / MM_PER_INCH
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    cmap = corrplot_colormap()
    norm = Normalize(vmin=-1.0, vmax=1.0)
    strong_text_threshold = max(0.30, min_abs_rho_circle)

    for _, row in assoc_df.iterrows():
        rho = row["rho"]
        if not np.isfinite(rho):
            continue
        if corrplot_orientation == "horizontal":
            x = x_positions[row["feature"]]
            y = y_positions[row["topology"]]
        else:
            x = x_positions[row["topology"]]
            y = y_positions[row["feature"]]
        has_circle = abs(float(rho)) >= min_abs_rho_circle
        if has_circle:
            ax.scatter(
                x,
                y,
                s=corrplot_bubble_size(
                    rho,
                    rho_limit,
                    min_size=corrplot_min_bubble_size,
                    max_size=corrplot_max_bubble_size,
                ),
                c=[rho],
                cmap=cmap,
                norm=norm,
                edgecolor="none",
                alpha=0.95,
                zorder=3,
            )
        if show_rho_values:
            text_color = (
                "black"
                if rho_text_color == "black"
                else "white" if has_circle and abs(float(rho)) >= strong_text_threshold else "black"
            )
            ax.text(
                x,
                y,
                f"{float(rho):.2f}",
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                color=text_color,
                zorder=4,
            )
        if not hide_significance_stars and np.isfinite(row["qval"]) and row["qval"] < q_cutoff:
            ax.text(x + 0.32, y - 0.32, "*", ha="center", va="center", fontsize=font_size, color="black", zorder=5)

    ax.set_xticks(range(len(x_labels)))
    if corrplot_orientation == "horizontal":
        ax.set_xticklabels(x_labels, rotation=0, ha="center")
    else:
        ax.set_xticklabels(x_labels, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.set_ylim(len(y_labels) - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="0.86", linewidth=0.45)
    ax.tick_params(axis="both", which="major", length=0)
    ax.tick_params(axis="both", which="minor", length=0)
    ax.set_title(f"{target['label']}-property association", fontsize=font_size, pad=7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    if corrplot_orientation == "horizontal":
        fig.subplots_adjust(left=0.28, right=0.82, top=0.88, bottom=0.19)
        colorbar_ax = fig.add_axes([0.86, 0.27, 0.032, 0.48])
    else:
        fig.subplots_adjust(left=0.28, right=0.78, top=0.91, bottom=0.22)
        colorbar_ax = fig.add_axes([0.83, 0.26, 0.035, 0.52])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    colorbar.set_label("Spearman \u03c1", fontsize=font_size, rotation=90, labelpad=5)
    colorbar.ax.tick_params(labelsize=font_size, width=0.6, length=2.5)
    colorbar.outline.set_linewidth(0.6)

    if panel_label:
        fig.text(0.015, 0.985, panel_label, fontsize=font_size + 5, fontweight="bold", va="top", ha="left")

    outputs: list[str] = []
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_prefix.with_suffix(f".{fmt}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": pad_inches}
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
    targets: list[dict[str, str]],
    features: list[dict[str, str]],
    outputs: list[str],
    source_summary: list[dict[str, Any]],
    plot_style: str,
    min_abs_rho_circle: float,
    show_rho_values: bool,
    hide_significance_stars: bool,
    font_name: str,
    font_path: Path | None,
    font_size: float,
    figure_width_mm: float | None,
    figure_height_mm: float | None,
    pad_inches: float,
    corrplot_min_bubble_size: float,
    corrplot_max_bubble_size: float,
    corrplot_orientation: str,
    rho_text_color: str,
    corrplot_negative_color: str,
    corrplot_positive_color: str,
) -> None:
    target_desc = ", ".join(f"`{item['id']}` -> `{item['column']}`" for item in targets)
    feature_desc = ", ".join(f"`{item['id']}` -> `{item['column']}`" for item in features)
    lines = [
        "# Property Association Bubble Plot Record",
        "",
        f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Command: `{command}`",
        f"- Config: `{config_path}`",
        f"- Output directory: `{config.get('output_dir', DEFAULT_OUTPUT_DIR)}`",
        f"- Plot style: `{plot_style}`",
        f"- Corrplot orientation: `{corrplot_orientation}`",
        f"- Rho text color: `{rho_text_color}`",
        f"- Font: `{font_name}`",
        f"- Font path: `{font_path}`",
        f"- Font size: `{font_size:g}` pt",
        f"- Figure size: `{figure_width_mm} mm x {figure_height_mm} mm`",
        f"- Pad inches: `{pad_inches:g}`",
        f"- Corrplot bubble size range: `{corrplot_min_bubble_size:g}-{corrplot_max_bubble_size:g}`",
        f"- Corrplot negative color: `{corrplot_negative_color}`",
        f"- Corrplot positive color: `{corrplot_positive_color}`",
        "- Corrplot color normalization: `[-1.0, 1.0]`",
        f"- Circle visibility threshold: `abs(rho) >= {min_abs_rho_circle:g}`",
        f"- Show rho values: `{show_rho_values}`",
        f"- Hide significance stars: `{hide_significance_stars}`",
        f"- Targets: {target_desc}",
        f"- Features: {feature_desc}",
        "",
        "## Inputs",
        "",
        "| topology | raw_rows | rows_after_topology_filter | plotted_rows | complete_required_rows | path |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in source_summary:
        lines.append(
            "| {topology_label} | {raw_rows} | {rows_after_topology_filter} | {plotted_rows} | "
            "{complete_required_rows} | `{path}` |".format(**item)
        )
    lines.extend(["", "## Outputs", ""])
    for output in outputs:
        lines.append(f"- `{output}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_command() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def main() -> None:
    args = parse_args()
    font_name = apply_publication_style(args.font_size, args.font_path)
    config = load_config(args.config)
    targets = [resolve_property(item) for item in config["targets"]]
    features = [resolve_property(item) for item in config["features"]]
    input_columns = required_columns(targets, features)
    columns = analysis_columns(targets, features)
    out_dir = args.out_dir or Path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    formats = args.formats or list(config.get("formats", DEFAULT_FORMATS))
    dpi = int(config.get("dpi", DEFAULT_DPI))
    rho_limit = float(config.get("rho_limit", DEFAULT_RHO_LIMIT))
    q_cutoff = float(config.get("q_cutoff", DEFAULT_Q_CUTOFF))
    min_n = int(config.get("min_n", DEFAULT_MIN_N))
    max_rows = args.max_rows_per_dataset
    if max_rows is None:
        max_rows = config.get("max_rows_per_dataset")
    layout = prepare_output_layout(out_dir)
    data, source_summary = load_datasets(
        config["datasets"],
        required_input_columns=input_columns,
        required_analysis_columns=columns,
        properties=[*targets, *features],
        max_rows_per_dataset=max_rows,
    )

    outputs: list[str] = []
    for target in targets:
        assoc_df = calculate_property_association(
            data,
            target=target,
            features=features,
            datasets=config["datasets"],
            min_n=min_n,
        )
        table_path = layout["tables"] / f"property_association_{target['stem']}.csv"
        assoc_df.to_csv(table_path, index=False)
        outputs.append(str(table_path))
        outputs.extend(
            (
                plot_corrplot_heatmap
                if args.plot_style == "corrplot"
                else plot_bubble_heatmap
            )(
                assoc_df,
                target=target,
                features=features,
                datasets=config["datasets"],
                out_prefix=layout["figures"] / f"property_association_{target['stem']}",
                formats=formats,
                dpi=dpi,
                rho_limit=rho_limit,
                q_cutoff=q_cutoff,
                panel_label=config.get("panel_label"),
                hide_significance_stars=args.hide_significance_stars,
                font_size=args.font_size,
                figure_width_mm=args.figure_width_mm,
                figure_height_mm=args.figure_height_mm,
                pad_inches=args.pad_inches,
                **(
                    {
                        "min_abs_rho_circle": args.min_abs_rho_circle,
                        "show_rho_values": args.show_rho_values,
                        "corrplot_min_bubble_size": args.corrplot_min_bubble_size,
                        "corrplot_max_bubble_size": args.corrplot_max_bubble_size,
                        "corrplot_orientation": args.corrplot_orientation,
                        "rho_text_color": args.rho_text_color,
                    }
                    if args.plot_style == "corrplot"
                    else {}
                ),
            )
        )

    source_summary_path = layout["tables"] / "source_summary.csv"
    pd.DataFrame(source_summary).to_csv(source_summary_path, index=False)
    outputs.append(str(source_summary_path))
    run_record = layout["root"] / "run_record.md"
    write_run_record(
        run_record,
        config_path=args.config,
        command=build_command(),
        config={**config, "output_dir": str(out_dir)},
        targets=targets,
        features=features,
        outputs=outputs,
        source_summary=source_summary,
        plot_style=args.plot_style,
        min_abs_rho_circle=args.min_abs_rho_circle,
        show_rho_values=args.show_rho_values,
        hide_significance_stars=args.hide_significance_stars,
        font_name=font_name,
        font_path=args.font_path,
        font_size=args.font_size,
        figure_width_mm=args.figure_width_mm,
        figure_height_mm=args.figure_height_mm,
        pad_inches=args.pad_inches,
        corrplot_min_bubble_size=args.corrplot_min_bubble_size,
        corrplot_max_bubble_size=args.corrplot_max_bubble_size,
        corrplot_orientation=args.corrplot_orientation,
        rho_text_color=args.rho_text_color,
        corrplot_negative_color=CORRPLOT_NEGATIVE_COLOR,
        corrplot_positive_color=CORRPLOT_POSITIVE_COLOR,
    )
    print(json.dumps({"output_dir": str(out_dir), "outputs": outputs, "run_record": str(run_record)}, indent=2))


if __name__ == "__main__":
    main()
