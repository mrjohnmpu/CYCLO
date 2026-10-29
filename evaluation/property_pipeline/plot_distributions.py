from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


MM_PER_INCH = 25.4
NATURE_SINGLE_COLUMN_WIDTH = 90.0 / MM_PER_INCH
NATURE_DOUBLE_COLUMN_WIDTH = 180.0 / MM_PER_INCH
NATURE_MAX_HEIGHT = 170.0 / MM_PER_INCH
NATURE_COMPOSITE_SIZE = (NATURE_DOUBLE_COLUMN_WIDTH, 160.0 / MM_PER_INCH)
NATURE_PANEL_SIZE = (NATURE_DOUBLE_COLUMN_WIDTH, 70.0 / MM_PER_INCH)
PLOT_DEFAULT_SAMPLE_SIZE = 5000
PLOT_DEFAULT_SAMPLE_SEED = 42
PUBLICATION_FONT_NAME = "Arial"
DEFAULT_DISCRETE_HISTOGRAM_DATASET_IDS = ("Reference", "CycGPT_LoRA")
FOCUSED_HISTOGRAM_PROPERTIES = {"Length", "HBD", "HBA"}
ACCESSIBLE_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#56B4E9",
    "#CC79A7",
    "#D55E00",
    "#F0E442",
    "#332288",
    "#88CCEE",
]

PUBLICATION_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        PUBLICATION_FONT_NAME,
        "Helvetica",
        "Nimbus Sans",
        "Liberation Sans",
        "DejaVu Sans",
    ],
    "font.size": 6,
    "axes.labelsize": 6,
    "axes.titlesize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "axes.unicode_minus": False,
}

_FONT_WARNING_EMITTED = False

CONTINUOUS_PROPERTIES = {
    "MW": {"column": "MW", "xlabel": "Exact Molecular Weight (Da)", "stem": "mw"},
    "LogP": {"column": "LogP", "xlabel": "Wildman-Crippen LogP", "stem": "logp"},
    "TPSA": {"column": "TPSA", "xlabel": "TPSA (A^2)", "stem": "tpsa"},
    "Half-life": {"column": "half_life_value", "xlabel": "Half-life", "stem": "half_life"},
    "PAMPA": {"column": "PAMPA_value", "xlabel": "PAMPA", "stem": "pampa"},
    "Caco-2": {"column": "Caco-2_value", "xlabel": "Caco-2", "stem": "caco2"},
}

DISCRETE_PROPERTIES = {
    "HBD": {"column": "HBD", "xlabel": "HBD", "stem": "hbd"},
    "HBA": {"column": "HBA", "xlabel": "HBA", "stem": "hba"},
    "RotBonds": {"column": "RotBonds", "xlabel": "Rotatable Bonds", "stem": "rotbonds"},
    "Length": {"column": "Length", "xlabel": "Residue Count", "stem": "length"},
}

ALL_PROPERTIES = {
    **CONTINUOUS_PROPERTIES,
    **DISCRETE_PROPERTIES,
}

CLASSIFICATION_PROPERTIES = {
    "Non-hemolytic": {"stem": "hemolysis"},
    "Non-fouling": {"stem": "non_fouling"},
    "Non-toxic": {"stem": "toxicity"},
}

COMPOSITE_LAYOUT = [
    [("Length", "hist"), ("HBD", "hist"), ("HBA", "hist")],
    [("MW", "violin"), ("TPSA", "violin"), ("RotBonds", "violin")],
    [("LogP", "half_violin"), ("PAMPA", "half_violin"), ("Half-life", "half_violin")],
]


def _apply_publication_style() -> None:
    global _FONT_WARNING_EMITTED
    plt.rcParams.update(PUBLICATION_RCPARAMS)
    if _FONT_WARNING_EMITTED:
        return
    try:
        font_manager.findfont(
            font_manager.FontProperties(family=PUBLICATION_FONT_NAME),
            fallback_to_default=False,
        )
    except ValueError:
        warnings.warn(
            f"{PUBLICATION_FONT_NAME} is not available; falling back to the closest sans-serif font.",
            RuntimeWarning,
            stacklevel=2,
        )
    _FONT_WARNING_EMITTED = True


def build_color_map(dataset_order: Iterable[str], dataset_label_map: dict[str, str]) -> dict[str, str]:
    ordered = list(dataset_order)
    color_map: dict[str, str] = {}
    non_reference = [dataset_id for dataset_id in ordered if dataset_id != "Reference"]
    color_map["Reference"] = "#B3B3B3"
    for idx, dataset_id in enumerate(non_reference):
        color_map[dataset_id] = ACCESSIBLE_PALETTE[idx % len(ACCESSIBLE_PALETTE)]
    for dataset_id in ordered:
        color_map.setdefault(dataset_id, ACCESSIBLE_PALETTE[0])
    return color_map


def sample_plotting_table(
    merged_table: pd.DataFrame,
    *,
    dataset_order: Iterable[str],
    sample_size: int | None = PLOT_DEFAULT_SAMPLE_SIZE,
    sample_seed: int = PLOT_DEFAULT_SAMPLE_SEED,
) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0:
        return merged_table.copy()

    rng = np.random.default_rng(sample_seed)
    sampled_frames: list[pd.DataFrame] = []
    for dataset_id in dataset_order:
        subset = merged_table[merged_table["dataset_id"] == dataset_id]
        if subset.empty:
            continue
        if len(subset) <= sample_size:
            sampled = subset.copy()
        else:
            sampled = subset.sample(
                n=sample_size,
                replace=False,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
        sampled_frames.append(sampled)
    if not sampled_frames:
        return merged_table.iloc[0:0].copy()
    return pd.concat(sampled_frames, ignore_index=True)


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    preserve_canvas: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not preserve_canvas:
        fig.tight_layout()
    save_kwargs = {} if preserve_canvas else {"bbox_inches": "tight"}
    fig.savefig(output_dir / f"{stem}.pdf", **save_kwargs)
    fig.savefig(output_dir / f"{stem}.svg", **save_kwargs)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, **save_kwargs)
    plt.close(fig)


def _get_values(dataframe: pd.DataFrame, dataset_id: str, column: str) -> np.ndarray:
    subset = dataframe[dataframe["dataset_id"] == dataset_id]
    return pd.to_numeric(subset[column], errors="coerce").dropna().to_numpy(dtype=float)


def _build_dataset_series(
    dataframe: pd.DataFrame,
    dataset_order: Iterable[str],
    dataset_label_map: dict[str, str],
    color_map: dict[str, str],
    column: str,
) -> tuple[list[np.ndarray], list[str], list[str], list[str]]:
    data: list[np.ndarray] = []
    dataset_ids: list[str] = []
    labels: list[str] = []
    colors: list[str] = []
    for dataset_id in dataset_order:
        values = _get_values(dataframe, dataset_id, column)
        if values.size == 0:
            continue
        data.append(values)
        dataset_ids.append(dataset_id)
        labels.append(dataset_label_map[dataset_id])
        colors.append(color_map[dataset_id])
    return data, dataset_ids, labels, colors


def _build_legend_handles(
    dataset_order: Iterable[str],
    dataset_label_map: dict[str, str],
    color_map: dict[str, str],
) -> list[Patch]:
    handles = []
    for dataset_id in dataset_order:
        handles.append(
            Patch(
                facecolor=color_map[dataset_id],
                edgecolor="black",
                alpha=0.9 if dataset_id == "Reference" else 0.75,
                label=dataset_label_map[dataset_id],
            )
        )
    return handles


def _choose_integer_ticks(min_value: int, max_value: int, *, max_ticks: int = 6) -> np.ndarray:
    if min_value >= max_value:
        return np.array([min_value], dtype=int)
    span = max_value - min_value
    if span <= max_ticks - 1:
        return np.arange(min_value, max_value + 1, dtype=int)
    step = max(1, math.ceil(span / (max_ticks - 1)))
    ticks = list(range(min_value, max_value + 1, step))
    if ticks[-1] != max_value:
        ticks.append(max_value)
    return np.array(sorted(set(ticks)), dtype=int)


def _style_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=5, length=2)
    ax.grid(False)


def _resolve_histogram_dataset_order(
    display_name: str,
    dataset_order: Iterable[str],
    focused_dataset_ids: Iterable[str] | None,
) -> list[str]:
    ordered = list(dataset_order)
    if display_name not in FOCUSED_HISTOGRAM_PROPERTIES or not focused_dataset_ids:
        return ordered
    selected = [dataset_id for dataset_id in ordered if dataset_id in set(focused_dataset_ids)]
    return selected if selected else ordered


def _draw_hist_panel(
    ax: plt.Axes,
    dataframe: pd.DataFrame,
    *,
    dataset_order: list[str],
    dataset_label_map: dict[str, str],
    color_map: dict[str, str],
    display_name: str,
    config: dict[str, str],
    show_legend: bool,
    focused_dataset_ids: Iterable[str] | None = None,
) -> None:
    active_dataset_order = _resolve_histogram_dataset_order(
        display_name,
        dataset_order,
        focused_dataset_ids,
    )
    column = config["column"]
    values_by_dataset = {
        dataset_id: _get_values(dataframe, dataset_id, column).astype(int)
        for dataset_id in active_dataset_order
    }
    populated_values = [values for values in values_by_dataset.values() if values.size > 0]
    if not populated_values:
        return
    all_values = np.concatenate(populated_values)
    x_values = np.arange(int(all_values.min()), int(all_values.max()) + 1)
    width = 0.72 / max(len(active_dataset_order), 1)

    for idx, dataset_id in enumerate(active_dataset_order):
        values = values_by_dataset[dataset_id]
        if values.size == 0:
            continue
        counts = np.array([(values == x).sum() for x in x_values], dtype=float)
        positions = x_values + (idx - (len(active_dataset_order) - 1) / 2.0) * width
        alpha = 0.9 if dataset_id == "Reference" else 0.75
        ax.bar(
            positions,
            counts,
            width=width,
            color=color_map[dataset_id],
            alpha=alpha,
            label=dataset_label_map[dataset_id],
        )

    ax.set_xlabel(config["xlabel"])
    ax.set_ylabel("Frequency")
    tick_values = _choose_integer_ticks(int(x_values.min()), int(x_values.max()), max_ticks=4)
    ax.set_xticks(tick_values)
    ax.set_xticklabels([str(value) for value in tick_values])
    if show_legend:
        ax.legend(frameon=False, fontsize=5)
    _style_axis(ax)


def _draw_violin_box_panel(
    ax: plt.Axes,
    dataframe: pd.DataFrame,
    *,
    dataset_order: list[str],
    dataset_label_map: dict[str, str],
    color_map: dict[str, str],
    display_name: str,
    config: dict[str, str],
) -> None:
    data, _, labels, colors = _build_dataset_series(
        dataframe,
        dataset_order,
        dataset_label_map,
        color_map,
        config["column"],
    )
    violin_parts = ax.violinplot(
        data,
        positions=np.arange(len(data)),
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violin_parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.5)
    box = ax.boxplot(
        data,
        positions=np.arange(len(data)),
        widths=0.18,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markersize=1.2, markerfacecolor="black", markeredgecolor="black"),
    )
    for patch in box["boxes"]:
        patch.set_facecolor("white")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(config["xlabel"])
    _style_axis(ax)


def _draw_half_violin_box_scatter_panel(
    ax: plt.Axes,
    dataframe: pd.DataFrame,
    *,
    dataset_order: list[str],
    dataset_label_map: dict[str, str],
    color_map: dict[str, str],
    display_name: str,
    config: dict[str, str],
    rng: np.random.Generator,
) -> None:
    data, _, labels, colors = _build_dataset_series(
        dataframe,
        dataset_order,
        dataset_label_map,
        color_map,
        config["column"],
    )
    for idx, values in enumerate(data):
        violin = ax.violinplot(
            values,
            positions=[idx],
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.clip(vertices[:, 0], idx, np.inf)
            body.set_facecolor(colors[idx])
            body.set_edgecolor("black")
            body.set_alpha(0.65)

    ax.boxplot(
        data,
        positions=np.arange(len(data)),
        widths=0.10,
        patch_artist=True,
        boxprops=dict(facecolor="white", color="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", color="black", markersize=1),
        showfliers=True,
    )

    for idx, values in enumerate(data):
        jittered_x = rng.normal(loc=idx - 0.10, scale=0.04, size=len(values))
        ax.scatter(jittered_x, values, color=colors[idx], alpha=0.25, s=2.0)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(config["xlabel"])
    _style_axis(ax)


def _annotate_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.20,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        clip_on=False,
    )


def plot_kde_panels(
    merged_table: pd.DataFrame,
    *,
    dataset_order: list[str],
    dataset_label_map: dict[str, str],
    output_dir: Path,
) -> None:
    _apply_publication_style()
    color_map = build_color_map(dataset_order, dataset_label_map)
    for display_name, config in CONTINUOUS_PROPERTIES.items():
        column = config["column"]
        fig, ax = plt.subplots(figsize=NATURE_PANEL_SIZE)
        reference_values = _get_values(merged_table, "Reference", column)
        if reference_values.size >= 2:
            grid = np.linspace(reference_values.min(), reference_values.max(), 256)
            density = gaussian_kde(reference_values)(grid)
            ax.fill_between(
                grid,
                density,
                color=color_map["Reference"],
                alpha=0.35,
                label=dataset_label_map["Reference"],
            )

        for dataset_id in dataset_order:
            if dataset_id == "Reference":
                continue
            values = _get_values(merged_table, dataset_id, column)
            if values.size < 2:
                continue
            lower = min(reference_values.min() if reference_values.size else values.min(), values.min())
            upper = max(reference_values.max() if reference_values.size else values.max(), values.max())
            grid = np.linspace(lower, upper, 256)
            density = gaussian_kde(values)(grid)
            ax.plot(grid, density, color=color_map[dataset_id], linewidth=2, label=dataset_label_map[dataset_id])

        ax.set_xlabel(config["xlabel"])
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=5)
        _style_axis(ax)
        _save_figure(fig, output_dir, f"kde_{config['stem']}")


def plot_violin_box_panels(
    merged_table: pd.DataFrame,
    *,
    dataset_order: list[str],
    dataset_label_map: dict[str, str],
    output_dir: Path,
) -> None:
    _apply_publication_style()
    color_map = build_color_map(dataset_order, dataset_label_map)
    violin_properties = ["MW", "LogP", "TPSA", "Half-life", "PAMPA", "Caco-2"]
    for display_name in violin_properties:
        config = CONTINUOUS_PROPERTIES[display_name]
        data, _, _, _ = _build_dataset_series(
            merged_table,
            dataset_order,
            dataset_label_map,
            color_map,
            config["column"],
        )
        if not data:
            continue
        fig, ax = plt.subplots(figsize=NATURE_PANEL_SIZE)
        _draw_violin_box_panel(
            ax,
            merged_table,
            dataset_order=dataset_order,
            dataset_label_map=dataset_label_map,
            color_map=color_map,
            display_name=display_name,
            config=config,
        )
        _save_figure(fig, output_dir, f"violin_box_{config['stem']}")


def plot_half_violin_box_scatter_panels(
    merged_table: pd.DataFrame,
    *,
    dataset_order: list[str],
    dataset_label_map: dict[str, str],
    output_dir: Path,
) -> None:
    _apply_publication_style()
    color_map = build_color_map(dataset_order, dataset_label_map)
    rng = np.random.default_rng(PLOT_DEFAULT_SAMPLE_SEED)
    half_violin_properties = ["LogP", "PAMPA", "Half-life", "Caco-2"]
    for display_name in half_violin_properties:
        config = CONTINUOUS_PROPERTIES[display_name]
        data, _, _, _ = _build_dataset_series(
            merged_table,
            dataset_order,
            dataset_label_map,
            color_map,
            config["column"],
        )
        if not data:
            continue
        fig, ax = plt.subplots(figsize=NATURE_PANEL_SIZE)
        _draw_half_violin_box_scatter_panel(
            ax,
            merged_table,
            dataset_order=dataset_order,
            dataset_label_map=dataset_label_map,
            color_map=color_map,
            display_name=display_name,
            config=config,
            rng=rng,
        )
        _save_figure(fig, output_dir, f"half_violin_box_scatter_{config['stem']}")


def plot_discrete_histograms(
    merged_table: pd.DataFrame,
    *,
    dataset_order: list[str],
    dataset_label_map: dict[str, str],
    output_dir: Path,
    focused_dataset_ids: Iterable[str] | None = DEFAULT_DISCRETE_HISTOGRAM_DATASET_IDS,
) -> None:
    _apply_publication_style()
    color_map = build_color_map(dataset_order, dataset_label_map)
    for display_name, config in DISCRETE_PROPERTIES.items():
        histogram_dataset_order = _resolve_histogram_dataset_order(
            display_name,
            dataset_order,
            focused_dataset_ids,
        )
        values_by_dataset = {
            dataset_id: _get_values(merged_table, dataset_id, config["column"]).astype(int)
            for dataset_id in histogram_dataset_order
        }
        if not any(values.size > 0 for values in values_by_dataset.values()):
            continue
        fig, ax = plt.subplots(figsize=NATURE_PANEL_SIZE)
        _draw_hist_panel(
            ax,
            merged_table,
            dataset_order=dataset_order,
            dataset_label_map=dataset_label_map,
            color_map=color_map,
            display_name=display_name,
            config=config,
            show_legend=True,
            focused_dataset_ids=focused_dataset_ids,
        )
        _save_figure(fig, output_dir, f"hist_{config['stem']}")


def plot_classification_ratio_bars(
    ratio_summary: pd.DataFrame,
    *,
    dataset_order: list[str],
    dataset_label_map: dict[str, str],
    output_dir: Path,
) -> None:
    _apply_publication_style()
    color_map = build_color_map(dataset_order, dataset_label_map)
    fig, axes = plt.subplots(1, 3, figsize=(NATURE_DOUBLE_COLUMN_WIDTH, 60.0 / MM_PER_INCH), sharey=True)
    for ax, (display_name, _) in zip(axes, CLASSIFICATION_PROPERTIES.items()):
        subset = ratio_summary.set_index("dataset_id")
        x = np.arange(len(dataset_order))
        y = [
            float(subset.loc[dataset_id, display_name]) if dataset_id in subset.index else np.nan
            for dataset_id in dataset_order
        ]
        colors = [color_map[dataset_id] for dataset_id in dataset_order]
        ax.bar(x, y, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [dataset_label_map[dataset_id] for dataset_id in dataset_order],
            rotation=30,
            ha="right",
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Positive Ratio")
        ax.set_xlabel(display_name)
        _style_axis(ax)
    _save_figure(fig, output_dir, "bar_classification_ratios")


def plot_composite_property_overview(
    merged_table: pd.DataFrame,
    *,
    dataset_order: list[str],
    dataset_label_map: dict[str, str],
    output_dir: Path,
    focused_histogram_dataset_ids: Iterable[str] | None = DEFAULT_DISCRETE_HISTOGRAM_DATASET_IDS,
) -> None:
    _apply_publication_style()
    color_map = build_color_map(dataset_order, dataset_label_map)
    rng = np.random.default_rng(PLOT_DEFAULT_SAMPLE_SEED)
    fig, axes = plt.subplots(3, 3, figsize=NATURE_COMPOSITE_SIZE)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.15, wspace=0.42, hspace=0.52)

    panel_index = 0
    for row_idx, row in enumerate(COMPOSITE_LAYOUT):
        for col_idx, (display_name, panel_type) in enumerate(row):
            ax = axes[row_idx, col_idx]
            config = ALL_PROPERTIES[display_name]
            if panel_type == "hist":
                _draw_hist_panel(
                    ax,
                    merged_table,
                    dataset_order=dataset_order,
                    dataset_label_map=dataset_label_map,
                    color_map=color_map,
                    display_name=display_name,
                    config=config,
                    show_legend=False,
                    focused_dataset_ids=focused_histogram_dataset_ids,
                )
            elif panel_type == "violin":
                _draw_violin_box_panel(
                    ax,
                    merged_table,
                    dataset_order=dataset_order,
                    dataset_label_map=dataset_label_map,
                    color_map=color_map,
                    display_name=display_name,
                    config=config,
                )
            elif panel_type == "half_violin":
                _draw_half_violin_box_scatter_panel(
                    ax,
                    merged_table,
                    dataset_order=dataset_order,
                    dataset_label_map=dataset_label_map,
                    color_map=color_map,
                    display_name=display_name,
                    config=config,
                    rng=rng,
                )
            else:
                raise ValueError(f"Unsupported composite panel type: {panel_type}")

            _annotate_panel_label(ax, f"({chr(ord('A') + panel_index)})")
            panel_index += 1

    handles = _build_legend_handles(dataset_order, dataset_label_map, color_map)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=min(5, len(handles)),
        frameon=False,
        fontsize=5,
    )
    _save_figure(
        fig,
        output_dir,
        "composite_3x3_property_overview",
        preserve_canvas=True,
    )
