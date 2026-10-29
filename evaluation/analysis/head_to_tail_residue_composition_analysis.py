#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import wasserstein_distance


TEST2_ROOT = Path(__file__).resolve().parents[2]
SSD_ROOT = Path('assets')

TRAIN_CSV = TEST2_ROOT / "data" / "restored_train.csv"
VALIDATION_CSV = TEST2_ROOT / "data" / "restored_validation.csv"
MODEL_PARSED_CSV = (
    SSD_ROOT
    / "test2_eval"
    / "property_annotation_runs"
    / "20260423_115024_lora_cyclic_n30000_gpu0"
    / "outputs"
    / "head_to_tail_n30000.with_properties.with_smiles2peptide.csv"
)
DEFAULT_OUTPUT_ROOT = SSD_ROOT / "test2_eval" / "head_to_tail_residue_composition_20260414"

TARGET_CYCLIZATION = "head_to_tail"
PLOT_SAMPLE_SEED = 42
FONT_FAMILY = "Arial"
PALETTE = {
    "Train": "#4C78A8",
    "Validation": "#9E9E9E",
    "Our model": "#F58518",
}
SUMMARY_COLUMNS = [
    "dataset_name",
    "N_total",
    "N_valid",
    "overall_nonnatural_fraction",
    "median_nonnatural_fraction",
    "iqr_nonnatural_fraction",
    "fraction_zero_nonnatural",
    "fraction_gt_0_5_nonnatural",
    "wasserstein_to_validation",
]
RECORD_COLUMNS = [
    "dataset_name",
    "sequence_id",
    "sequence",
    "is_valid",
    "length",
    "natural_count",
    "nonnatural_count",
    "nonnatural_fraction",
    "natural_fraction",
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    input_path: Path
    source_kind: str
    parsed_cache_path: Path | None = None


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_name="Train",
        input_path=TRAIN_CSV,
        source_kind="raw_smiles",
        parsed_cache_path=DEFAULT_OUTPUT_ROOT / "cache" / "train_head_to_tail_smiles2peptide.csv",
    ),
    DatasetSpec(
        dataset_name="Validation",
        input_path=VALIDATION_CSV,
        source_kind="raw_smiles",
        parsed_cache_path=DEFAULT_OUTPUT_ROOT / "cache" / "validation_head_to_tail_smiles2peptide.csv",
    ),
    DatasetSpec(
        dataset_name="Our model",
        input_path=MODEL_PARSED_CSV,
        source_kind="parsed_sequences",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze natural vs non-natural residue composition for head-to-tail cyclic peptides.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--validation-csv", type=Path, default=VALIDATION_CSV)
    parser.add_argument("--model-parsed-csv", type=Path, default=MODEL_PARSED_CSV)
    parser.add_argument("--smiles2peptide-root", type=Path, default=TEST2_ROOT / "third_party" / "SMILES2PEPTIDE")
    parser.add_argument("--plot-seed", type=int, default=PLOT_SAMPLE_SEED)
    parser.add_argument("--force-reparse", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def build_dataset_specs(args: argparse.Namespace) -> tuple[DatasetSpec, ...]:
    return (
        DatasetSpec(
            dataset_name="Train",
            input_path=args.train_csv,
            source_kind="raw_smiles",
            parsed_cache_path=args.output_root / "cache" / "train_head_to_tail_smiles2peptide.csv",
        ),
        DatasetSpec(
            dataset_name="Validation",
            input_path=args.validation_csv,
            source_kind="raw_smiles",
            parsed_cache_path=args.output_root / "cache" / "validation_head_to_tail_smiles2peptide.csv",
        ),
        DatasetSpec(
            dataset_name="Our model",
            input_path=args.model_parsed_csv,
            source_kind="parsed_sequences",
        ),
    )


def load_smiles2peptide_batch_module(test2_root: Path = TEST2_ROOT) -> Any:
    script_path = test2_root / "evaluation" / "analysis" / "smiles2peptide_batch.py"
    spec = importlib.util.spec_from_file_location("smiles2peptide_batch", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load SMILES2PEPTIDE batch module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_smiles2peptide_root(batch_module: Any, project_root: Path) -> None:
    if not project_root.exists():
        raise FileNotFoundError(f"SMILES2PEPTIDE project root not found: {project_root}")
    original_loader = batch_module.load_peptide_analyzer
    batch_module.load_peptide_analyzer = lambda: original_loader(project_root)


def detect_column(dataframe: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    raise ValueError(f"Could not find {label} column in {list(dataframe.columns)}")


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def build_head_to_tail_subset(dataframe: pd.DataFrame) -> pd.DataFrame:
    smiles_col = detect_column(dataframe, ("SMILES", "Smiles", "smiles"), "SMILES")
    cyclization_col = detect_column(dataframe, ("Cyclization", "cyclization"), "Cyclization")
    subset = dataframe[dataframe[cyclization_col] == TARGET_CYCLIZATION].copy()
    subset = subset[subset[smiles_col].notna()].copy()
    subset.rename(columns={smiles_col: "SMILES"}, inplace=True)
    subset.reset_index(drop=True, inplace=True)
    subset["sequence_id"] = [f"raw_{idx:07d}" for idx in range(len(subset))]
    return subset[["sequence_id", "SMILES"]]


def build_composition_records_from_parsed(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
) -> pd.DataFrame:
    required = {
        "three_letter_sequence",
        "parse_status",
        "peptide_length",
        "n_natural_resolved",
        "n_non_natural_resolved",
        "frac_natural_resolved",
        "frac_non_natural_resolved",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing parsed sequence columns for {dataset_name}: {sorted(missing)}")

    sequence_id_col = None
    for candidate in ("mol_id", "sequence_id", "id", "ID"):
        if candidate in dataframe.columns:
            sequence_id_col = candidate
            break

    records = pd.DataFrame(
        {
            "dataset_name": dataset_name,
            "sequence_id": (
                dataframe[sequence_id_col].astype(str)
                if sequence_id_col is not None
                else [f"{dataset_name}_{idx:07d}" for idx in range(len(dataframe))]
            ),
            "sequence": dataframe["three_letter_sequence"].fillna("").astype(str),
            "is_valid": dataframe["parse_status"].fillna("").astype(str).ne("failed")
            & pd.to_numeric(dataframe["peptide_length"], errors="coerce").notna(),
            "length": pd.to_numeric(dataframe["peptide_length"], errors="coerce"),
            "natural_count": pd.to_numeric(dataframe["n_natural_resolved"], errors="coerce"),
            "nonnatural_count": pd.to_numeric(dataframe["n_non_natural_resolved"], errors="coerce"),
            "nonnatural_fraction": pd.to_numeric(dataframe["frac_non_natural_resolved"], errors="coerce"),
            "natural_fraction": pd.to_numeric(dataframe["frac_natural_resolved"], errors="coerce"),
        }
    )
    return records[RECORD_COLUMNS]


def parse_raw_dataset_to_cache(
    spec: DatasetSpec,
    *,
    batch_module: Any,
    force_reparse: bool,
) -> Path:
    if spec.parsed_cache_path is None:
        raise ValueError(f"{spec.dataset_name} does not define a parsed cache path")
    cache_path = spec.parsed_cache_path
    if cache_path.exists() and not force_reparse:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    source_df = pd.read_csv(spec.input_path)
    subset_df = build_head_to_tail_subset(source_df)
    analyzer = batch_module.load_peptide_analyzer()
    parsed_df, _, summary = batch_module.convert_dataframe(
        subset_df,
        smiles_column="SMILES",
        id_column="sequence_id",
        analyzer=analyzer,
        max_rows=None,
    )
    parsed_df.to_csv(cache_path, index=False)
    summary_path = cache_path.with_suffix(".summary.json")
    summary_payload = {
        "dataset_name": spec.dataset_name,
        "source_csv": str(spec.input_path),
        "subset_rows": int(len(subset_df)),
        **summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return cache_path


def load_dataset_records(
    spec: DatasetSpec,
    *,
    batch_module: Any,
    force_reparse: bool,
) -> pd.DataFrame:
    if spec.source_kind == "parsed_sequences":
        parsed_path = spec.input_path
    elif spec.source_kind == "raw_smiles":
        parsed_path = parse_raw_dataset_to_cache(spec, batch_module=batch_module, force_reparse=force_reparse)
    else:
        raise ValueError(f"Unsupported source_kind={spec.source_kind!r} for {spec.dataset_name}")

    parsed_df = pd.read_csv(parsed_path)
    return build_composition_records_from_parsed(parsed_df, dataset_name=spec.dataset_name)


def summarize_dataset(records: pd.DataFrame, validation_fractions: np.ndarray) -> dict[str, Any]:
    dataset_name = str(records["dataset_name"].iloc[0])
    valid = records[records["is_valid"]].copy()
    valid_fractions = valid["nonnatural_fraction"].dropna().to_numpy(dtype=float)
    total_natural = float(valid["natural_count"].fillna(0).sum())
    total_nonnatural = float(valid["nonnatural_count"].fillna(0).sum())
    total_resolved = total_natural + total_nonnatural

    iqr = None
    if len(valid_fractions):
        q1 = float(np.quantile(valid_fractions, 0.25))
        q3 = float(np.quantile(valid_fractions, 0.75))
        iqr = q3 - q1

    wasserstein = 0.0 if dataset_name == "Validation" else None
    if dataset_name != "Validation" and len(valid_fractions) and len(validation_fractions):
        wasserstein = float(wasserstein_distance(valid_fractions, validation_fractions))

    return {
        "dataset_name": dataset_name,
        "N_total": int(len(records)),
        "N_valid": int(len(valid)),
        "overall_nonnatural_fraction": (float(total_nonnatural / total_resolved) if total_resolved else None),
        "median_nonnatural_fraction": (float(np.median(valid_fractions)) if len(valid_fractions) else None),
        "iqr_nonnatural_fraction": iqr,
        "fraction_zero_nonnatural": (float(np.mean(valid_fractions == 0.0)) if len(valid_fractions) else None),
        "fraction_gt_0_5_nonnatural": (float(np.mean(valid_fractions > 0.5)) if len(valid_fractions) else None),
        "wasserstein_to_validation": wasserstein,
    }


def build_plotting_table(records: pd.DataFrame, *, plot_seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    valid = records[records["is_valid"]].copy()
    counts = valid.groupby("dataset_name").size().to_dict()
    model_n = int(counts.get("Our model", 0))
    if model_n <= 0:
        raise ValueError("Our model has no valid sequences; cannot build plotting table.")

    sampled_frames: list[pd.DataFrame] = []
    sampling_info: dict[str, int] = {}
    for dataset_name, group in valid.groupby("dataset_name", sort=False):
        if dataset_name in {"Train", "Validation"} and len(group) > model_n:
            sampled = group.sample(n=model_n, replace=False, random_state=plot_seed).copy()
        else:
            sampled = group.copy()
        sampled_frames.append(sampled)
        sampling_info[dataset_name] = int(len(sampled))
    plotting = pd.concat(sampled_frames, ignore_index=True)
    return plotting, sampling_info


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "figure.dpi": 300,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def draw_panel_a(ax: plt.Axes, plotting: pd.DataFrame) -> None:
    order = ["Train", "Validation", "Our model"]
    data = [
        plotting.loc[plotting["dataset_name"] == dataset_name, "nonnatural_fraction"].dropna().to_numpy(dtype=float)
        for dataset_name in order
    ]
    positions = np.arange(1, len(order) + 1)
    violin = ax.violinplot(data, positions=positions, widths=0.8, showmeans=False, showextrema=False, showmedians=False)
    for body, dataset_name in zip(violin["bodies"], order):
        body.set_facecolor(PALETTE[dataset_name])
        body.set_edgecolor("black")
        body.set_alpha(0.45)
        body.set_linewidth(0.8)

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        boxprops={"edgecolor": "black", "linewidth": 0.8},
    )
    for patch, dataset_name in zip(box["boxes"], order):
        patch.set_facecolor(PALETTE[dataset_name])
        patch.set_alpha(0.9)

    ax.set_xticks(positions, order)
    ax.set_ylabel("Per-sequence non-natural fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_b(ax: plt.Axes, plotting: pd.DataFrame) -> None:
    rows = []
    for dataset_name in ["Train", "Validation", "Our model"]:
        subset = plotting[plotting["dataset_name"] == dataset_name]
        total_natural = float(subset["natural_count"].fillna(0).sum())
        total_nonnatural = float(subset["nonnatural_count"].fillna(0).sum())
        total = total_natural + total_nonnatural
        rows.append(
            {
                "dataset_name": dataset_name,
                "natural_fraction": (total_natural / total if total else 0.0),
                "nonnatural_fraction": (total_nonnatural / total if total else 0.0),
            }
        )
    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    ax.bar(x, frame["natural_fraction"], color="#C8C8C8", edgecolor="black", linewidth=0.8, label="Natural")
    ax.bar(
        x,
        frame["nonnatural_fraction"],
        bottom=frame["natural_fraction"],
        color="#D55E00",
        edgecolor="black",
        linewidth=0.8,
        label="Non-natural",
    )
    ax.set_xticks(x, frame["dataset_name"].tolist())
    ax.set_ylabel("Overall residue fraction")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_c(ax: plt.Axes, plotting: pd.DataFrame) -> None:
    subset = plotting[plotting["dataset_name"] == "Our model"].copy()
    x = pd.to_numeric(subset["length"], errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(subset["nonnatural_fraction"], errors="coerce").dropna().to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        raise ValueError("Our model has no valid length/fraction pairs for hexbin plotting.")
    hb = ax.hexbin(
        x,
        y,
        gridsize=25,
        cmap="Blues",
        mincnt=1,
        linewidths=0.0,
    )
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Per-sequence non-natural fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    colorbar = plt.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
    colorbar.set_label("Count")


def build_combined_figure(plotting: pd.DataFrame, output_root: Path) -> Path:
    configure_matplotlib()
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    draw_panel_a(axes[0], plotting)
    draw_panel_b(axes[1], plotting)
    draw_panel_c(axes[2], plotting)

    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    add_panel_label(axes[2], "C")

    fig.tight_layout(w_pad=2.0)
    stem = figure_dir / "head_to_tail_residue_composition"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".pdf")


def write_plot_metadata(
    output_root: Path,
    sampling_info: dict[str, int],
    summary_df: pd.DataFrame,
    *,
    plot_seed: int,
) -> None:
    payload = {
        "plot_seed": plot_seed,
        "plot_sampling_counts": sampling_info,
        "summary_preview": summary_df.to_dict(orient="records"),
    }
    (output_root / "plot_metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    batch_module = load_smiles2peptide_batch_module()
    set_smiles2peptide_root(batch_module, args.smiles2peptide_root)
    record_frames = []
    dataset_specs = build_dataset_specs(args)
    for spec in dataset_specs:
        print(f"Loading dataset: {spec.dataset_name} ({spec.source_kind})")
        record_frames.append(load_dataset_records(spec, batch_module=batch_module, force_reparse=args.force_reparse))
    records = pd.concat(record_frames, ignore_index=True)
    records.to_csv(args.output_root / "composition_records.csv", index=False)

    validation_fractions = (
        records[(records["dataset_name"] == "Validation") & (records["is_valid"])]["nonnatural_fraction"]
        .dropna()
        .to_numpy(dtype=float)
    )
    summary_rows = [
        summarize_dataset(records[records["dataset_name"] == spec.dataset_name].copy(), validation_fractions)
        for spec in dataset_specs
    ]
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(args.output_root / "composition_summary.csv", index=False)

    plotting, sampling_info = build_plotting_table(records, plot_seed=args.plot_seed)
    plotting.to_csv(args.output_root / "plotting_records.csv", index=False)
    build_combined_figure(plotting, args.output_root)
    write_plot_metadata(args.output_root, sampling_info, summary_df, plot_seed=args.plot_seed)

    print(args.output_root / "composition_summary.csv")
    print(args.output_root / "figures" / "head_to_tail_residue_composition.pdf")


if __name__ == "__main__":
    main()
