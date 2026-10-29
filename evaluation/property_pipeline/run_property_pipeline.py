#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Optional

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from property_pipeline.data_inventory import (
    DEFAULT_MANIFEST_CSV,
    DEFAULT_REFERENCE_CSV,
    DatasetSpec,
    build_inventory_rows,
    build_reference_dataset,
    load_manifest_datasets,
    write_inventory,
)
from property_pipeline.distribution_metrics import (
    compute_positive_ratio_summary,
    compute_wasserstein_summary,
    merge_property_tables,
)
from property_pipeline.mol_cache import build_dataset_cache, write_cache_tables
from property_pipeline.peptiverse_batch import (
    PeptiVerseBatchPredictor,
    run_peptiverse_predictions,
    summarize_peptiverse_predictions,
)
from property_pipeline.plot_distributions import (
    DEFAULT_DISCRETE_HISTOGRAM_DATASET_IDS,
    PLOT_DEFAULT_SAMPLE_SEED,
    PLOT_DEFAULT_SAMPLE_SIZE,
    plot_classification_ratio_bars,
    plot_composite_property_overview,
    plot_discrete_histograms,
    plot_half_violin_box_scatter_panels,
    plot_kde_panels,
    plot_violin_box_panels,
    sample_plotting_table,
)
from property_pipeline.rdkit_properties import LengthEstimator, compute_rdkit_property_table, summarize_rdkit_properties


DEFAULT_OUTPUT_ROOT = Path('results/property_pipeline_20260411')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cyclic peptide property prediction and comparison pipeline.")
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cyclization-type", type=str, default="head_to_tail")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--limit-datasets", type=str, default=None, help="Comma-separated dataset_ids to keep.")
    parser.add_argument("--plot-sample-size", type=int, default=PLOT_DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--plot-sample-seed", type=int, default=PLOT_DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--discrete-histogram-dataset-ids",
        type=str,
        default=",".join(DEFAULT_DISCRETE_HISTOGRAM_DATASET_IDS),
        help="Comma-separated dataset_ids to draw for Length/HBD/HBA histograms.",
    )
    parser.add_argument("--skip-peptiverse", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def ensure_layout(output_root: Path) -> dict[str, Path]:
    layout = {
        "root": output_root,
        "manifest": output_root / "manifest",
        "cache": output_root / "cache",
        "rdkit": output_root / "rdkit",
        "peptiverse": output_root / "peptiverse",
        "metrics": output_root / "metrics",
        "figures_kde": output_root / "figures" / "kde",
        "figures_violin": output_root / "figures" / "violin",
        "figures_hist": output_root / "figures" / "hist",
        "figures_bar": output_root / "figures" / "bar",
        "figures_half_violin": output_root / "figures" / "scatter_box_half_violin",
        "figures_composite": output_root / "figures" / "composite",
        "summary": output_root / "summary",
        "logs": output_root / "logs",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def log(message: str) -> None:
    print(f"[property-pipeline] {message}", flush=True)


def filter_specs(specs: list[DatasetSpec], limit_datasets: Optional[str]) -> list[DatasetSpec]:
    if not limit_datasets:
        return specs
    keep = {value.strip() for value in limit_datasets.split(",") if value.strip()}
    return [spec for spec in specs if spec.dataset_id in keep or spec.dataset_kind == "reference"]


def dataset_label_map(specs: Iterable[DatasetSpec]) -> dict[str, str]:
    return {spec.dataset_id: spec.dataset_label for spec in specs}


def parse_dataset_id_list(value: str) -> list[str] | None:
    if not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def write_markdown_summary(
    summary_path: Path,
    *,
    args: argparse.Namespace,
    inventory_path: Path,
    rdkit_summary_path: Path,
    peptiverse_summary_path: Optional[Path],
    wasserstein_path: Optional[Path],
    ratio_path: Optional[Path],
) -> None:
    lines = [
        "# Property Pipeline Summary",
        "",
        f"- Reference CSV: `{args.reference_csv}`",
        f"- Manifest CSV: `{args.manifest_csv}`",
        f"- Cyclization type filter: `{args.cyclization_type}`",
        f"- Output root: `{args.output_root}`",
        f"- Inventory: `{inventory_path}`",
        f"- RDKit property summary: `{rdkit_summary_path}`",
        f"- Plot sample size: `{args.plot_sample_size}`",
        f"- Plot sample seed: `{args.plot_sample_seed}`",
        f"- Histogram dataset ids: `{args.discrete_histogram_dataset_ids}`",
    ]
    if peptiverse_summary_path is not None:
        lines.append(f"- PeptiVerse summary: `{peptiverse_summary_path}`")
    if wasserstein_path is not None:
        lines.append(f"- Wasserstein summary: `{wasserstein_path}`")
    if ratio_path is not None:
        lines.append(f"- Positive ratio summary: `{ratio_path}`")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    layout = ensure_layout(args.output_root)
    log(f"Output root: {args.output_root}")

    specs = [build_reference_dataset(args.reference_csv), *load_manifest_datasets(args.manifest_csv)]
    specs = filter_specs(specs, args.limit_datasets)
    order = [spec.dataset_id for spec in specs]
    labels = dataset_label_map(specs)
    focused_histogram_dataset_ids = parse_dataset_id_list(args.discrete_histogram_dataset_ids)
    log(f"Datasets: {order}")

    inventory_rows = build_inventory_rows(specs, cyclization_type=args.cyclization_type)
    inventory_path = write_inventory(inventory_rows, layout["manifest"] / "dataset_inventory.csv")
    log(f"Wrote inventory: {inventory_path}")

    log("Building RDKit-valid molecule caches")
    caches = []
    for spec in specs:
        cache = build_dataset_cache(
            spec,
            cyclization_type=args.cyclization_type,
            max_rows=args.max_rows_per_dataset,
        )
        write_cache_tables(cache, layout["cache"])
        caches.append(cache)
        log(
            f"Cache ready for {spec.dataset_id}: total={len(cache.dataframe)} valid={len(cache.valid_dataframe)}"
        )

    log("Loading canonical peptide length estimator")
    length_estimator = LengthEstimator.load()
    rdkit_tables = []
    log("Computing RDKit properties")
    for cache in caches:
        table = compute_rdkit_property_table(cache, length_estimator)
        rdkit_tables.append(table)
        table.to_csv(layout["rdkit"] / f"{cache.spec.dataset_id}_rdkit_properties.csv", index=False)
        log(f"RDKit properties written for {cache.spec.dataset_id}")
    rdkit_property_table = pd.concat(rdkit_tables, ignore_index=True) if rdkit_tables else pd.DataFrame()
    rdkit_property_table.to_csv(layout["summary"] / "rdkit_property_values.csv", index=False)
    rdkit_summary = summarize_rdkit_properties(rdkit_property_table)
    rdkit_summary_path = layout["summary"] / "rdkit_property_summary.csv"
    rdkit_summary.to_csv(rdkit_summary_path, index=False)
    log(f"RDKit summary written: {rdkit_summary_path}")

    peptiverse_summary_path: Optional[Path] = None
    wasserstein_path: Optional[Path] = None
    ratio_path: Optional[Path] = None
    merged_table = rdkit_property_table.copy()
    if not args.skip_peptiverse:
        log(f"Loading PeptiVerse predictor on device={args.device}")
        predictor = PeptiVerseBatchPredictor.load(device=args.device)
        log("Running PeptiVerse predictions")
        peptiverse_tables = []
        for cache in caches:
            prediction_table = run_peptiverse_predictions(cache, predictor)
            peptiverse_tables.append(prediction_table)
            prediction_table.to_csv(layout["peptiverse"] / f"{cache.spec.dataset_id}_peptiverse_predictions.csv", index=False)
            predictor.clear_runtime_cache()
            log(f"PeptiVerse predictions written for {cache.spec.dataset_id}")
        peptiverse_table = pd.concat(peptiverse_tables, ignore_index=True) if peptiverse_tables else pd.DataFrame()
        peptiverse_table.to_csv(layout["summary"] / "peptiverse_prediction_values.csv", index=False)
        peptiverse_summary = summarize_peptiverse_predictions(peptiverse_table)
        peptiverse_summary_path = layout["summary"] / "peptiverse_prediction_summary.csv"
        peptiverse_summary.to_csv(peptiverse_summary_path, index=False)
        log(f"PeptiVerse summary written: {peptiverse_summary_path}")

        merged_table = merge_property_tables(rdkit_property_table, peptiverse_table)
        merged_table.to_csv(layout["summary"] / "merged_property_table.csv", index=False)
        log("Merged RDKit and PeptiVerse tables")

        wasserstein_summary = compute_wasserstein_summary(merged_table, reference_dataset_id="Reference")
        wasserstein_path = layout["metrics"] / "wasserstein_summary.csv"
        wasserstein_summary.to_csv(wasserstein_path, index=False)
        log(f"Wasserstein summary written: {wasserstein_path}")

        ratio_summary = compute_positive_ratio_summary(merged_table, dataset_order=order)
        ratio_path = layout["metrics"] / "positive_ratio_summary.csv"
        ratio_summary.to_csv(ratio_path, index=False)
        log(f"Positive ratio summary written: {ratio_path}")

        if not args.skip_plots:
            log("Generating figures")
            plotting_table = sample_plotting_table(
                merged_table,
                dataset_order=order,
                sample_size=args.plot_sample_size,
                sample_seed=args.plot_sample_seed,
            )
            log(
                f"Plotting with sampled rows={len(plotting_table)} "
                f"(sample_size={args.plot_sample_size}, seed={args.plot_sample_seed})"
            )
            plot_kde_panels(
                plotting_table,
                dataset_order=order,
                dataset_label_map=labels,
                output_dir=layout["figures_kde"],
            )
            plot_violin_box_panels(
                plotting_table,
                dataset_order=order,
                dataset_label_map=labels,
                output_dir=layout["figures_violin"],
            )
            plot_half_violin_box_scatter_panels(
                plotting_table,
                dataset_order=order,
                dataset_label_map=labels,
                output_dir=layout["figures_half_violin"],
            )
            plot_discrete_histograms(
                plotting_table,
                dataset_order=order,
                dataset_label_map=labels,
                output_dir=layout["figures_hist"],
                focused_dataset_ids=focused_histogram_dataset_ids,
            )
            plot_composite_property_overview(
                plotting_table,
                dataset_order=order,
                dataset_label_map=labels,
                output_dir=layout["figures_composite"],
                focused_histogram_dataset_ids=focused_histogram_dataset_ids,
            )
            plot_classification_ratio_bars(
                compute_positive_ratio_summary(plotting_table, dataset_order=order),
                dataset_order=order,
                dataset_label_map=labels,
                output_dir=layout["figures_bar"],
            )
            log("Figure generation complete")

    write_markdown_summary(
        layout["summary"] / "property_pipeline_summary.md",
        args=args,
        inventory_path=inventory_path,
        rdkit_summary_path=rdkit_summary_path,
        peptiverse_summary_path=peptiverse_summary_path,
        wasserstein_path=wasserstein_path,
        ratio_path=ratio_path,
    )
    log("Markdown summary written")

    report = {
        "output_root": str(args.output_root),
        "inventory_path": str(inventory_path),
        "rdkit_summary_path": str(rdkit_summary_path),
        "peptiverse_summary_path": str(peptiverse_summary_path) if peptiverse_summary_path else None,
        "wasserstein_path": str(wasserstein_path) if wasserstein_path else None,
        "positive_ratio_path": str(ratio_path) if ratio_path else None,
        "num_datasets": len(specs),
        "dataset_ids": order,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
