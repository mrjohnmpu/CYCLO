#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from dataclasses import dataclass
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from property_pipeline.plot_distributions import (
    PLOT_DEFAULT_SAMPLE_SEED,
    plot_composite_property_overview,
    plot_discrete_histograms,
    plot_half_violin_box_scatter_panels,
    plot_kde_panels,
    plot_violin_box_panels,
    sample_plotting_table,
)


REFERENCE_DATASET_ID = "Reference"
REFERENCE_DATASET_LABEL = "Reference"
DEFAULT_REFERENCE_CYCLIZATION_COLUMN = "Cyclization"
DEFAULT_WRAPPER_PLOT_SAMPLE_SIZE = 3000
DEFAULT_REFERENCE_MIN_LENGTH = 6
DEFAULT_DISCRETE_FOCUS_DATASET_IDS = (REFERENCE_DATASET_ID, "ours")

NUMERIC_COLUMNS = [
    "MW",
    "LogP",
    "HBD",
    "HBA",
    "TPSA",
    "RotBonds",
    "half_life_value",
    "PAMPA_value",
    "Caco-2_value",
    "peptide_length",
]


@dataclass
class DatasetEntry:
    dataset_id: str
    dataset_label: str
    csv_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot property comparison figures from pre-annotated peptide property CSV files."
    )
    parser.add_argument(
        "--topology",
        required=True,
        help="Topology to compare, for example head_to_tail, sc_to_tail, or disulfide_bridge.",
    )
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        help="Dataset definition in the form dataset_id::dataset_label::csv_path. Repeat for each generated model.",
    )
    parser.add_argument(
        "--reference-cyclization-col",
        default=DEFAULT_REFERENCE_CYCLIZATION_COLUMN,
        help="Column used to filter the reference dataset by topology.",
    )
    parser.add_argument(
        "--reference-min-length",
        type=int,
        default=DEFAULT_REFERENCE_MIN_LENGTH,
        help="Minimum peptide_length allowed for the reference dataset after topology filtering.",
    )
    parser.add_argument("--plot-sample-size", type=int, default=DEFAULT_WRAPPER_PLOT_SAMPLE_SIZE)
    parser.add_argument("--plot-sample-seed", type=int, default=PLOT_DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--discrete-focus-dataset-ids",
        type=str,
        default=",".join(DEFAULT_DISCRETE_FOCUS_DATASET_IDS),
        help="Comma-separated dataset ids to keep in Residue Count/HBD/HBA bar charts.",
    )
    return parser.parse_args()


def parse_dataset_entries(values: Iterable[str]) -> list[DatasetEntry]:
    entries: list[DatasetEntry] = []
    seen_ids: set[str] = set()
    for raw in values:
        parts = raw.split("::", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --dataset value '{raw}'. Expected format dataset_id::dataset_label::csv_path."
            )
        dataset_id, dataset_label, csv_path = parts
        entry = DatasetEntry(
            dataset_id=dataset_id.strip(),
            dataset_label=dataset_label.strip(),
            csv_path=Path(csv_path.strip()),
        )
        if not entry.dataset_id or not entry.dataset_label or not str(entry.csv_path):
            raise ValueError(f"Invalid --dataset value '{raw}'. None of the three fields may be empty.")
        if entry.dataset_id in seen_ids:
            raise ValueError(f"Duplicate dataset_id '{entry.dataset_id}' in --dataset arguments.")
        seen_ids.add(entry.dataset_id)
        entries.append(entry)
    return entries


def normalize_topology(value: str) -> str:
    return value.strip().lower()


def ensure_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")


def parse_dataset_id_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


def filter_reference_by_topology(
    dataframe: pd.DataFrame,
    *,
    topology: str,
    cyclization_col: str,
    source_path: Path,
) -> tuple[pd.DataFrame, int]:
    if cyclization_col not in dataframe.columns:
        raise ValueError(f"Reference CSV {source_path} is missing required column '{cyclization_col}'.")
    filtered = dataframe[dataframe[cyclization_col].astype(str).str.lower() == topology].copy()
    filtered.reset_index(drop=True, inplace=True)
    if filtered.empty:
        raise ValueError(f"Reference CSV {source_path} has no rows for topology '{topology}'.")
    return filtered, int(len(filtered))


def filter_reference_by_min_length(
    dataframe: pd.DataFrame,
    *,
    min_length: int,
    source_path: Path,
) -> pd.DataFrame:
    lengths = pd.to_numeric(dataframe["peptide_length"], errors="coerce")
    filtered = dataframe[lengths >= float(min_length)].copy()
    filtered.reset_index(drop=True, inplace=True)
    if filtered.empty:
        raise ValueError(
            f"Reference CSV {source_path} has no rows remaining after peptide_length >= {min_length} filtering."
        )
    return filtered


def filter_generated_dataset_by_topology(
    dataframe: pd.DataFrame,
    *,
    topology: str,
    source_path: Path,
) -> tuple[pd.DataFrame, str, int]:
    if DEFAULT_REFERENCE_CYCLIZATION_COLUMN in dataframe.columns:
        filtered = dataframe[dataframe[DEFAULT_REFERENCE_CYCLIZATION_COLUMN].astype(str).str.lower() == topology]
        if filtered.empty:
            raise ValueError(f"Generated CSV {source_path} has no rows for topology '{topology}'.")
        filtered = filtered.copy()
        filtered.reset_index(drop=True, inplace=True)
        return filtered, f"{DEFAULT_REFERENCE_CYCLIZATION_COLUMN} == {topology}", int(len(filtered))

    topology_marker = topology.lower()
    source_name = source_path.name.lower()
    source_full = str(source_path).lower()
    if topology_marker not in source_name and topology_marker not in source_full:
        raise ValueError(
            f"Generated CSV {source_path} has no '{DEFAULT_REFERENCE_CYCLIZATION_COLUMN}' column and its path does "
            f"not contain topology marker '{topology_marker}'. Refusing to mix topologies."
        )
    filtered = dataframe.copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered, f"path contains '{topology}'", int(len(filtered))


def coerce_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Length"] = frame["peptide_length"] if "peptide_length" in frame.columns else pd.Series(dtype=float)
    return frame


def attach_dataset_metadata(dataframe: pd.DataFrame, *, dataset_id: str, dataset_label: str) -> pd.DataFrame:
    frame = dataframe.copy()
    frame["dataset_id"] = dataset_id
    frame["dataset_label"] = dataset_label
    return frame


def ensure_required_plot_columns(dataframe: pd.DataFrame, *, source_path: Path) -> None:
    required = {
        "MW",
        "LogP",
        "HBD",
        "HBA",
        "TPSA",
        "RotBonds",
        "half_life_value",
        "PAMPA_value",
        "Caco-2_value",
        "peptide_length",
    }
    missing = sorted(column for column in required if column not in dataframe.columns)
    if missing:
        raise ValueError(f"CSV {source_path} is missing required plotting columns: {missing}")


def build_cli_command() -> str:
    command = [sys.executable, *sys.argv]
    return " ".join(shlex.quote(part) for part in command)


def prepare_layout(output_root: Path) -> dict[str, Path]:
    layout = {
        "root": output_root,
        "summary": output_root / "summary",
        "kde": output_root / "figures" / "kde",
        "violin": output_root / "figures" / "violin",
        "half_violin": output_root / "figures" / "scatter_box_half_violin",
        "hist": output_root / "figures" / "hist",
        "composite": output_root / "figures" / "composite",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def write_markdown_record(
    output_path: Path,
    *,
    command: str,
    manifest: dict[str, Any],
    source_inventory: list[dict[str, Any]],
    plotted_rows_per_dataset: dict[str, int],
    key_figure_paths: dict[str, str],
) -> None:
    lines = [
        "# Plot Run Record",
        "",
        f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Command: `{command}`",
        f"- Topology: `{manifest['topology']}`",
        f"- Reference minimum length: `{manifest['reference_min_length']}`",
        "- Discrete histogram focus dataset ids: "
        + (
            ", ".join(f"`{dataset_id}`" for dataset_id in manifest["discrete_focus_dataset_ids"])
            if manifest["discrete_focus_dataset_ids"]
            else "`all datasets`"
        ),
        f"- Plot sample size per dataset: `{manifest['plot_sample_size_per_dataset']}`",
        f"- Plot sample seed: `{manifest['plot_sample_seed']}`",
        f"- Output root: `{manifest['output_root']}`",
        "",
        "## Inputs",
        "",
        "| dataset_id | dataset_label | csv_path | filter_mode | raw_rows | rows_after_topology_filter | rows_after_all_filters | plotted_rows |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in source_inventory:
        dataset_id = str(row["dataset_id"])
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset_id,
                    str(row["dataset_label"]),
                    f"`{row['csv_path']}`",
                    str(row["filter_mode"]),
                    str(row["raw_row_count"]),
                    str(row["row_count_after_topology_filter"]),
                    str(row["row_count_after_filter"]),
                    str(plotted_rows_per_dataset.get(dataset_id, 0)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Plot manifest: `{manifest['plot_manifest_path']}`",
            f"- Source inventory CSV: `{manifest['source_inventory_csv']}`",
            f"- Plotting table sample CSV: `{manifest['plotting_table_sample_csv']}`",
            f"- Composite figure: `{key_figure_paths['composite']}`",
            f"- Residue Count histogram: `{key_figure_paths['hist_length']}`",
            f"- HBD histogram: `{key_figure_paths['hist_hbd']}`",
            f"- HBA histogram: `{key_figure_paths['hist_hba']}`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    topology = normalize_topology(args.topology)
    dataset_entries = parse_dataset_entries(args.datasets)
    discrete_focus_dataset_ids = parse_dataset_id_list(args.discrete_focus_dataset_ids)
    dataset_order = [REFERENCE_DATASET_ID] + [entry.dataset_id for entry in dataset_entries]
    dataset_label_map = {
        REFERENCE_DATASET_ID: REFERENCE_DATASET_LABEL,
        **{entry.dataset_id: entry.dataset_label for entry in dataset_entries},
    }
    if discrete_focus_dataset_ids:
        missing_focus = [dataset_id for dataset_id in discrete_focus_dataset_ids if dataset_id not in dataset_order]
        if missing_focus:
            raise ValueError(
                f"--discrete-focus-dataset-ids contains unknown dataset ids: {missing_focus}. "
                f"Known ids: {dataset_order}"
            )

    ensure_exists(args.reference_csv)
    for entry in dataset_entries:
        ensure_exists(entry.csv_path)

    layout = prepare_layout(args.output_root)

    reference_raw_df = pd.read_csv(args.reference_csv)
    reference_raw_row_count = int(len(reference_raw_df))
    reference_topology_df, reference_rows_after_topology = filter_reference_by_topology(
        reference_raw_df,
        topology=topology,
        cyclization_col=args.reference_cyclization_col,
        source_path=args.reference_csv,
    )
    ensure_required_plot_columns(reference_topology_df, source_path=args.reference_csv)
    reference_df = filter_reference_by_min_length(
        reference_topology_df,
        min_length=args.reference_min_length,
        source_path=args.reference_csv,
    )
    reference_df = coerce_numeric_columns(reference_df)
    reference_df = attach_dataset_metadata(
        reference_df,
        dataset_id=REFERENCE_DATASET_ID,
        dataset_label=dataset_label_map[REFERENCE_DATASET_ID],
    )

    prepared_frames = [reference_df]
    dataset_row_counts: dict[str, int] = {REFERENCE_DATASET_ID: int(len(reference_df))}
    source_inventory: list[dict[str, str | int]] = [
        {
            "dataset_id": REFERENCE_DATASET_ID,
            "dataset_label": dataset_label_map[REFERENCE_DATASET_ID],
            "csv_path": str(args.reference_csv),
            "topology": topology,
            "raw_row_count": reference_raw_row_count,
            "row_count_after_topology_filter": reference_rows_after_topology,
            "row_count_after_filter": int(len(reference_df)),
            "filter_mode": (
                f"{args.reference_cyclization_col} == {topology}; peptide_length >= {args.reference_min_length}"
            ),
        }
    ]

    for entry in dataset_entries:
        raw_dataframe = pd.read_csv(entry.csv_path)
        raw_row_count = int(len(raw_dataframe))
        dataframe, filter_mode, rows_after_topology = filter_generated_dataset_by_topology(
            raw_dataframe,
            topology=topology,
            source_path=entry.csv_path,
        )
        ensure_required_plot_columns(dataframe, source_path=entry.csv_path)
        dataframe = coerce_numeric_columns(dataframe)
        dataframe = attach_dataset_metadata(dataframe, dataset_id=entry.dataset_id, dataset_label=entry.dataset_label)
        prepared_frames.append(dataframe)
        dataset_row_counts[entry.dataset_id] = int(len(dataframe))
        source_inventory.append(
            {
                "dataset_id": entry.dataset_id,
                "dataset_label": entry.dataset_label,
                "csv_path": str(entry.csv_path),
                "topology": topology,
                "raw_row_count": raw_row_count,
                "row_count_after_topology_filter": rows_after_topology,
                "row_count_after_filter": int(len(dataframe)),
                "filter_mode": filter_mode,
            }
        )

    merged_table = pd.concat(prepared_frames, ignore_index=True)
    plotting_table = sample_plotting_table(
        merged_table,
        dataset_order=dataset_order,
        sample_size=args.plot_sample_size,
        sample_seed=args.plot_sample_seed,
    )
    plotting_table_path = layout["summary"] / "plotting_table_sample.csv"
    source_inventory_path = layout["summary"] / "source_inventory.csv"
    plotting_table.to_csv(plotting_table_path, index=False)
    pd.DataFrame(source_inventory).to_csv(source_inventory_path, index=False)

    plotted_rows_per_dataset = {
        dataset_id: int((plotting_table["dataset_id"] == dataset_id).sum()) for dataset_id in dataset_order
    }

    plot_kde_panels(
        plotting_table,
        dataset_order=dataset_order,
        dataset_label_map=dataset_label_map,
        output_dir=layout["kde"],
    )
    plot_violin_box_panels(
        plotting_table,
        dataset_order=dataset_order,
        dataset_label_map=dataset_label_map,
        output_dir=layout["violin"],
    )
    plot_half_violin_box_scatter_panels(
        plotting_table,
        dataset_order=dataset_order,
        dataset_label_map=dataset_label_map,
        output_dir=layout["half_violin"],
    )
    plot_discrete_histograms(
        plotting_table,
        dataset_order=dataset_order,
        dataset_label_map=dataset_label_map,
        output_dir=layout["hist"],
        focused_dataset_ids=discrete_focus_dataset_ids,
    )
    plot_composite_property_overview(
        plotting_table,
        dataset_order=dataset_order,
        dataset_label_map=dataset_label_map,
        output_dir=layout["composite"],
        focused_histogram_dataset_ids=discrete_focus_dataset_ids,
    )

    key_figure_paths = {
        "composite": str(layout["composite"] / "composite_3x3_property_overview.pdf"),
        "hist_length": str(layout["hist"] / "hist_length.pdf"),
        "hist_hbd": str(layout["hist"] / "hist_hbd.pdf"),
        "hist_hba": str(layout["hist"] / "hist_hba.pdf"),
    }

    manifest = {
        "topology": topology,
        "reference_csv": str(args.reference_csv),
        "reference_label": dataset_label_map[REFERENCE_DATASET_ID],
        "reference_cyclization_col": args.reference_cyclization_col,
        "reference_min_length": args.reference_min_length,
        "dataset_order": dataset_order,
        "dataset_label_map": dataset_label_map,
        "discrete_focus_dataset_ids": discrete_focus_dataset_ids,
        "plot_sample_size_per_dataset": args.plot_sample_size,
        "plot_sample_seed": args.plot_sample_seed,
        "raw_input_rows_per_dataset": {
            str(row["dataset_id"]): int(row["raw_row_count"]) for row in source_inventory
        },
        "rows_after_topology_filter_per_dataset": {
            str(row["dataset_id"]): int(row["row_count_after_topology_filter"]) for row in source_inventory
        },
        "rows_after_all_filters_per_dataset": dataset_row_counts,
        "plotted_rows_per_dataset": plotted_rows_per_dataset,
        "output_root": str(args.output_root),
        "figure_dirs": {key: str(path) for key, path in layout.items() if key not in {"root", "summary"}},
        "source_inventory_csv": str(source_inventory_path),
        "plotting_table_sample_csv": str(plotting_table_path),
        "key_figure_paths": key_figure_paths,
    }
    plot_manifest_path = layout["summary"] / "plot_manifest.json"
    plot_run_record_path = layout["summary"] / "plot_run_record.md"
    manifest["plot_manifest_path"] = str(plot_manifest_path)
    manifest["plot_run_record_md"] = str(plot_run_record_path)
    manifest["command"] = build_cli_command()
    plot_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown_record(
        plot_run_record_path,
        command=manifest["command"],
        manifest=manifest,
        source_inventory=source_inventory,
        plotted_rows_per_dataset=plotted_rows_per_dataset,
        key_figure_paths=key_figure_paths,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
