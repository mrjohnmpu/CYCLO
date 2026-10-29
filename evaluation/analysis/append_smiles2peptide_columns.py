#!/usr/bin/env python3
"""Append SMILES2PEPTIDE-derived columns to an existing CSV."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import socket
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_INPUT_CSV = Path(
    'results/property_annotation_runs/20260423_115024_lora_cyclic_n30000_gpu0/outputs/head_to_tail_n30000.with_properties.csv'
)
DEFAULT_OUTPUT_CSV = Path(
    'results/property_annotation_runs/20260423_115024_lora_cyclic_n30000_gpu0/outputs/head_to_tail_n30000.with_properties.with_smiles2peptide.csv'
)
DEFAULT_RECORD_MD = Path(
    'results/property_annotation_runs/20260423_115024_lora_cyclic_n30000_gpu0/outputs/head_to_tail_n30000.with_properties.with_smiles2peptide.md'
)
DEFAULT_PROJECT_ROOT = Path(os.environ.get(
    "SMILES2PEPTIDE_ROOT", SCRIPT_PATH.parents[2] / "third_party" / "SMILES2PEPTIDE"
)).expanduser().resolve()
DEFAULT_SMILES_COLUMN = "SMILES"
APPEND_COLUMNS = [
    "three_letter_sequence",
    "one_letter_sequence",
    "is_cyclic",
    "residues",
    "peptide_length",
    "detected_uaas",
    "n_detected_uaas",
    "n_natural_resolved",
    "n_non_natural_resolved",
    "frac_natural_resolved",
    "frac_non_natural_resolved",
    "parse_status",
    "notes",
]


@dataclass
class RunResult:
    started_at: str
    finished_at: str
    status: str
    input_csv: Path
    output_csv: Path
    record_md: Path
    project_root: Path
    smiles_column: str
    max_rows: int | None
    rows_read: int | None = None
    rows_written: int | None = None
    parse_status_counts: dict[str, int] | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append SMILES2PEPTIDE-derived columns to an existing CSV without overwriting the source file."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--record-md", type=Path, default=DEFAULT_RECORD_MD)
    parser.add_argument("--smiles-column", type=str, default=DEFAULT_SMILES_COLUMN)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def load_smiles2peptide_batch_module() -> Any:
    script_path = SCRIPT_PATH.with_name("smiles2peptide_batch.py")
    spec = importlib.util.spec_from_file_location("smiles2peptide_batch", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load batch module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if not args.project_root.exists():
        raise FileNotFoundError(f"SMILES2PEPTIDE project root not found: {args.project_root}")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be a positive integer")
    if args.input_csv.resolve() == args.output_csv.resolve():
        raise ValueError("Output CSV must be different from input CSV")
    if args.output_csv.exists():
        raise FileExistsError(f"Output CSV already exists: {args.output_csv}")


def read_input_csv(input_csv: Path, max_rows: int | None) -> pd.DataFrame:
    read_kwargs: dict[str, Any] = {}
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows
    return pd.read_csv(input_csv, **read_kwargs)


def ensure_expected_columns(dataframe: pd.DataFrame, smiles_column: str) -> None:
    if smiles_column not in dataframe.columns:
        raise KeyError(f"Missing SMILES column {smiles_column!r} in input columns {list(dataframe.columns)}")
    conflicts = [column for column in APPEND_COLUMNS if column in dataframe.columns]
    if conflicts:
        raise ValueError(f"Input CSV already contains target columns: {conflicts}")


def build_command_string() -> str:
    return shlex.join([sys.executable, str(SCRIPT_PATH), *sys.argv[1:]])


def append_columns(args: argparse.Namespace) -> RunResult:
    started_at = iso_now()
    result = RunResult(
        started_at=started_at,
        finished_at=started_at,
        status="failed",
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        record_md=args.record_md,
        project_root=args.project_root,
        smiles_column=args.smiles_column,
        max_rows=args.max_rows,
    )

    try:
        validate_args(args)
        source_df = read_input_csv(args.input_csv, args.max_rows)
        ensure_expected_columns(source_df, args.smiles_column)

        batch_module = load_smiles2peptide_batch_module()
        analyzer = batch_module.load_peptide_analyzer(args.project_root)
        parsed_df, _, _ = batch_module.convert_dataframe(
            source_df,
            smiles_column=args.smiles_column,
            id_column=None,
            analyzer=analyzer,
            max_rows=None,
        )

        appended_df = pd.concat(
            [
                source_df.reset_index(drop=True),
                parsed_df.loc[:, APPEND_COLUMNS].reset_index(drop=True),
            ],
            axis=1,
        )
        if len(appended_df) != len(source_df):
            raise RuntimeError(
                f"Row count mismatch after append: source={len(source_df)} appended={len(appended_df)}"
            )

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        appended_df.to_csv(args.output_csv, index=False)

        parse_status_counts = (
            parsed_df["parse_status"].fillna("missing").astype(str).value_counts(dropna=False).to_dict()
        )
        result.rows_read = int(len(source_df))
        result.rows_written = int(len(appended_df))
        result.parse_status_counts = {str(key): int(value) for key, value in parse_status_counts.items()}
        result.status = "success"
        return result
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result.finished_at = iso_now()
        write_record_md(result)


def write_record_md(result: RunResult) -> None:
    result.record_md.parent.mkdir(parents=True, exist_ok=True)
    status_counts = result.parse_status_counts or {}
    parsed_ok = sum(count for key, count in status_counts.items() if key != "failed")
    failed = status_counts.get("failed", 0)

    lines = [
        "# SMILES2PEPTIDE Column Append Record",
        "",
        "## Operation",
        "Append non-duplicate SMILES2PEPTIDE parsing fields to an existing property-annotated CSV without modifying the source file.",
        "",
        "## Runtime",
        f"- Started: {result.started_at}",
        f"- Finished: {result.finished_at}",
        f"- Host: {socket.gethostname()}",
        f"- Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', sys.prefix)}",
        f"- Python executable: {sys.executable}",
        "",
        "## Inputs and Outputs",
        f"- Script: {SCRIPT_PATH}",
        f"- Input CSV: {result.input_csv}",
        f"- Output CSV: {result.output_csv}",
        f"- Record file: {result.record_md}",
        f"- SMILES column: {result.smiles_column}",
        f"- SMILES2PEPTIDE project root: {result.project_root}",
        f"- Max rows: {result.max_rows if result.max_rows is not None else 'full file'}",
        "",
        "## Added Columns",
    ]
    lines.extend([f"- {column}" for column in APPEND_COLUMNS])
    lines.extend(
        [
            "",
            "## Command",
            "```bash",
            build_command_string(),
            "```",
            "",
            "## Result",
            f"- Status: {result.status}",
            f"- Rows read: {result.rows_read if result.rows_read is not None else 'n/a'}",
            f"- Rows written: {result.rows_written if result.rows_written is not None else 'n/a'}",
            f"- Parsed successfully (full or partial): {parsed_ok}",
            f"- Failed parses: {failed}",
            "",
            "## Parse Status Counts",
        ]
    )

    if status_counts:
        lines.extend([f"- {key}: {value}" for key, value in sorted(status_counts.items())])
    else:
        lines.append("- unavailable")

    lines.extend(["", "## Summary"])
    if result.status == "success":
        lines.append("- Completed successfully.")
        lines.append("- Source CSV was left unchanged.")
    else:
        lines.append("- Run failed.")
        lines.append(f"- Error: {result.error or 'unknown error'}")

    result.record_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = append_columns(args)
    print(
        json.dumps(
            {
                "status": result.status,
                "input_csv": str(result.input_csv),
                "output_csv": str(result.output_csv),
                "record_md": str(result.record_md),
                "rows_read": result.rows_read,
                "rows_written": result.rows_written,
                "parse_status_counts": result.parse_status_counts,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
