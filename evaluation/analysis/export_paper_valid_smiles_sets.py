#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger, rdBase


TRUE_VALUES = {"1", "true", "t", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RDKit-valid SMILES sets with standardized cyclization labels."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def rule_mask(series: pd.Series, rule: dict[str, Any]) -> pd.Series:
    normalized = normalize_text(series).str.lower()
    kind = rule.get("kind", "equals")
    if kind == "bool_true":
        return normalized.isin(TRUE_VALUES)
    if kind == "equals":
        accepted = {str(value).strip().lower() for value in rule.get("values", [])}
        return normalized.isin(accepted)
    raise ValueError(f"Unsupported validity rule kind: {kind}")


def rdkit_is_valid(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles, sanitize=True) is not None
    except Exception:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_optional(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def process_source(source: dict[str, Any], output_root: Path, overwrite: bool) -> dict[str, Any]:
    source_path = Path(source["path"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source_path}")

    header = pd.read_csv(source_path, nrows=0)
    smiles_column = source["smiles_column"]
    rules = source.get("source_validity", [])
    required = {smiles_column, *(rule["column"] for rule in rules)}
    missing = sorted(required - set(header.columns))
    if missing:
        raise KeyError(f"{source['dataset_id']} missing columns: {missing}")

    frame = pd.read_csv(source_path, usecols=list(required), low_memory=False)
    smiles = normalize_text(frame[smiles_column])
    nonempty = smiles.ne("")

    source_valid = pd.Series(True, index=frame.index, dtype=bool)
    for rule in rules:
        source_valid &= rule_mask(frame[rule["column"]], rule)

    rdkit_valid = smiles.map(rdkit_is_valid)
    export_mask = nonempty & rdkit_valid
    if rules:
        export_mask &= source_valid

    output_path = output_root / source["relative_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace it: {output_path}")

    exported = pd.DataFrame(
        {
            "SMILES": smiles.loc[export_mask].to_numpy(),
            "Cyclization": source["cyclization"],
        }
    )
    exported.to_csv(output_path, index=False)

    total_rows = len(frame)
    rdkit_valid_rows = int(rdkit_valid.sum())
    output_rows = len(exported)
    source_valid_rows = int(source_valid.sum()) if rules else None
    source_valid_but_rdkit_invalid = int((source_valid & ~rdkit_valid).sum()) if rules else None
    source_invalid_but_rdkit_valid = int((~source_valid & rdkit_valid).sum()) if rules else None
    rdkit_rate = rdkit_valid_rows / total_rows if total_rows else float("nan")
    source_rate = source_valid_rows / total_rows if rules and total_rows else None
    output_rate = output_rows / total_rows if total_rows else float("nan")
    manuscript_validity = source.get("manuscript_validity")

    return {
        "dataset_id": source["dataset_id"],
        "category": source["category"],
        "method": source["method"],
        "stage": source["stage"],
        "cyclization": source["cyclization"],
        "source_path": str(source_path),
        "smiles_column": smiles_column,
        "source_validity_rules": json.dumps(rules, separators=(",", ":")),
        "output_path": str(output_path),
        "total_rows": total_rows,
        "nonempty_smiles_rows": int(nonempty.sum()),
        "source_valid_rows": source_valid_rows,
        "source_validity_rate": source_rate,
        "rdkit_valid_rows": rdkit_valid_rows,
        "rdkit_validity_rate": rdkit_rate,
        "source_valid_but_rdkit_invalid": source_valid_but_rdkit_invalid,
        "source_invalid_but_rdkit_valid": source_invalid_but_rdkit_valid,
        "output_rows": output_rows,
        "output_rate": output_rate,
        "unique_output_smiles": int(exported["SMILES"].nunique()),
        "duplicate_output_rows": int(output_rows - exported["SMILES"].nunique()),
        "manuscript_validity": manuscript_validity,
        "rdkit_minus_manuscript": (
            rdkit_rate - float(manuscript_validity) if manuscript_validity is not None else None
        ),
        "matches_manuscript_4dp": (
            round(rdkit_rate, 4) == round(float(manuscript_validity), 4)
            if manuscript_validity is not None
            else None
        ),
        "output_sha256": sha256_file(output_path),
    }


def write_run_record(
    output_root: Path,
    manifest_path: Path,
    summaries: pd.DataFrame,
    command: str,
) -> None:
    lines = [
        "# Paper sequence export record",
        "",
        "## Runtime",
        f"- Timestamp: {datetime.now().astimezone().isoformat()}",
        f"- Host: {socket.gethostname()}",
        f"- Python: {sys.executable}",
        f"- Python version: {platform.python_version()}",
        f"- RDKit version: {rdBase.rdkitVersion}",
        f"- Manifest: {manifest_path}",
        f"- Output root: {output_root}",
        f"- Command: `{command}`",
        "",
        "## Rules",
        "- RDKit validity requires `Chem.MolFromSmiles(smiles, sanitize=True)` to succeed.",
        "- When a source validity field exists, exported rows must pass both source validity and RDKit validity.",
        "- PepINVENT sources do not contain a validity column, so RDKit validity is the sole filter.",
        "- Row order and duplicate valid SMILES are preserved.",
        "- Every exported CSV contains exactly `SMILES,Cyclization`.",
        "",
        "## Summary",
        "",
        "| Dataset | Category | Method | Stage | Cyclization | Total | RDKit valid | RDKit rate | Exported | Manuscript | Match 4dp |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries.to_dict(orient="records"):
        lines.append(
            "| {dataset_id} | {category} | {method} | {stage} | {cyclization} | "
            "{total_rows} | {rdkit_valid_rows} | {rdkit_rate} | {output_rows} | "
            "{manuscript} | {match} |".format(
                **row,
                rdkit_rate=format_optional(row["rdkit_validity_rate"]),
                manuscript=format_optional(row["manuscript_validity"], 4),
                match=format_optional(row["matches_manuscript_4dp"]),
            )
        )
    lines.extend(
        [
            "",
            "## Machine-readable outputs",
            f"- {output_root / 'tables' / 'validity_summary.csv'}",
            f"- {output_root / 'tables' / 'manuscript_validity_comparison.csv'}",
            f"- {output_root / 'source_manifest.json'}",
        ]
    )
    (output_root / "run_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = manifest.get("sources", [])
    if not sources:
        raise ValueError("Manifest has no sources")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_root / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    RDLogger.DisableLog("rdApp.error")
    summaries: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        print(f"[{index}/{len(sources)}] {source['dataset_id']}", flush=True)
        summaries.append(process_source(source, args.output_root, args.overwrite))

    summary_frame = pd.DataFrame(summaries)
    summary_path = args.output_root / "tables" / "validity_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    comparison = summary_frame[summary_frame["manuscript_validity"].notna()].copy()
    comparison.to_csv(
        args.output_root / "tables" / "manuscript_validity_comparison.csv", index=False
    )
    command = " ".join(shlex.quote(part) for part in sys.argv)
    write_run_record(args.output_root, args.manifest, summary_frame, command)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
