#!/usr/bin/env python3
"""Prepare and verify the Shared-base property annotation without changing the predictor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from importlib import metadata
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "evaluation/configs/shared_base_property_annotation_seed42_20260923.json"
METRICS = ROOT / "results/ablations/token_lora_shared_seed42_20260914/metrics_summary.csv"
OUTPUT = ROOT / "results/property_annotation_runs/shared_base_seed42"
FIGURES = ROOT / "results/figures/shared_base_property_distribution"
PLOT_CONFIGS = ROOT / "evaluation/plots/configs"
TOPOLOGIES = ("head_to_tail", "sc_to_tail", "disulfide_bridge")
PROPERTIES = (
    "MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds",
    "hemolysis_score", "non_fouling_score", "toxicity_score",
    "half_life_value", "PAMPA_value", "Caco-2_value",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selected_metrics(config: dict) -> tuple[list[dict], list[dict], list[str]]:
    with METRICS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row["Method"] == "Shared token+LoRA"]
    if len(rows) != 3 or [row["Cyclization"] for row in rows] != list(TOPOLOGIES):
        raise ValueError("Expected exactly three ordered Shared token+LoRA MOSES rows")
    jobs = {job["name"].removesuffix("_n30000"): job for job in config["inputs"]}
    if set(jobs) != set(TOPOLOGIES):
        raise ValueError("Configuration must contain each topology exactly once")
    sources = []
    for row in rows:
        topology = row["Cyclization"]
        path = Path(jobs[topology]["csv_path"])
        if row["GeneratedCSV"] != str(path) or int(row["TotalCount"]) != 30000:
            raise ValueError(f"MOSES source or denominator mismatch: {topology}")
        source = pd.read_csv(path, dtype={"SMILES": "string", "Valid": "string"})
        if list(source.columns) != ["SMILES", "Valid"] or len(source) != 30000:
            raise ValueError(f"Unexpected generated CSV schema or length: {path}")
        flags = source["Valid"].str.lower()
        if not flags.isin(["true", "false"]).all():
            raise ValueError(f"Non-boolean Valid column: {path}")
        valid_count = int(flags.eq("true").sum())
        if valid_count != int(row["ValidCount"]):
            raise ValueError(f"MOSES valid count mismatch: {path}")
        if abs(float(row["Validity"]) - valid_count / 30000) > 1e-12:
            raise ValueError(f"MOSES validity mismatch: {path}")
        sources.append({
            "Cyclization": topology,
            "GeneratedCSV": str(path),
            "SHA256": sha256(path),
            "TotalCount": 30000,
            "ValidCount": valid_count,
            "InvalidCount": 30000 - valid_count,
        })
    return rows, sources, fields


def prepare() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {OUTPUT}")
    if FIGURES.exists() and any(FIGURES.iterdir()):
        raise FileExistsError(f"Nonempty figure directory: {FIGURES}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["output_dir"] = str(OUTPUT / "outputs")
    for job in config["inputs"]:
        job["csv_path"] = str(resolve_path(job["csv_path"]))
    rows, sources, fields = selected_metrics(config)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "tables").mkdir()
    (OUTPUT / "smoke/inputs").mkdir(parents=True)
    (OUTPUT / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(OUTPUT / "tables/shared_base_moses_metrics.csv", rows, fields)
    write_csv(OUTPUT / "tables/input_summary.csv", sources, list(sources[0]))
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"source_metrics": str(METRICS), "sources": sources}, indent=2) + "\n",
        encoding="utf-8",
    )
    versions = {"python": sys.version.split()[0]}
    for package in ("pandas", "rdkit", "torch", "transformers", "numpy", "peptiverse"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed as distribution"
    (OUTPUT / "software_versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")

    smoke_config = dict(config)
    smoke_config["output_dir"] = str(OUTPUT / "smoke/outputs")
    smoke_config["inputs"] = []
    for source in sources:
        topology = source["Cyclization"]
        input_path = OUTPUT / "smoke/inputs" / f"{topology}_n32.csv"
        pd.read_csv(source["GeneratedCSV"], nrows=32).to_csv(input_path, index=False)
        smoke_config["inputs"].append({"name": f"{topology}_n32", "csv_path": str(input_path)})
    (OUTPUT / "smoke/config.json").write_text(json.dumps(smoke_config, indent=2) + "\n", encoding="utf-8")

    manifest_dir = FIGURES / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for topology in TOPOLOGIES:
        source = PLOT_CONFIGS / f"{topology}_property_sources.json"
        target = manifest_dir / f"{topology}_property_sources_shared_seed42_20260923.json"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing manifest: {target}")
        manifest = json.loads(source.read_text(encoding="utf-8"))
        for item in manifest["datasets"]:
            item["path"] = str(resolve_path(item["path"]))
        ours = [item for item in manifest["datasets"] if item["id"] == "ours"]
        if len(ours) != 1 or ours[0]["label"] != "TopoCycGPT":
            raise ValueError(f"Unexpected ours entry: {source}")
        ours[0]["path"] = str(OUTPUT / "outputs" / f"{topology}_n30000.with_properties.csv")
        ours[0]["label"] = "Shared token+LoRA"
        target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_outputs(smoke: bool) -> None:
    report_path = OUTPUT / ("smoke/annotation_report.json" if smoke else "annotation_report.json")
    reports = json.loads(report_path.read_text(encoding="utf-8"))["reports"]
    if len(reports) != 3:
        raise ValueError("Expected three annotation reports")
    sources = json.loads((OUTPUT / "source_manifest.json").read_text(encoding="utf-8"))["sources"]
    source_by_topology = {item["Cyclization"]: item for item in sources}
    summary = []
    property_rows = []
    for topology, report in zip(TOPOLOGIES, reports):
        expected_suffix = f"{topology}_n32" if smoke else f"{topology}_n30000"
        path = Path(report["output_csv"])
        if path.name != f"{expected_suffix}.with_properties.csv":
            raise ValueError(f"Unexpected report order or filename: {path}")
        frame = pd.read_csv(path)
        if not set(PROPERTIES).issubset(frame.columns):
            raise ValueError(f"Missing property columns in {path}")
        if len(frame) != int(report["rows_written"]) or frame.empty:
            raise ValueError(f"Unexpected output length: {path}")
        if not smoke:
            expected_valid = source_by_topology[topology]["ValidCount"]
            if int(report["rows_in"]) != 30000 or int(report["rows_rdkit_valid"]) != expected_valid:
                raise ValueError(f"RDKit count does not match MOSES: {path}")
            if len(frame) != expected_valid:
                raise ValueError(f"Canonical de-duplication changed row count: {path}")
        for prop in PROPERTIES:
            values = pd.to_numeric(frame[prop], errors="coerce")
            if values.notna().sum() == 0:
                raise ValueError(f"Property {prop} is entirely missing in {path}")
            if not np.isfinite(values.dropna()).all():
                raise ValueError(f"Property {prop} contains non-finite values in {path}")
            if not smoke:
                property_rows.append({
                    "Cyclization": topology,
                    "Property": prop,
                    "Count": int(values.notna().sum()),
                    "MissingCount": int(values.isna().sum()),
                    "Mean": float(values.mean()),
                    "Median": float(values.median()),
                    "Std": float(values.std()),
                    "Q25": float(values.quantile(0.25)),
                    "Q75": float(values.quantile(0.75)),
                })
        if not smoke:
            summary.append({
                "Cyclization": topology,
                "RowsInput": int(report["rows_in"]),
                "RowsRDKitValid": int(report["rows_rdkit_valid"]),
                "RowsOutput": len(frame),
                "DuplicatesRemoved": int(report["rows_rdkit_valid"]) - len(frame),
                "PredictionFailureCounts": json.dumps(report["peptiverse_failure_counts"], sort_keys=True),
                "OutputCSV": str(path),
                "OutputSHA256": sha256(path),
            })
    if not smoke:
        write_csv(OUTPUT / "tables/property_annotation_summary.csv", summary, list(summary[0]))
        write_csv(OUTPUT / "tables/property_summary.csv", property_rows, list(property_rows[0]))
        (OUTPUT / "run_record.md").write_text(
            "# Shared base property evaluation\n\n"
            f"- Annotation config: `{CONFIG}`\n"
            f"- Source metrics: `{METRICS}` (reused, not recalculated)\n"
            "- Device and GPU binding are configured by the annotation command.\n"
            f"- Figures: `{FIGURES}`\n"
            "- `Cyclization` denotes requested token, not inferred topology.\n"
            "- MOSES validity uses all 30,000 generated rows; properties use RDKit-valid canonical-unique rows.\n",
            encoding="utf-8",
        )


def main() -> None:
    global CONFIG, METRICS, OUTPUT, FIGURES, PLOT_CONFIGS
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "verify-smoke", "finalize"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--metrics-csv", type=Path, default=METRICS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    parser.add_argument("--plot-config-dir", type=Path, default=PLOT_CONFIGS)
    args = parser.parse_args()
    CONFIG = resolve_path(args.config)
    METRICS = resolve_path(args.metrics_csv)
    OUTPUT = resolve_path(args.output_dir)
    FIGURES = resolve_path(args.figures_dir)
    PLOT_CONFIGS = resolve_path(args.plot_config_dir)
    if args.stage == "prepare":
        prepare()
    else:
        verify_outputs(smoke=args.stage == "verify-smoke")


if __name__ == "__main__":
    main()
