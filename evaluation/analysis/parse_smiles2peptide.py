#!/usr/bin/env python3
"""Parse peptide SMILES into residue sequences with peptide_parser."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import shlex
import shutil
import socket
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger


DEFAULT_INPUT_CSV = Path('./data/restored_validation.csv')
DEFAULT_OUTPUT_DIR = Path(
    'results/peptide_parser/restored_validation_parse_smiles2peptide'
)
DEFAULT_PEPTIDE_PARSER_ROOT = Path(os.environ.get(
    "PEPTIDE_PARSER_ROOT", Path(__file__).resolve().parents[2] / "third_party" / "peptide_parser"
)).expanduser().resolve()
DEFAULT_SMILES_COLUMN = "SMILES"
DEFAULT_MAX_ROWS = 10

SUMMARY_COLUMNS = [
    "source_index",
    "mol_id",
    "input_smiles",
    "parse_status",
    "error",
    "peptide_sequence",
    "monomer_id_sequence",
    "peptide_length",
    "topology_json_path",
    "report_dir",
    "report_html_path",
    "molecule_svg_path",
    "report_status",
    "report_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse SMILES from a CSV into peptide residue sequences and per-molecule "
            "topology JSON files using the private peptide_parser project."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smiles-column", type=str, default=DEFAULT_SMILES_COLUMN)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument(
        "--peptide-parser-root",
        type=Path,
        default=DEFAULT_PEPTIDE_PARSER_ROOT,
        help="Path to the private zhaisilong/peptide_parser source checkout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove previous generated outputs under --output-dir before running.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse existing topology_json/row_xxxxxxx.topology.json files and only "
            "parse missing or invalid rows."
        ),
    )
    parser.add_argument(
        "--write-reports",
        action="store_true",
        help=(
            "Write per-molecule images and HTML reports under <output-dir>/reports. "
            "Uses peptide_parser's full report when svgutils is available; otherwise "
            "writes a lightweight fallback report with molecule and side-chain PNGs."
        ),
    )
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def normalize_smiles(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def validate_args(args: argparse.Namespace) -> None:
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if not args.peptide_parser_root.exists():
        raise FileNotFoundError(
            f"peptide_parser root not found: {args.peptide_parser_root}"
        )
    parser_file = args.peptide_parser_root / "peptide_parser" / "parser.py"
    if not parser_file.exists():
        raise FileNotFoundError(
            f"peptide_parser source tree looks invalid; missing {parser_file}"
        )
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be a positive integer")


def prepare_output_dir(output_dir: Path, overwrite: bool, resume: bool) -> None:
    generated_paths = [
        output_dir / "parsed_peptides.csv",
        output_dir / "run_record.md",
        output_dir / "topology_json",
        output_dir / "reports",
    ]
    existing = [path for path in generated_paths if path.exists()]
    if existing and not overwrite and not resume:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Output already exists. Pass --overwrite to replace generated files: "
            f"or --resume to reuse existing topology JSON files: {existing_text}"
        )

    if overwrite:
        for path in generated_paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    (output_dir / "topology_json").mkdir(parents=True, exist_ok=True)


def load_peptide_parser_api(peptide_parser_root: Path):
    root_text = str(peptide_parser_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    # The private package is chatty when used as an API; keep batch logs readable.
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    from peptide_parser.parser import Parser
    from peptide_parser.topology import build_topology, write_topology_json

    return Parser, build_topology, write_topology_json


def try_load_full_html_report() -> Any | None:
    try:
        from peptide_parser.utils.reports import html_report
    except ModuleNotFoundError:
        return None
    return html_report


def try_load_draw_side_chain() -> Any | None:
    try:
        from peptide_parser.utils.drawing import draw_side_chain
    except Exception:
        return None
    return draw_side_chain


def read_input_csv(input_csv: Path, smiles_column: str, max_rows: int) -> pd.DataFrame:
    dataframe = pd.read_csv(input_csv, nrows=max_rows)
    if smiles_column not in dataframe.columns:
        raise KeyError(
            f"Missing SMILES column {smiles_column!r}. Available columns: "
            f"{list(dataframe.columns)}"
        )
    return dataframe


def row_id_from_index(source_index: Any, row_number: int) -> str:
    try:
        return f"row_{int(source_index):07d}"
    except (TypeError, ValueError):
        return f"row_{row_number:07d}"


def residue_names_from_topology(topology: dict[str, Any]) -> list[str]:
    residues = topology.get("topology", {}).get("residues", [])
    names: list[str] = []
    for residue in residues:
        name = (
            residue.get("name")
            or residue.get("side_chain_name")
            or residue.get("monomer_id")
            or ""
        )
        name_text = str(name).strip()
        if name_text:
            names.append(name_text)
    return names


def monomer_ids_from_topology(topology: dict[str, Any]) -> list[str]:
    residues = topology.get("topology", {}).get("residues", [])
    monomer_ids: list[str] = []
    for residue in residues:
        monomer_id = residue.get("monomer_id") or residue.get("name") or ""
        monomer_id_text = str(monomer_id).strip()
        if monomer_id_text:
            monomer_ids.append(monomer_id_text)
    return monomer_ids


def summary_from_topology_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty topology JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        topology = json.load(handle)

    residues = topology.get("topology", {}).get("residues", [])
    residue_count = topology.get("topology", {}).get("residue_count", len(residues))
    if not isinstance(residues, list) or not residues:
        raise ValueError(f"No residues found in topology JSON: {path}")

    return {
        "parse_status": "success",
        "error": "",
        "peptide_sequence": "|".join(residue_names_from_topology(topology)),
        "monomer_id_sequence": "|".join(monomer_ids_from_topology(topology)),
        "peptide_length": int(residue_count),
        "topology_json_path": str(path),
        "report_dir": "",
        "report_html_path": "",
        "molecule_svg_path": "",
        "report_status": "not_requested",
        "report_error": "",
    }


def write_fallback_report(
    *,
    molecule_ser: pd.Series,
    match_df: pd.DataFrame,
    report_dir: Path,
    mol_id: str,
    peptide_sequence: str,
    peptide_length: int,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    side_chain_image_dir = report_dir / "side_chain_images"
    side_chain_image_dir.mkdir(parents=True, exist_ok=True)

    molecule_png = report_dir / "molecule.png"
    mol = molecule_ser.get("mol")
    if mol is not None:
        Draw.MolToFile(mol, str(molecule_png), size=(900, 650))

    draw_side_chain = try_load_draw_side_chain()
    residue_rows = []
    for idx, (_, residue) in enumerate(match_df.iterrows()):
        residue_name = str(
            residue.get("residue_name")
            or residue.get("side_chain_name")
            or residue.get("monomer_id")
            or f"residue_{idx}"
        )
        image_rel = ""
        graph = residue.get("everything_graph")
        if draw_side_chain is not None and graph is not None:
            image_path = side_chain_image_dir / f"r{idx:03d}.png"
            try:
                draw_side_chain(graph).save(image_path)
                image_rel = f"side_chain_images/{image_path.name}"
            except Exception:
                image_rel = ""

        image_cell = (
            f'<img src="{html.escape(image_rel)}" width="80" height="80">'
            if image_rel
            else ""
        )
        residue_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(residue_name)}</td>"
            f"<td>{html.escape(str(residue.get('ld', '')))}</td>"
            f"<td>{html.escape(str(residue.get('backbone_type', '')))}</td>"
            f"<td>{image_cell}</td>"
            "</tr>"
        )

    molecule_svg = report_dir / "molecule.svg"
    if mol is not None:
        Draw.MolToFile(mol, str(molecule_svg), size=(900, 650))

    report_html_path = report_dir / "report.html"
    molecule_img = (
        '<img src="molecule.svg" width="700" alt="molecule">'
        if molecule_svg.exists()
        else '<img src="molecule.png" width="700" alt="molecule">'
        if molecule_png.exists()
        else "<p>Molecule image was not generated.</p>"
    )
    report_html_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(mol_id)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    table {{ border-collapse: collapse; margin-top: 18px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    .sequence {{ max-width: 1100px; overflow-wrap: anywhere; line-height: 1.6; }}
  </style>
</head>
<body>
  <h1>{html.escape(mol_id)}</h1>
  <h2>Peptide sequence</h2>
  <p class="sequence">{html.escape(peptide_sequence)}</p>
  <p><b>Peptide length:</b> {peptide_length}</p>
  <h2>Molecule</h2>
  {molecule_img}
  <h2>Residues</h2>
  <table>
    <thead>
      <tr><th>Index</th><th>Residue</th><th>L/D</th><th>Backbone type</th><th>Side-chain image</th></tr>
    </thead>
    <tbody>
      {''.join(residue_rows)}
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_html_path


def write_report_artifacts(
    *,
    molecule_ser: pd.Series,
    match_df: pd.DataFrame,
    report_dir: Path,
    mol_id: str,
    peptide_sequence: str,
    peptide_length: int,
    full_html_report: Any | None,
) -> dict[str, str]:
    try:
        if full_html_report is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            report_html_path = report_dir / "report.html"
            molecule_svg_path = report_dir / "mol.svg"
            full_html_report(
                molecule_ser,
                match_df,
                report_html_path,
                save_mol_svg=True,
            )
            return {
                "report_dir": str(report_dir),
                "report_html_path": str(report_html_path),
                "molecule_svg_path": str(molecule_svg_path),
                "report_status": "full",
                "report_error": "",
            }

        report_html_path = write_fallback_report(
            molecule_ser=molecule_ser,
            match_df=match_df,
            report_dir=report_dir,
            mol_id=mol_id,
            peptide_sequence=peptide_sequence,
            peptide_length=peptide_length,
        )
        return {
            "report_dir": str(report_dir),
            "report_html_path": str(report_html_path),
            "molecule_svg_path": str(report_dir / "molecule.svg"),
            "report_status": "fallback",
            "report_error": "svgutils_not_available_for_full_report",
        }
    except Exception as exc:  # noqa: BLE001 - report generation is optional.
        return {
            "report_dir": str(report_dir),
            "report_html_path": "",
            "molecule_svg_path": "",
            "report_status": "failed",
            "report_error": f"{type(exc).__name__}: {exc}",
        }


def parse_one_smiles(
    *,
    smiles: str,
    mol_id: str,
    parser: Any,
    build_topology: Any,
    write_topology_json: Any,
    topology_json_path: Path,
    report_dir: Path | None,
    full_html_report: Any | None,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not smiles:
        return {
            "parse_status": "failed",
            "error": "empty_smiles",
            "peptide_sequence": "",
            "monomer_id_sequence": "",
            "peptide_length": pd.NA,
            "topology_json_path": "",
            "report_dir": "",
            "report_html_path": "",
            "molecule_svg_path": "",
            "report_status": "not_requested" if report_dir is None else "skipped",
            "report_error": "",
        }

    try:
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            molecule_ser, match_df = parser.parse_smiles(smiles, name=mol_id)
        topology = build_topology(molecule_ser, match_df)
    except Exception as exc:  # noqa: BLE001 - record per-row parser failures.
        return {
            "parse_status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "peptide_sequence": "",
            "monomer_id_sequence": "",
            "peptide_length": pd.NA,
            "topology_json_path": "",
            "report_dir": "",
            "report_html_path": "",
            "molecule_svg_path": "",
            "report_status": "not_requested" if report_dir is None else "skipped",
            "report_error": "",
        }

    reason = molecule_ser.get("reason_for_exclusion")
    residue_count = topology.get("topology", {}).get("residue_count", 0)
    if pd.notna(reason):
        return {
            "parse_status": "failed",
            "error": f"reason_for_exclusion:{reason}",
            "peptide_sequence": "",
            "monomer_id_sequence": "",
            "peptide_length": pd.NA,
            "topology_json_path": "",
            "report_dir": "",
            "report_html_path": "",
            "molecule_svg_path": "",
            "report_status": "not_requested" if report_dir is None else "skipped",
            "report_error": "",
        }
    if not residue_count:
        return {
            "parse_status": "failed",
            "error": "no_residues_detected",
            "peptide_sequence": "",
            "monomer_id_sequence": "",
            "peptide_length": pd.NA,
            "topology_json_path": "",
            "report_dir": "",
            "report_html_path": "",
            "molecule_svg_path": "",
            "report_status": "not_requested" if report_dir is None else "skipped",
            "report_error": "",
        }

    write_topology_json(molecule_ser, match_df, topology_json_path)
    if stats is not None:
        stats["parsed_success"] = stats.get("parsed_success", 0) + 1
    peptide_sequence = "|".join(residue_names_from_topology(topology))
    monomer_id_sequence = "|".join(monomer_ids_from_topology(topology))
    report_info = {
        "report_dir": "",
        "report_html_path": "",
        "molecule_svg_path": "",
        "report_status": "not_requested",
        "report_error": "",
    }
    if report_dir is not None:
        report_info = write_report_artifacts(
            molecule_ser=molecule_ser,
            match_df=match_df,
            report_dir=report_dir,
            mol_id=mol_id,
            peptide_sequence=peptide_sequence,
            peptide_length=int(residue_count),
            full_html_report=full_html_report,
        )
    return {
        "parse_status": "success",
        "error": "",
        "peptide_sequence": peptide_sequence,
        "monomer_id_sequence": monomer_id_sequence,
        "peptide_length": int(residue_count),
        "topology_json_path": str(topology_json_path),
        **report_info,
    }


def parse_dataframe(
    dataframe: pd.DataFrame,
    *,
    smiles_column: str,
    output_dir: Path,
    parser: Any,
    build_topology: Any,
    write_topology_json: Any,
    write_reports: bool,
    full_html_report: Any | None,
    resume: bool,
    stats: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    topology_dir = output_dir / "topology_json"
    reports_dir = output_dir / "reports"

    for row_number, (source_index, row) in enumerate(dataframe.iterrows()):
        mol_id = row_id_from_index(source_index, row_number)
        smiles = normalize_smiles(row.get(smiles_column))
        topology_json_path = topology_dir / f"{mol_id}.topology.json"
        report_dir = reports_dir / mol_id if write_reports else None
        parsed = None
        if resume and topology_json_path.exists():
            try:
                parsed = summary_from_topology_json(topology_json_path)
                stats["resume_restored"] = stats.get("resume_restored", 0) + 1
            except Exception:
                stats["resume_invalid"] = stats.get("resume_invalid", 0) + 1
                parsed = None
        if parsed is None:
            parsed = parse_one_smiles(
                smiles=smiles,
                mol_id=mol_id,
                parser=parser,
                build_topology=build_topology,
                write_topology_json=write_topology_json,
                topology_json_path=topology_json_path,
                report_dir=report_dir,
                full_html_report=full_html_report,
                stats=stats,
            )
            if parsed["parse_status"] == "failed":
                stats["parsed_failed"] = stats.get("parsed_failed", 0) + 1
        rows.append(
            {
                "source_index": source_index,
                "mol_id": mol_id,
                "input_smiles": smiles,
                **parsed,
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def command_string() -> str:
    return shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])


def write_run_record(
    *,
    path: Path,
    args: argparse.Namespace,
    started_at: str,
    finished_at: str,
    summary_df: pd.DataFrame,
    stats: dict[str, int],
) -> None:
    status_counts = (
        summary_df["parse_status"]
        .fillna("missing")
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )
    lines = [
        "# peptide_parser SMILES-to-Peptide Parse Record",
        "",
        "## Runtime",
        f"- Started: {started_at}",
        f"- Finished: {finished_at}",
        f"- Host: {socket.gethostname()}",
        f"- Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', sys.prefix)}",
        f"- Python executable: {sys.executable}",
        f"- Command: `{command_string()}`",
        "",
        "## Inputs and Outputs",
        f"- Input CSV: {args.input_csv}",
        f"- Output directory: {args.output_dir}",
        f"- Summary CSV: {args.output_dir / 'parsed_peptides.csv'}",
        f"- Topology JSON directory: {args.output_dir / 'topology_json'}",
        f"- Reports directory: {args.output_dir / 'reports'}",
        f"- SMILES column: {args.smiles_column}",
        f"- Max rows: {args.max_rows}",
        f"- peptide_parser root: {args.peptide_parser_root}",
        f"- Write reports: {args.write_reports}",
        f"- Resume: {args.resume}",
        "",
        "## Summary",
        f"- Rows processed: {len(summary_df)}",
        f"- Success: {int(status_counts.get('success', 0))}",
        f"- Failed: {int(status_counts.get('failed', 0))}",
        "",
        "## Resume Stats",
        f"- Restored from existing JSON: {stats.get('resume_restored', 0)}",
        f"- Invalid existing JSON re-parsed: {stats.get('resume_invalid', 0)}",
        f"- Newly parsed success: {stats.get('parsed_success', 0)}",
        f"- Newly parsed failed: {stats.get('parsed_failed', 0)}",
        "",
        "## Parse Status Counts",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {int(count)}")

    failed = summary_df[summary_df["parse_status"] == "failed"]
    if not failed.empty:
        error_counts = failed["error"].fillna("unknown").value_counts().to_dict()
        lines.extend(["", "## Failure Reasons"])
        for error, count in error_counts.items():
            lines.append(f"- {error}: {int(count)}")

    if "report_status" in summary_df.columns:
        report_counts = (
            summary_df["report_status"]
            .fillna("missing")
            .astype(str)
            .value_counts(dropna=False)
            .to_dict()
        )
        lines.extend(["", "## Report Status Counts"])
        for status, count in sorted(report_counts.items()):
            lines.append(f"- {status}: {int(count)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started_at = iso_now()

    validate_args(args)
    prepare_output_dir(args.output_dir, args.overwrite, args.resume)

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    Parser, build_topology, write_topology_json = load_peptide_parser_api(
        args.peptide_parser_root
    )
    full_html_report = try_load_full_html_report() if args.write_reports else None
    parser = Parser()
    stats = {
        "resume_restored": 0,
        "resume_invalid": 0,
        "parsed_success": 0,
        "parsed_failed": 0,
    }
    dataframe = read_input_csv(args.input_csv, args.smiles_column, args.max_rows)
    summary_df = parse_dataframe(
        dataframe,
        smiles_column=args.smiles_column,
        output_dir=args.output_dir,
        parser=parser,
        build_topology=build_topology,
        write_topology_json=write_topology_json,
        write_reports=args.write_reports,
        full_html_report=full_html_report,
        resume=args.resume,
        stats=stats,
    )

    summary_path = args.output_dir / "parsed_peptides.csv"
    summary_df.to_csv(summary_path, index=False)

    finished_at = iso_now()
    write_run_record(
        path=args.output_dir / "run_record.md",
        args=args,
        started_at=started_at,
        finished_at=finished_at,
        summary_df=summary_df,
        stats=stats,
    )

    status_counts = summary_df["parse_status"].value_counts().to_dict()
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote topology JSON files to: {args.output_dir / 'topology_json'}")
    if args.write_reports:
        print(f"Wrote report artifacts to: {args.output_dir / 'reports'}")
    print(f"Wrote run record: {args.output_dir / 'run_record.md'}")
    print(
        "Resume stats: "
        + ", ".join(f"{key}={int(value)}" for key, value in sorted(stats.items()))
    )
    print(
        "Status counts: "
        + ", ".join(f"{key}={int(value)}" for key, value in sorted(status_counts.items()))
    )


if __name__ == "__main__":
    main()


# ssh weilab4090

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate general
# cd .

# RUN_NAME="restored_validation_parse_smiles2peptide_full_20260611"
# OUT_DIR="results/peptide_parser/${RUN_NAME}"
# LOG_DIR="results/peptide_parser/logs"
# LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
# PID_FILE="${LOG_DIR}/${RUN_NAME}.pid"

# mkdir -p "${LOG_DIR}"

# nohup python ./evaluation/analysis/parse_smiles2peptide.py \
#   --input-csv ./data/restored_validation.csv \
#   --output-dir "${OUT_DIR}" \
#   --smiles-column SMILES \
#   --max-rows 50000 \
#   --overwrite \
#   >"${LOG_FILE}" 2>&1 < /dev/null &

# echo $! > "${PID_FILE}"
