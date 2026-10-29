#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


TEST2_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = TEST2_ROOT / "output" / "generated" / "lora" / "gen_results_head_to_tail_30000.csv"
DEFAULT_OUTPUT_ROOT = Path('results/head_to_tail_cyclic_visualization_20260412')
DEFAULT_LENGTH_BANDS = 5
DEFAULT_SAMPLES_PER_BAND = 4
DEFAULT_SEED = 42
TARGET_CYCLIZATION = "head_to_tail"

PX_PER_MM = 300.0 / 25.4
CANVAS_WIDTH_MM = 180.0
CANVAS_HEIGHT_MM = 150.0
CANVAS_WIDTH_PX = int(round(CANVAS_WIDTH_MM * PX_PER_MM))
CANVAS_HEIGHT_PX = int(round(CANVAS_HEIGHT_MM * PX_PER_MM))

LEFT_MARGIN_PX = 150
RIGHT_MARGIN_PX = 50
TOP_MARGIN_PX = 30
BOTTOM_MARGIN_PX = 35
ROW_GAP_PX = 16
COL_GAP_PX = 18

ROW_LABEL_COLOR = "#222222"
ROW_LABEL_FONT_SIZE = 28
MOL_LABEL_FONT_SIZE = 20
TITLE_FONT_SIZE = 22
SUBTITLE_FONT_SIZE = 17
BODY_FONT_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a head-to-tail cyclic peptide montage across length bands.",
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cyclization-type", type=str, default=TARGET_CYCLIZATION)
    parser.add_argument("--length-bands", type=int, default=DEFAULT_LENGTH_BANDS)
    parser.add_argument("--samples-per-band", type=int, default=DEFAULT_SAMPLES_PER_BAND)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_length_module() -> Any:
    length_script = TEST2_ROOT / "evaluation" / "analysis" / "compute_peptide_lengths.py"
    spec = importlib.util.spec_from_file_location("compute_peptide_lengths", length_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load canonical length estimator from {length_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_length_estimator() -> Any:
    module = load_length_module()
    type_oracle = module.load_type_oracle(TEST2_ROOT)
    return module, type_oracle


def detect_smiles_column(df: pd.DataFrame) -> str:
    candidates = ["SMILES", "Smiles", "smiles"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Unable to find a SMILES column. Available columns: {list(df.columns)}")


def detect_valid_column(df: pd.DataFrame) -> str | None:
    candidates = ["VALID", "Valid", "valid", "is_valid"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def detect_cyclization_column(df: pd.DataFrame) -> str | None:
    candidates = ["Cyclization", "cyclization", "Topology", "topology"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def infer_length(module: Any, type_oracle: Any, smiles: str) -> dict[str, Any]:
    return module.infer_length_from_smiles(smiles, type_oracle)


def build_valid_length_table(
    dataframe: pd.DataFrame,
    *,
    source_csv: Path,
    length_module: Any,
    type_oracle: Any,
    cyclization_type: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    smiles_column = detect_smiles_column(dataframe)
    valid_column = detect_valid_column(dataframe)
    cyclization_column = detect_cyclization_column(dataframe)

    stats = {
        "input_rows": int(len(dataframe)),
        "rdkit_valid_rows": 0,
        "valid_flag_rows": 0,
        "cyclization_match_rows": 0,
        "length_valid_rows": 0,
        "duplicate_canonical_rows": 0,
    }

    records: list[dict[str, Any]] = []
    seen_canonical: set[str] = set()

    for row_index, row in dataframe.iterrows():
        smiles = str(row[smiles_column]).strip()
        if not smiles or smiles.lower() == "nan":
            continue

        valid_flag = True
        if valid_column is not None:
            valid_flag = coerce_bool(row[valid_column])
            if valid_flag:
                stats["valid_flag_rows"] += 1
        if not valid_flag:
            continue

        if cyclization_column is not None:
            cyclization_value = str(row[cyclization_column])
            if cyclization_value != cyclization_type:
                continue
            stats["cyclization_match_rows"] += 1

        canonical_smiles = canonicalize_smiles(smiles)
        if canonical_smiles is None:
            continue
        stats["rdkit_valid_rows"] += 1

        if canonical_smiles in seen_canonical:
            stats["duplicate_canonical_rows"] += 1
            continue
        seen_canonical.add(canonical_smiles)

        length_info = infer_length(length_module, type_oracle, smiles)
        if not length_info["valid_smiles"] or length_info["peptide_length"] is None:
            continue
        stats["length_valid_rows"] += 1

        records.append(
            {
                "source_row_index": int(row_index),
                "source_csv": str(source_csv),
                "SMILES": smiles,
                "canonical_smiles": canonical_smiles,
                "length": int(length_info["peptide_length"]),
                "valid_flag": bool(valid_flag),
                "length_failure_reasons": ";".join(length_info["failure_reasons"]),
            }
        )

    return pd.DataFrame(records), stats


def build_length_bands(table: pd.DataFrame, *, n_bands: int, samples_per_band: int, seed: int) -> pd.DataFrame:
    if table.empty:
        raise ValueError("No valid peptides available for visualization.")

    ordered = table.sort_values(["length", "canonical_smiles", "source_row_index"], kind="stable").reset_index(drop=True)
    n_bands = max(1, min(n_bands, len(ordered)))

    band_frames = []
    row_slices = list(range(len(ordered)))
    for band_index, band_rows in enumerate(np.array_split(row_slices, n_bands)):
        if len(band_rows) == 0:
            continue
        band_df = ordered.iloc[list(band_rows)].copy()
        band_min = int(band_df["length"].min())
        band_max = int(band_df["length"].max())
        band_label = f"Band {band_index + 1}: L={band_min}–{band_max}"
        band_df["band_index"] = band_index
        band_df["band_label"] = band_label
        band_df["band_min_length"] = band_min
        band_df["band_max_length"] = band_max
        band_df["row_in_band"] = range(len(band_df))
        band_frames.append(band_df)

    banded = pd.concat(band_frames, ignore_index=True)
    rng = random.Random(seed)
    selected_rows: list[pd.Series] = []
    for band_index, band_df in banded.groupby("band_index", sort=True):
        band_df = band_df.sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)
        if len(band_df) >= samples_per_band:
            chosen = band_df.head(samples_per_band).copy()
        else:
            chosen = band_df.copy()
            while len(chosen) < samples_per_band:
                chosen = pd.concat([chosen, band_df.iloc[[len(chosen) % len(band_df)]]], ignore_index=True)
        chosen["sample_rank"] = range(1, len(chosen) + 1)
        selected_rows.append(chosen)

    selected = pd.concat(selected_rows, ignore_index=True)
    selected = selected.sort_values(["band_index", "sample_rank", "length", "canonical_smiles"], kind="stable").reset_index(drop=True)
    return selected


def extract_inner_svg(svg_text: str) -> str:
    start = svg_text.find("<svg")
    if start < 0:
        raise ValueError("RDKit SVG output does not contain an <svg> tag.")
    end = svg_text.rfind("</svg>")
    if end < 0:
        raise ValueError("RDKit SVG output does not contain a closing </svg> tag.")
    return svg_text[start : end + len("</svg>")]


def build_montage_svg(
    selected: pd.DataFrame,
    *,
    output_root: Path,
    length_bands: int,
    samples_per_band: int,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    width = CANVAS_WIDTH_PX
    height = CANVAS_HEIGHT_PX
    inner_width = width - LEFT_MARGIN_PX - RIGHT_MARGIN_PX
    inner_height = height - TOP_MARGIN_PX - BOTTOM_MARGIN_PX
    row_height = int((inner_height - (length_bands - 1) * ROW_GAP_PX) / length_bands)
    col_width = int((inner_width - (samples_per_band - 1) * COL_GAP_PX) / samples_per_band)
    molecule_height = int(row_height * 0.78)
    label_height = row_height - molecule_height

    rows_summary: list[dict[str, Any]] = []
    svg_parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{CANVAS_WIDTH_MM}mm' height='{CANVAS_HEIGHT_MM}mm' viewBox='0 0 {width} {height}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='white' />",
        f"<text x='{LEFT_MARGIN_PX}' y='18' font-family='Arial, Helvetica, sans-serif' font-size='{TITLE_FONT_SIZE}' font-weight='700' fill='#111111'>Head-to-tail cyclic peptides selected across length bands</text>",
        f"<text x='{LEFT_MARGIN_PX}' y='42' font-family='Arial, Helvetica, sans-serif' font-size='{SUBTITLE_FONT_SIZE}' fill='#444444'>Valid, non-duplicate model outputs sampled reproducibly from the canonical 97.1% residue-length estimator</text>",
    ]

    for band_index in range(length_bands):
        band_df = selected[selected["band_index"] == band_index].copy().sort_values(["sample_rank", "length", "canonical_smiles"])
        if band_df.empty:
            continue
        band_label = str(band_df["band_label"].iloc[0])
        band_min = int(band_df["band_min_length"].iloc[0])
        band_max = int(band_df["band_max_length"].iloc[0])
        band_top = TOP_MARGIN_PX + band_index * (row_height + ROW_GAP_PX)
        band_center_y = band_top + row_height * 0.50
        rows_summary.append(
            {
                "band_index": band_index,
                "band_label": band_label,
                "band_min_length": band_min,
                "band_max_length": band_max,
                "selected_count": int(len(band_df)),
            }
        )
        svg_parts.append(
            f"<text x='16' y='{band_center_y}' font-family='Arial, Helvetica, sans-serif' font-size='{ROW_LABEL_FONT_SIZE}' font-weight='700' fill='{ROW_LABEL_COLOR}' dominant-baseline='middle'>{escape(band_label)}</text>"
        )

        for col_index, (_, row) in enumerate(band_df.iterrows()):
            cell_left = LEFT_MARGIN_PX + col_index * (col_width + COL_GAP_PX)
            molecule_svg = extract_inner_svg(molecule_svg_text(row["SMILES"], width=col_width, height=molecule_height))
            svg_parts.append(
                f"<g transform='translate({cell_left}, {band_top})'>"
                f"<rect x='0' y='0' width='{col_width}' height='{molecule_height}' fill='none' stroke='#DDDDDD' stroke-width='1' rx='8' ry='8' />"
                f"<svg x='0' y='0' width='{col_width}' height='{molecule_height}' viewBox='0 0 {col_width} {molecule_height}'>"
                f"{molecule_svg}"
                f"</svg>"
                f"<text x='{col_width / 2:.1f}' y='{molecule_height + label_height * 0.70:.1f}' text-anchor='middle' font-family='Arial, Helvetica, sans-serif' font-size='{MOL_LABEL_FONT_SIZE}' font-weight='600' fill='#222222'>L={int(row['length'])}</text>"
                f"</g>"
            )

    svg_parts.append("</svg>")
    summary = {
        "output_root": str(output_root),
        "seed": int(seed),
        "length_bands": int(length_bands),
        "samples_per_band": int(samples_per_band),
        "band_summaries": rows_summary,
        "canvas": {
            "width_mm": CANVAS_WIDTH_MM,
            "height_mm": CANVAS_HEIGHT_MM,
            "width_px": width,
            "height_px": height,
            "left_margin_px": LEFT_MARGIN_PX,
            "right_margin_px": RIGHT_MARGIN_PX,
            "top_margin_px": TOP_MARGIN_PX,
            "bottom_margin_px": BOTTOM_MARGIN_PX,
        },
    }
    return "\n".join(svg_parts), summary


def molecule_svg_text(smiles: str, *, width: int, height: int) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot draw invalid SMILES: {smiles}")
    mol = Chem.RemoveHs(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.continuousHighlight = False
    opts.bondLineWidth = 1.4
    opts.minFontSize = 10
    opts.maxFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def ensure_cairosvg() -> Any:
    try:
        import cairosvg
    except ImportError as exc:  # pragma: no cover - dependency availability is environment specific
        raise RuntimeError(
            "cairosvg is required to export PDF/PNG. Install it in the target environment before running."
        ) from exc
    return cairosvg


def export_montage(svg_text: str, output_root: Path) -> tuple[Path, Path, Path]:
    cairosvg = ensure_cairosvg()
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    svg_path = figures_dir / "head_to_tail_cyclic_peptide_montage.svg"
    pdf_path = figures_dir / "head_to_tail_cyclic_peptide_montage.pdf"
    png_path = figures_dir / "head_to_tail_cyclic_peptide_montage.png"
    svg_path.write_text(svg_text, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg_text.encode("utf-8"), write_to=str(pdf_path), dpi=300)
    cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(png_path), dpi=300)
    return svg_path, pdf_path, png_path


def write_manifest(selected: pd.DataFrame, output_root: Path) -> Path:
    manifest_dir = output_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "selected_molecules.csv"
    columns = [
        "band_index",
        "band_label",
        "band_min_length",
        "band_max_length",
        "sample_rank",
        "length",
        "SMILES",
        "canonical_smiles",
        "source_row_index",
        "source_csv",
        "valid_flag",
        "length_failure_reasons",
    ]
    selected.loc[:, columns].to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return manifest_path


def write_summary(summary: dict[str, Any], output_root: Path) -> Path:
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.force:
        expected_svg = output_root / "figures" / "head_to_tail_cyclic_peptide_montage.svg"
        expected_manifest = output_root / "manifest" / "selected_molecules.csv"
        if expected_svg.exists() and expected_manifest.exists():
            print(json.dumps({"status": "already_exists", "output_root": str(output_root)}, indent=2))
            return

    dataframe = pd.read_csv(args.input_csv)
    length_module, type_oracle = load_length_estimator()
    valid_table, load_stats = build_valid_length_table(
        dataframe,
        source_csv=args.input_csv,
        length_module=length_module,
        type_oracle=type_oracle,
        cyclization_type=args.cyclization_type,
    )
    if valid_table.empty:
        raise ValueError(f"No valid head-to-tail peptides found in {args.input_csv}")

    selected = build_length_bands(
        valid_table,
        n_bands=args.length_bands,
        samples_per_band=args.samples_per_band,
        seed=args.seed,
    )
    manifest_path = write_manifest(selected, output_root)
    svg_text, montage_summary = build_montage_svg(
        selected,
        output_root=output_root,
        length_bands=args.length_bands,
        samples_per_band=args.samples_per_band,
        seed=args.seed,
    )
    svg_path, pdf_path, png_path = export_montage(svg_text, output_root)

    summary = {
        "source_csv": str(args.input_csv),
        "target_cyclization_type": args.cyclization_type,
        "output_root": str(output_root),
        "seed": int(args.seed),
        "length_bands": int(args.length_bands),
        "samples_per_band": int(args.samples_per_band),
        "input_rows": int(load_stats["input_rows"]),
        "rdkit_valid_rows": int(load_stats["rdkit_valid_rows"]),
        "valid_flag_rows": int(load_stats["valid_flag_rows"]),
        "cyclization_match_rows": int(load_stats["cyclization_match_rows"]),
        "length_valid_rows": int(load_stats["length_valid_rows"]),
        "duplicate_canonical_rows": int(load_stats["duplicate_canonical_rows"]),
        "selected_count": int(len(selected)),
        "unique_lengths_selected": int(selected["length"].nunique()),
        "length_min": int(selected["length"].min()),
        "length_max": int(selected["length"].max()),
        "length_mean": float(mean(selected["length"].tolist())),
        "band_summaries": montage_summary["band_summaries"],
        "manifest_csv": str(manifest_path),
        "figures": {
            "svg": str(svg_path),
            "pdf": str(pdf_path),
            "png": str(png_path),
        },
        "notes": "Selected molecules are valid, deduplicated by canonical SMILES, and partitioned into equal-count length bands before sampling.",
    }
    summary_path = write_summary(summary, output_root)

    print(json.dumps(
        {
            "manifest_csv": str(manifest_path),
            "summary_json": str(summary_path),
            "svg": str(svg_path),
            "pdf": str(pdf_path),
            "png": str(png_path),
            "selected_count": int(len(selected)),
            "length_min": int(selected["length"].min()),
            "length_max": int(selected["length"].max()),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
