#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw molecules selected by rank from a CSV as individual SVG files."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--rank-column", default="rank")
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser.parse_args()


def normalize_rank(value: str) -> int:
    try:
        return int(float(str(value).strip()))
    except ValueError as exc:
        raise ValueError(f"Invalid rank value: {value!r}") from exc


def select_rows(
    input_csv: Path,
    *,
    ranks: list[int],
    rank_column: str,
    smiles_column: str,
) -> list[dict[str, str]]:
    requested = set(ranks)
    selected: dict[int, dict[str, str]] = {}

    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in (rank_column, smiles_column) if column not in fieldnames]
        if missing:
            raise ValueError(f"Missing required columns {missing}; available columns: {fieldnames}")

        for row in reader:
            rank = normalize_rank(row[rank_column])
            if rank not in requested:
                continue
            if rank in selected:
                raise ValueError(f"Rank {rank} occurs more than once in {input_csv}")
            selected[rank] = row

    missing_ranks = sorted(requested.difference(selected))
    if missing_ranks:
        raise ValueError(f"Requested ranks not found: {missing_ranks}")
    return [selected[rank] for rank in ranks]


def render_svg(smiles: str, *, width: int, height: int) -> tuple[str, str, int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    mol = Chem.RemoveHs(mol)
    rdDepictor.Compute2DCoords(
        mol,
        canonOrient=True,
        clearConfs=True,
        forceRDKit=True,
    )

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.addStereoAnnotation = False
    options.bondLineWidth = 1.6
    options.minFontSize = 14
    options.maxFontSize = 24
    options.padding = 0.10
    options.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText(), canonical_smiles, mol.GetNumAtoms()


def write_manifest(rows: list[dict[str, str]], output_dir: Path, rank_column: str) -> Path:
    manifest_path = output_dir / "selected_molecules.csv"
    preferred = [
        rank_column,
        "SMILES",
        "canonical_smiles",
        "three_letter_sequence",
        "one_letter_sequence",
        "Cyclization",
        "kras_kd_pred",
        "kras_score",
    ]
    columns = [column for column in preferred if any(column in row for row in rows)]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if len(args.ranks) != len(set(args.ranks)):
        raise ValueError("--ranks must not contain duplicates")

    rows = select_rows(
        args.input_csv,
        ranks=args.ranks,
        rank_column=args.rank_column,
        smiles_column=args.smiles_column,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    for row in rows:
        rank = normalize_rank(row[args.rank_column])
        smiles = str(row[args.smiles_column]).strip()
        svg_text, canonical_smiles, atom_count = render_svg(
            smiles,
            width=args.width,
            height=args.height,
        )
        output_path = args.output_dir / f"rank_{rank:04d}.svg"
        output_path.write_text(svg_text, encoding="utf-8")
        rendered.append(
            {
                "rank": rank,
                "output_svg": str(output_path),
                "canonical_smiles": canonical_smiles,
                "atom_count": atom_count,
            }
        )

    manifest_path = write_manifest(rows, args.output_dir, args.rank_column)
    record_path = args.output_dir / "render_record.json"
    record_path.write_text(
        json.dumps(
            {
                "input_csv": str(args.input_csv),
                "rank_column": args.rank_column,
                "smiles_column": args.smiles_column,
                "canvas_px": [args.width, args.height],
                "manifest_csv": str(manifest_path),
                "rendered": rendered,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(record_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
