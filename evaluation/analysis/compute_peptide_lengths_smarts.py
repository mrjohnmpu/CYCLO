#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from rdkit import Chem, RDLogger


REASON_INVALID_SMILES = "invalid_smiles"
REASON_LENGTH_MISMATCH = "length_mismatch_vs_validation"
REASON_TYPE_MISMATCH = "type_mismatch_vs_validation"

DEFAULT_RESTORED_CSV = Path('./data/restored_validation.csv')
DEFAULT_ORIGINAL_CSV = Path('./data/validation_original.csv')
DEFAULT_OUTPUT_ROOT = Path('results/peptide_length_validation_smarts_20260411')

CA_PATTERN = Chem.MolFromSmarts("[NX3][C;!$(C=O);X4][CX3](=[OX1])")
SS_PATTERN = Chem.MolFromSmarts("[SX2H0][SX2H0]")
TERM_NH2_PATTERN = Chem.MolFromSmarts("[NH2;!R][CX4][CX3](=O)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate peptide length from full SMILES with a SMARTS-only C-alpha rule."
    )
    parser.add_argument("--restored-csv", type=Path, default=DEFAULT_RESTORED_CSV)
    parser.add_argument("--original-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--diagnostics-csv", type=Path, default=None)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_outputs(args: argparse.Namespace) -> Dict[str, Path]:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "output_csv": args.output_csv or output_root / "restored_validation_with_length_smarts.csv",
        "summary_json": args.summary_json or output_root / "length_summary_smarts.json",
        "diagnostics_csv": args.diagnostics_csv or output_root / "length_diagnostics_smarts.csv",
    }


def split_fragments(value: object) -> List[str]:
    if value is None or pd.isna(value):
        return []
    normalized = str(value).replace("｜", "|").strip()
    if not normalized:
        return []
    parts = normalized.split("|")
    if any(part == "" for part in parts):
        raise ValueError(f"Encountered empty fragment token in: {value}")
    return parts


def load_mol(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        RDLogger.DisableLog("rdApp.error")
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None
    finally:
        RDLogger.EnableLog("rdApp.error")


def count_disulfide_matches(mol: Chem.Mol) -> int:
    matches = mol.GetSubstructMatches(SS_PATTERN)
    unique_pairs = {tuple(sorted(match)) for match in matches}
    return len(unique_pairs)


def detect_peptide_type(mol: Chem.Mol) -> Dict[str, Any]:
    ss_bond_count = count_disulfide_matches(mol)
    ring_count = int(mol.GetRingInfo().NumRings())
    has_ss = ss_bond_count > 0
    has_ring = ring_count > 0
    has_free_nterm = mol.HasSubstructMatch(TERM_NH2_PATTERN)

    if has_ss:
        predicted_type = "disulfide_bridge"
    elif not has_ring:
        predicted_type = "linear"
    else:
        predicted_type = "sc_to_tail" if has_free_nterm else "head_to_tail"

    return {
        "predicted_type": predicted_type,
        "has_ss": has_ss,
        "ss_bond_count": ss_bond_count,
        "has_ring": has_ring,
        "ring_count": ring_count,
        "has_free_nterm": has_free_nterm,
    }


def count_residues(mol: Chem.Mol) -> Dict[str, Any]:
    matches = mol.GetSubstructMatches(CA_PATTERN)
    ca_indices = sorted({match[1] for match in matches})
    return {
        "peptide_length": len(ca_indices),
        "ca_match_count": len(matches),
        "unique_ca_count": len(ca_indices),
        "ca_indices": ca_indices,
    }


def peptide_info(smiles: str) -> Dict[str, Any]:
    mol = load_mol(smiles)
    if mol is None:
        return {"error": f"SMILES parse failed: {smiles}"}
    type_info = detect_peptide_type(mol)
    residue_info = count_residues(mol)
    return {
        "type": type_info["predicted_type"],
        "length": residue_info["peptide_length"],
        "smiles": smiles[:80] + ("..." if len(smiles) > 80 else ""),
    }


def batch_peptide_info(smiles_list: List[str]) -> List[Dict[str, Any]]:
    return [peptide_info(smiles) for smiles in smiles_list]


def infer_from_smiles(smiles: str) -> Dict[str, Any]:
    mol = load_mol(smiles)
    if mol is None:
        return {
            "valid_smiles": False,
            "peptide_length": None,
            "predicted_peptide_type": None,
            "ca_match_count": 0,
            "unique_ca_count": 0,
            "ss_bond_count": 0,
            "ring_count": 0,
            "has_ss": False,
            "has_ring": False,
            "has_free_nterm": False,
            "failure_reasons": [REASON_INVALID_SMILES],
        }

    type_info = detect_peptide_type(mol)
    residue_info = count_residues(mol)
    return {
        "valid_smiles": True,
        "peptide_length": residue_info["peptide_length"],
        "predicted_peptide_type": type_info["predicted_type"],
        "ca_match_count": residue_info["ca_match_count"],
        "unique_ca_count": residue_info["unique_ca_count"],
        "ss_bond_count": type_info["ss_bond_count"],
        "ring_count": type_info["ring_count"],
        "has_ss": type_info["has_ss"],
        "has_ring": type_info["has_ring"],
        "has_free_nterm": type_info["has_free_nterm"],
        "failure_reasons": [],
    }


def counter_to_json(counter: Counter) -> Dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def build_summary(
    rows: Iterable[Dict[str, Any]],
    *,
    restored_csv: Path,
    original_csv: Optional[Path],
    strict: bool,
) -> Dict[str, Any]:
    total_rows = 0
    valid_smiles_count = 0
    invalid_smiles_count = 0
    compared_to_validation_count = 0
    length_match_count = 0
    length_mismatch_count = 0
    type_match_count = 0
    type_mismatch_count = 0
    predicted_length_distribution: Counter = Counter()
    true_length_distribution: Counter = Counter()
    predicted_type_distribution: Counter = Counter()
    validation_type_distribution: Counter = Counter()
    per_cyclization: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total_rows": 0,
            "valid_smiles_count": 0,
            "invalid_smiles_count": 0,
            "compared_to_validation_count": 0,
            "length_match_count": 0,
            "length_mismatch_count": 0,
            "type_match_count": 0,
            "type_mismatch_count": 0,
            "predicted_length_distribution": Counter(),
            "true_length_distribution": Counter(),
            "predicted_type_distribution": Counter(),
        }
    )

    for row in rows:
        total_rows += 1
        cyclization = row["Cyclization"]
        cyclization_stats = per_cyclization[cyclization]
        cyclization_stats["total_rows"] += 1

        if row["valid_smiles"]:
            valid_smiles_count += 1
            cyclization_stats["valid_smiles_count"] += 1
        else:
            invalid_smiles_count += 1
            cyclization_stats["invalid_smiles_count"] += 1

        if row["peptide_length"] is not None:
            predicted_length_distribution[row["peptide_length"]] += 1
            cyclization_stats["predicted_length_distribution"][row["peptide_length"]] += 1

        if row["predicted_peptide_type"] is not None:
            predicted_type_distribution[row["predicted_peptide_type"]] += 1
            cyclization_stats["predicted_type_distribution"][row["predicted_peptide_type"]] += 1

        if row["true_length"] is not None:
            compared_to_validation_count += 1
            cyclization_stats["compared_to_validation_count"] += 1
            true_length_distribution[row["true_length"]] += 1
            cyclization_stats["true_length_distribution"][row["true_length"]] += 1
            validation_type_distribution[row["validation_cyclization"]] += 1
            if row["length_matches_validation"]:
                length_match_count += 1
                cyclization_stats["length_match_count"] += 1
            else:
                length_mismatch_count += 1
                cyclization_stats["length_mismatch_count"] += 1
            if row["type_matches_validation"]:
                type_match_count += 1
                cyclization_stats["type_match_count"] += 1
            else:
                type_mismatch_count += 1
                cyclization_stats["type_mismatch_count"] += 1

    exact_length_match_rate = (
        float(length_match_count) / float(compared_to_validation_count)
        if compared_to_validation_count
        else None
    )
    exact_type_match_rate = (
        float(type_match_count) / float(compared_to_validation_count)
        if compared_to_validation_count
        else None
    )

    serialized_per_cyclization: Dict[str, Any] = {}
    for cyclization, stats in per_cyclization.items():
        serialized_per_cyclization[cyclization] = {
            "total_rows": stats["total_rows"],
            "valid_smiles_count": stats["valid_smiles_count"],
            "invalid_smiles_count": stats["invalid_smiles_count"],
            "compared_to_validation_count": stats["compared_to_validation_count"],
            "length_match_count": stats["length_match_count"],
            "length_mismatch_count": stats["length_mismatch_count"],
            "type_match_count": stats["type_match_count"],
            "type_mismatch_count": stats["type_mismatch_count"],
            "predicted_length_distribution": counter_to_json(stats["predicted_length_distribution"]),
            "true_length_distribution": counter_to_json(stats["true_length_distribution"]),
            "predicted_type_distribution": counter_to_json(stats["predicted_type_distribution"]),
        }

    return {
        "restored_csv": str(restored_csv),
        "original_csv": str(original_csv) if original_csv else None,
        "strict": bool(strict),
        "total_rows": total_rows,
        "valid_smiles_count": valid_smiles_count,
        "invalid_smiles_count": invalid_smiles_count,
        "compared_to_validation_count": compared_to_validation_count,
        "length_match_count": length_match_count,
        "length_mismatch_count": length_mismatch_count,
        "type_match_count": type_match_count,
        "type_mismatch_count": type_mismatch_count,
        "exact_length_match_rate": exact_length_match_rate,
        "exact_type_match_rate": exact_type_match_rate,
        "predicted_length_distribution": counter_to_json(predicted_length_distribution),
        "true_length_distribution": counter_to_json(true_length_distribution),
        "predicted_type_distribution": counter_to_json(predicted_type_distribution),
        "validation_type_distribution": counter_to_json(validation_type_distribution),
        "per_cyclization": serialized_per_cyclization,
    }


def main() -> None:
    args = parse_args()
    outputs = resolve_outputs(args)

    restored_df = pd.read_csv(args.restored_csv)
    original_df = pd.read_csv(args.original_csv) if args.original_csv else None

    if original_df is not None and len(original_df) != len(restored_df):
        raise ValueError(
            f"Row count mismatch: {args.original_csv} has {len(original_df)} rows but "
            f"{args.restored_csv} has {len(restored_df)} rows."
        )

    result_rows: List[Dict[str, Any]] = []
    diagnostics_rows: List[Dict[str, Any]] = []

    for row_idx, restored_row in restored_df.iterrows():
        cyclization = str(restored_row["Cyclization"])
        smiles = restored_row["SMILES"]
        inferred = infer_from_smiles(smiles)

        true_length: Optional[int] = None
        validation_cyclization: Optional[str] = None
        if original_df is not None:
            original_row = original_df.iloc[row_idx]
            validation_cyclization = str(original_row["Cyclization"])
            if validation_cyclization != cyclization:
                raise ValueError(
                    f"Cyclization mismatch at row {row_idx}: "
                    f"{validation_cyclization} != {cyclization}"
                )
            true_length = len(split_fragments(original_row["Source_Mol"]))
            if inferred["peptide_length"] != true_length:
                inferred["failure_reasons"] = list(inferred["failure_reasons"]) + [REASON_LENGTH_MISMATCH]
            if inferred["predicted_peptide_type"] != validation_cyclization:
                inferred["failure_reasons"] = list(inferred["failure_reasons"]) + [REASON_TYPE_MISMATCH]

        failure_reasons = list(dict.fromkeys(inferred["failure_reasons"]))
        length_matches_validation = None if true_length is None else not (
            REASON_LENGTH_MISMATCH in failure_reasons
        )
        type_matches_validation = None if validation_cyclization is None else not (
            REASON_TYPE_MISMATCH in failure_reasons
        )

        result_row = {
            "Cyclization": cyclization,
            "SMILES": smiles,
            "peptide_length": inferred["peptide_length"],
            "predicted_peptide_type": inferred["predicted_peptide_type"],
            "ca_match_count": inferred["ca_match_count"],
            "unique_ca_count": inferred["unique_ca_count"],
            "ss_bond_count": inferred["ss_bond_count"],
            "ring_count": inferred["ring_count"],
            "has_ss": inferred["has_ss"],
            "has_ring": inferred["has_ring"],
            "has_free_nterm": inferred["has_free_nterm"],
            "valid_smiles": inferred["valid_smiles"],
            "true_length": true_length,
            "validation_cyclization": validation_cyclization,
            "length_matches_validation": length_matches_validation,
            "type_matches_validation": type_matches_validation,
            "failure_reasons": ";".join(failure_reasons),
        }
        result_rows.append(result_row)

        if failure_reasons:
            diagnostics_rows.append(
                {
                    "row_index": row_idx,
                    "Cyclization": cyclization,
                    "validation_cyclization": validation_cyclization,
                    "SMILES": smiles,
                    "predicted_length": inferred["peptide_length"],
                    "true_length": true_length,
                    "predicted_peptide_type": inferred["predicted_peptide_type"],
                    "ca_match_count": inferred["ca_match_count"],
                    "unique_ca_count": inferred["unique_ca_count"],
                    "ss_bond_count": inferred["ss_bond_count"],
                    "ring_count": inferred["ring_count"],
                    "has_ss": inferred["has_ss"],
                    "has_ring": inferred["has_ring"],
                    "has_free_nterm": inferred["has_free_nterm"],
                    "failure_reasons": ";".join(failure_reasons),
                }
            )

    result_df = pd.DataFrame(result_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    summary = build_summary(
        result_rows,
        restored_csv=args.restored_csv,
        original_csv=args.original_csv,
        strict=args.strict,
    )

    outputs["output_csv"].parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(outputs["output_csv"], index=False)

    outputs["diagnostics_csv"].parent.mkdir(parents=True, exist_ok=True)
    diagnostics_columns = [
        "row_index",
        "Cyclization",
        "validation_cyclization",
        "SMILES",
        "predicted_length",
        "true_length",
        "predicted_peptide_type",
        "ca_match_count",
        "unique_ca_count",
        "ss_bond_count",
        "ring_count",
        "has_ss",
        "has_ring",
        "has_free_nterm",
        "failure_reasons",
    ]
    if diagnostics_df.empty:
        pd.DataFrame(columns=diagnostics_columns).to_csv(outputs["diagnostics_csv"], index=False)
    else:
        diagnostics_df.to_csv(outputs["diagnostics_csv"], index=False)

    outputs["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_csv": str(outputs["output_csv"]),
                "summary_json": str(outputs["summary_json"]),
                "diagnostics_csv": str(outputs["diagnostics_csv"]),
                "total_rows": summary["total_rows"],
                "invalid_smiles_count": summary["invalid_smiles_count"],
                "length_mismatch_count": summary["length_mismatch_count"],
                "type_mismatch_count": summary["type_mismatch_count"],
                "exact_length_match_rate": summary["exact_length_match_rate"],
                "exact_type_match_rate": summary["exact_type_match_rate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.strict and diagnostics_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
