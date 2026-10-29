#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


REASON_INVALID_SMILES = "invalid_smiles"
REASON_AMBIGUOUS_BACKBONE = "ambiguous_backbone_count"
REASON_LENGTH_MISMATCH = "length_mismatch_vs_validation"

DEFAULT_RESTORED_CSV = Path('./data/restored_validation.csv')
DEFAULT_ORIGINAL_CSV = Path('./data/validation_original.csv')
DEFAULT_OUTPUT_ROOT = Path('results/peptide_length_validation_20260411')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate peptide length directly from full SMILES.")
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
        "output_csv": args.output_csv or output_root / "restored_validation_with_length.csv",
        "summary_json": args.summary_json or output_root / "length_summary.json",
        "diagnostics_csv": args.diagnostics_csv or output_root / "length_diagnostics.csv",
    }


def load_type_oracle(repo_root: Path) -> Any:
    type_oracle_path = repo_root / "model" / "type_oracle.py"
    if not type_oracle_path.is_file():
        raise FileNotFoundError(
            "The legacy type_oracle.py source is unavailable in this snapshot. "
            "Use smiles2peptide_batch.py or parse_smiles2peptide.py for residue statistics; "
            "do not substitute a different topology algorithm silently."
        )
    spec = importlib.util.spec_from_file_location("type_oracle", type_oracle_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load type oracle from {type_oracle_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def build_residue_backbone_graph(mol: Any, type_oracle: Any) -> Dict[str, Any]:
    analysis = type_oracle._analyze_backbone(mol)
    alpha_carbons = set(analysis["alpha_carbons"])
    graph = {atom_idx: set() for atom_idx in alpha_carbons}

    carbonyl_to_alpha: Dict[int, set[int]] = {}
    n_to_alpha: Dict[int, set[int]] = {}
    for atom_idx in alpha_carbons:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if type_oracle._is_backbone_carbonyl(neighbor):
                carbonyl_to_alpha.setdefault(neighbor.GetIdx(), set()).add(atom_idx)
            if neighbor.GetAtomicNum() == 7:
                n_to_alpha.setdefault(neighbor.GetIdx(), set()).add(atom_idx)

    edge_count = 0
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        carbonyl_atom = None
        nitrogen_atom = None

        if type_oracle._is_backbone_carbonyl(begin_atom) and end_atom.GetAtomicNum() == 7:
            carbonyl_atom = begin_atom
            nitrogen_atom = end_atom
        elif type_oracle._is_backbone_carbonyl(end_atom) and begin_atom.GetAtomicNum() == 7:
            carbonyl_atom = end_atom
            nitrogen_atom = begin_atom

        if carbonyl_atom is None or nitrogen_atom is None:
            continue

        previous_residues = carbonyl_to_alpha.get(carbonyl_atom.GetIdx(), set())
        next_residues = n_to_alpha.get(nitrogen_atom.GetIdx(), set())
        for previous_idx in previous_residues:
            for next_idx in next_residues:
                if previous_idx == next_idx:
                    continue
                if next_idx not in graph[previous_idx]:
                    graph[previous_idx].add(next_idx)
                    graph[next_idx].add(previous_idx)
                    edge_count += 1

    components: List[set[int]] = []
    visited: set[int] = set()
    for atom_idx in alpha_carbons:
        if atom_idx in visited:
            continue
        stack = [atom_idx]
        visited.add(atom_idx)
        component: set[int] = set()
        while stack:
            current_idx = stack.pop()
            component.add(current_idx)
            for neighbor_idx in graph[current_idx]:
                if neighbor_idx in visited:
                    continue
                visited.add(neighbor_idx)
                stack.append(neighbor_idx)
        components.append(component)

    main_component = max(components, key=len) if components else set()
    main_component_edges = sum(len(graph[atom_idx] & main_component) for atom_idx in main_component) // 2

    return {
        "analysis": analysis,
        "graph": graph,
        "components": components,
        "main_component": main_component,
        "main_component_size": len(main_component),
        "main_component_edges": main_component_edges,
        "edge_count": edge_count,
    }


def infer_length_from_smiles(smiles: str, type_oracle: Any) -> Dict[str, Any]:
    mol = type_oracle._load_mol(smiles)
    if mol is None:
        return {
            "valid_smiles": False,
            "peptide_length": None,
            "alpha_carbon_count": 0,
            "backbone_carbonyl_count": 0,
            "backbone_n_count": 0,
            "amide_bond_count": 0,
            "residue_graph_node_count": 0,
            "residue_graph_edge_count": 0,
            "residue_graph_component_count": 0,
            "failure_reasons": [REASON_INVALID_SMILES],
        }

    residue_graph = build_residue_backbone_graph(mol, type_oracle)
    analysis = residue_graph["analysis"]
    alpha_carbon_count = len(analysis["alpha_carbons"])
    backbone_carbonyl_count = len(analysis["backbone_carbonyls"])
    backbone_n_count = len(analysis["backbone_n"])
    amide_bond_count = int(analysis["amide_bond_count"])
    residue_graph_node_count = int(residue_graph["main_component_size"])
    residue_graph_edge_count = int(residue_graph["main_component_edges"])
    residue_graph_component_count = len(residue_graph["components"])

    failure_reasons: List[str] = []
    if residue_graph_node_count == 0 or residue_graph_edge_count == 0:
        failure_reasons.append(REASON_AMBIGUOUS_BACKBONE)

    return {
        "valid_smiles": True,
        "peptide_length": residue_graph_node_count,
        "alpha_carbon_count": alpha_carbon_count,
        "backbone_carbonyl_count": backbone_carbonyl_count,
        "backbone_n_count": backbone_n_count,
        "amide_bond_count": amide_bond_count,
        "residue_graph_node_count": residue_graph_node_count,
        "residue_graph_edge_count": residue_graph_edge_count,
        "residue_graph_component_count": residue_graph_component_count,
        "failure_reasons": failure_reasons,
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
    ambiguous_backbone_count = 0
    compared_to_validation_count = 0
    length_match_count = 0
    length_mismatch_count = 0
    predicted_distribution: Counter = Counter()
    true_distribution: Counter = Counter()
    per_cyclization: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total_rows": 0,
            "valid_smiles_count": 0,
            "invalid_smiles_count": 0,
            "ambiguous_backbone_count": 0,
            "compared_to_validation_count": 0,
            "length_match_count": 0,
            "length_mismatch_count": 0,
            "predicted_length_distribution": Counter(),
            "true_length_distribution": Counter(),
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

        if REASON_AMBIGUOUS_BACKBONE in row["failure_reasons"]:
            ambiguous_backbone_count += 1
            cyclization_stats["ambiguous_backbone_count"] += 1

        if row["peptide_length"] is not None:
            predicted_distribution[row["peptide_length"]] += 1
            cyclization_stats["predicted_length_distribution"][row["peptide_length"]] += 1

        if row["true_length"] is not None:
            compared_to_validation_count += 1
            true_distribution[row["true_length"]] += 1
            cyclization_stats["compared_to_validation_count"] += 1
            cyclization_stats["true_length_distribution"][row["true_length"]] += 1
            if row["length_matches_validation"]:
                length_match_count += 1
                cyclization_stats["length_match_count"] += 1
            else:
                length_mismatch_count += 1
                cyclization_stats["length_mismatch_count"] += 1

    exact_match_rate = (
        float(length_match_count) / float(compared_to_validation_count)
        if compared_to_validation_count
        else None
    )

    serialized_per_cyclization: Dict[str, Any] = {}
    for cyclization, stats in per_cyclization.items():
        serialized_per_cyclization[cyclization] = {
            "total_rows": stats["total_rows"],
            "valid_smiles_count": stats["valid_smiles_count"],
            "invalid_smiles_count": stats["invalid_smiles_count"],
            "ambiguous_backbone_count": stats["ambiguous_backbone_count"],
            "compared_to_validation_count": stats["compared_to_validation_count"],
            "length_match_count": stats["length_match_count"],
            "length_mismatch_count": stats["length_mismatch_count"],
            "predicted_length_distribution": counter_to_json(stats["predicted_length_distribution"]),
            "true_length_distribution": counter_to_json(stats["true_length_distribution"]),
        }

    return {
        "restored_csv": str(restored_csv),
        "original_csv": str(original_csv) if original_csv else None,
        "strict": bool(strict),
        "total_rows": total_rows,
        "valid_smiles_count": valid_smiles_count,
        "invalid_smiles_count": invalid_smiles_count,
        "ambiguous_backbone_count": ambiguous_backbone_count,
        "compared_to_validation_count": compared_to_validation_count,
        "length_match_count": length_match_count,
        "length_mismatch_count": length_mismatch_count,
        "exact_match_rate": exact_match_rate,
        "predicted_length_distribution": counter_to_json(predicted_distribution),
        "true_length_distribution": counter_to_json(true_distribution),
        "per_cyclization": serialized_per_cyclization,
    }


def main() -> None:
    args = parse_args()
    outputs = resolve_outputs(args)

    repo_root = Path(__file__).resolve().parents[2]
    type_oracle = load_type_oracle(repo_root)

    restored_df = pd.read_csv(args.restored_csv)
    original_df = pd.read_csv(args.original_csv) if args.original_csv else None

    if original_df is not None:
        if len(original_df) != len(restored_df):
            raise ValueError(
                f"Row count mismatch: {args.original_csv} has {len(original_df)} rows but "
                f"{args.restored_csv} has {len(restored_df)} rows."
            )

    result_rows: List[Dict[str, Any]] = []
    diagnostics_rows: List[Dict[str, Any]] = []

    for row_idx, restored_row in restored_df.iterrows():
        cyclization = restored_row["Cyclization"]
        smiles = restored_row["SMILES"]
        inferred = infer_length_from_smiles(smiles, type_oracle)

        true_length: Optional[int] = None
        validation_cyclization: Optional[str] = None
        if original_df is not None:
            original_row = original_df.iloc[row_idx]
            validation_cyclization = str(original_row["Cyclization"])
            if validation_cyclization != str(cyclization):
                raise ValueError(
                    f"Cyclization mismatch at row {row_idx}: "
                    f"{validation_cyclization} != {cyclization}"
                )
            true_length = len(split_fragments(original_row["Source_Mol"]))
            if inferred["peptide_length"] != true_length:
                inferred["failure_reasons"] = list(inferred["failure_reasons"]) + [REASON_LENGTH_MISMATCH]

        failure_reasons = list(dict.fromkeys(inferred["failure_reasons"]))
        length_matches_validation = None if true_length is None else not (REASON_LENGTH_MISMATCH in failure_reasons)

        result_row = {
            "Cyclization": cyclization,
            "SMILES": smiles,
            "peptide_length": inferred["peptide_length"],
            "alpha_carbon_count": inferred["alpha_carbon_count"],
            "backbone_carbonyl_count": inferred["backbone_carbonyl_count"],
            "backbone_n_count": inferred["backbone_n_count"],
            "amide_bond_count": inferred["amide_bond_count"],
            "residue_graph_node_count": inferred["residue_graph_node_count"],
            "residue_graph_edge_count": inferred["residue_graph_edge_count"],
            "residue_graph_component_count": inferred["residue_graph_component_count"],
            "valid_smiles": inferred["valid_smiles"],
            "true_length": true_length,
            "length_matches_validation": length_matches_validation,
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
                    "alpha_carbon_count": inferred["alpha_carbon_count"],
                    "backbone_carbonyl_count": inferred["backbone_carbonyl_count"],
                    "backbone_n_count": inferred["backbone_n_count"],
                    "amide_bond_count": inferred["amide_bond_count"],
                    "residue_graph_node_count": inferred["residue_graph_node_count"],
                    "residue_graph_edge_count": inferred["residue_graph_edge_count"],
                    "residue_graph_component_count": inferred["residue_graph_component_count"],
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
        "alpha_carbon_count",
        "backbone_carbonyl_count",
        "backbone_n_count",
        "amide_bond_count",
        "residue_graph_node_count",
        "residue_graph_edge_count",
        "residue_graph_component_count",
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
                "ambiguous_backbone_count": summary["ambiguous_backbone_count"],
                "length_mismatch_count": summary["length_mismatch_count"],
                "exact_match_rate": summary["exact_match_rate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.strict and diagnostics_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
