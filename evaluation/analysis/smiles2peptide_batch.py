#!/usr/bin/env python3
"""Batch converter for SMILES2PEPTIDE-generated peptide SMILES."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rdkit import RDLogger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from smiles2peptide_adapter import DEFAULT_PROJECT_ROOT, load_peptide_analyzer

DEFAULT_INPUT_CSV = Path('./output/generated/lora/gen_results_head_to_tail_30000.csv')
DEFAULT_OUTPUT_DIR = Path('results/smiles2peptide_head_to_tail_30000')

CANONICAL_NATURAL_RESIDUES = {
    'Ala', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His', 'Ile', 'Lys', 'Leu',
    'Met', 'Asn', 'Pro', 'Gln', 'Arg', 'Ser', 'Thr', 'Val', 'Trp', 'Tyr',
}
CANONICAL_NATURAL_RESIDUE_TOKENS = CANONICAL_NATURAL_RESIDUES | {residue.lower() for residue in CANONICAL_NATURAL_RESIDUES}

OUTPUT_COLUMNS = [
    'mol_id',
    'input_smiles',
    'three_letter_sequence',
    'one_letter_sequence',
    'is_cyclic',
    'residues',
    'peptide_length',
    'detected_uaas',
    'n_detected_uaas',
    'n_natural_resolved',
    'n_non_natural_resolved',
    'frac_natural_resolved',
    'frac_non_natural_resolved',
    'parse_status',
    'notes',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert peptide SMILES into amino-acid sequences using SMILES2PEPTIDE logic.'
    )
    parser.add_argument('--input-csv', type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--project-root', type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument('--smiles-column', type=str, default='SMILES')
    parser.add_argument('--id-column', type=str, default=None)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--strict', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        for path in output_dir.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                import shutil
                shutil.rmtree(path)
    output_dir.mkdir(parents=True, exist_ok=True)


def pick_smiles_column(df: pd.DataFrame, preferred: str) -> str:
    candidates = [preferred, 'SMILES', 'Smiles', 'smiles']
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f'No SMILES column found. Tried: {candidates}')


def normalize_smiles(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and math.isnan(value):
        return ''
    return str(value).strip()


def infer_mol_id(row: pd.Series, idx: int, id_column: Optional[str]) -> str:
    if id_column and id_column in row and pd.notna(row[id_column]):
        value = str(row[id_column]).strip()
        if value:
            return value
    return f'mol_{idx:07d}'


def summarize_notes(notes: List[str]) -> str:
    cleaned = [note.strip() for note in notes if note and note.strip()]
    return '; '.join(cleaned)


def ordered_unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def classify_residues(residues: List[Any]) -> Dict[str, Any]:
    cleaned = [str(residue).strip() for residue in residues if str(residue).strip()]
    natural = [residue for residue in cleaned if residue in CANONICAL_NATURAL_RESIDUE_TOKENS]
    non_natural = [residue for residue in cleaned if residue not in CANONICAL_NATURAL_RESIDUE_TOKENS]
    detected_uaas = ordered_unique(non_natural)
    total = len(cleaned)
    if total:
        frac_natural = len(natural) / total
        frac_non_natural = len(non_natural) / total
        peptide_length = total
    else:
        frac_natural = pd.NA
        frac_non_natural = pd.NA
        peptide_length = pd.NA
    return {
        'resolved_residues': cleaned,
        'peptide_length': peptide_length,
        'detected_uaas': json.dumps(detected_uaas, ensure_ascii=False),
        'n_detected_uaas': len(detected_uaas),
        'n_natural_resolved': len(natural),
        'n_non_natural_resolved': len(non_natural),
        'frac_natural_resolved': frac_natural,
        'frac_non_natural_resolved': frac_non_natural,
    }


def failed_result(notes: str, is_cyclic: bool = False) -> Dict[str, Any]:
    return {
        'three_letter_sequence': '',
        'one_letter_sequence': '',
        'is_cyclic': is_cyclic,
        'residues': '[]',
        'peptide_length': pd.NA,
        'detected_uaas': '[]',
        'n_detected_uaas': 0,
        'n_natural_resolved': 0,
        'n_non_natural_resolved': 0,
        'frac_natural_resolved': pd.NA,
        'frac_non_natural_resolved': pd.NA,
        'parse_status': 'failed',
        'notes': notes,
    }


def analyze_smiles(smiles: str, analyzer: Any) -> Dict[str, Any]:
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')

    if not smiles:
        return failed_result('empty_smiles')

    try:
        peptide_like = bool(analyzer.is_peptide(smiles))
    except Exception as exc:  # pragma: no cover
        return failed_result(f'is_peptide_error:{type(exc).__name__}')

    if not peptide_like:
        return failed_result('not_peptide_like')

    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            analysis = analyzer.analyze_structure(smiles, verbose=True)
    except Exception as exc:
        return failed_result(f'analysis_error:{type(exc).__name__}')

    residues = list(analysis.get('residues', []))
    residue_stats = classify_residues(residues)
    three_letter = str(analysis.get('three_letter', ''))
    one_letter = str(analysis.get('one_letter', ''))
    is_cyclic = bool(analysis.get('is_cyclic', False))
    details = str(analysis.get('details', ''))
    warning_lines = [line.strip() for line in details.splitlines() if line.strip().startswith('Warning:')]

    parse_status = 'full'
    notes: List[str] = []
    if warning_lines:
        parse_status = 'partial'
        notes.append(f'warnings={len(warning_lines)}')
        notes.append(warning_lines[0])
    elif not residues:
        parse_status = 'partial'
        notes.append('no_residues_resolved')

    if parse_status == 'partial' and not notes:
        notes.append('partial_parse')

    return {
        'three_letter_sequence': three_letter,
        'one_letter_sequence': one_letter,
        'is_cyclic': is_cyclic,
        'residues': json.dumps(residue_stats['resolved_residues'], ensure_ascii=False),
        'peptide_length': residue_stats['peptide_length'],
        'detected_uaas': residue_stats['detected_uaas'],
        'n_detected_uaas': residue_stats['n_detected_uaas'],
        'n_natural_resolved': residue_stats['n_natural_resolved'],
        'n_non_natural_resolved': residue_stats['n_non_natural_resolved'],
        'frac_natural_resolved': residue_stats['frac_natural_resolved'],
        'frac_non_natural_resolved': residue_stats['frac_non_natural_resolved'],
        'parse_status': parse_status,
        'notes': summarize_notes(notes),
    }


def convert_dataframe(
    df: pd.DataFrame,
    *,
    smiles_column: str,
    id_column: Optional[str],
    analyzer: Any,
    max_rows: Optional[int],
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if max_rows is not None:
        df = df.head(max_rows).copy()
    else:
        df = df.copy()

    rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    status_counts = {'full': 0, 'partial': 0, 'failed': 0}
    length_values: List[int] = []
    cyclic_count = 0
    total_natural = 0
    total_non_natural = 0
    total_resolved = 0
    uaa_counter: Counter[str] = Counter()

    for idx, row in df.iterrows():
        smiles = normalize_smiles(row.get(smiles_column, ''))
        mol_id = infer_mol_id(row, int(idx), id_column)
        parsed = analyze_smiles(smiles, analyzer)
        result = {
            'mol_id': mol_id,
            'input_smiles': smiles,
            **parsed,
        }
        rows.append(result)

        status_counts[result['parse_status']] += 1
        if result['parse_status'] != 'failed' and pd.notna(result['peptide_length']):
            length_values.append(int(result['peptide_length']))
        if bool(result['is_cyclic']):
            cyclic_count += 1
        total_natural += int(result['n_natural_resolved'])
        total_non_natural += int(result['n_non_natural_resolved'])
        total_resolved += int(result['n_natural_resolved']) + int(result['n_non_natural_resolved'])
        detected_uaas = json.loads(result['detected_uaas']) if result['detected_uaas'] else []
        uaa_counter.update(detected_uaas)
        if result['parse_status'] != 'full':
            failure_rows.append(result)

    output_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    failure_df = pd.DataFrame(failure_rows, columns=OUTPUT_COLUMNS)
    summary: Dict[str, Any] = {
        'total_rows': int(len(output_df)),
        'status_counts': status_counts,
        'cyclic_count': int(cyclic_count),
        'acyclic_count': int(len(output_df) - cyclic_count),
        'length_summary': {
            'min': int(min(length_values)) if length_values else None,
            'max': int(max(length_values)) if length_values else None,
            'mean': float(sum(length_values) / len(length_values)) if length_values else None,
            'median': float(pd.Series(length_values).median()) if length_values else None,
        },
        'resolved_residue_totals': {
            'counting_basis': 'resolved residues only',
            'total_resolved_residues': int(total_resolved),
            'total_natural_residues': int(total_natural),
            'total_non_natural_residues': int(total_non_natural),
            'frac_natural_resolved_global': (float(total_natural / total_resolved) if total_resolved else None),
            'frac_non_natural_resolved_global': (float(total_non_natural / total_resolved) if total_resolved else None),
        },
        'detected_uaa_counts': dict(uaa_counter.most_common()),
    }
    return output_df, failure_df, summary


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir, args.overwrite)

    df = pd.read_csv(args.input_csv)
    smiles_column = pick_smiles_column(df, args.smiles_column)
    analyzer = load_peptide_analyzer(args.project_root)

    output_df, failure_df, summary = convert_dataframe(
        df,
        smiles_column=smiles_column,
        id_column=args.id_column,
        analyzer=analyzer,
        max_rows=args.max_rows,
    )

    output_csv = args.output_dir / 'smiles2peptide_sequences.csv'
    failed_csv = args.output_dir / 'smiles2peptide_non_full.csv'
    summary_json = args.output_dir / 'smiles2peptide_summary.json'
    residue_summary_csv = args.output_dir / 'smiles2peptide_global_residue_stats.csv'

    output_df.to_csv(output_csv, index=False)
    failure_df.to_csv(failed_csv, index=False)
    pd.DataFrame([
        {
            **summary['resolved_residue_totals'],
            'total_rows': summary['total_rows'],
            'full_rows': summary['status_counts']['full'],
            'partial_rows': summary['status_counts']['partial'],
            'failed_rows': summary['status_counts']['failed'],
        }
    ]).to_csv(residue_summary_csv, index=False)
    summary.update(
        {
            'input_csv': str(args.input_csv),
            'output_csv': str(output_csv),
            'failed_csv': str(failed_csv),
            'residue_summary_csv': str(residue_summary_csv),
            'project_root': str(args.project_root),
            'smiles_column': smiles_column,
            'id_column': args.id_column,
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Wrote: {output_csv}')
    print(f'Wrote: {failed_csv}')
    print(f'Wrote: {residue_summary_csv}')
    print(f'Wrote: {summary_json}')


if __name__ == '__main__':
    main()
