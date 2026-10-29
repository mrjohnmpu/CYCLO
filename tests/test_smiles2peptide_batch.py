from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation.analysis.smiles2peptide_adapter import load_peptide_analyzer
from evaluation.analysis.smiles2peptide_batch import convert_dataframe


def test_adapter_imports_app():
    analyzer = load_peptide_analyzer()
    assert analyzer.__class__.__name__ == 'PeptideAnalyzer'
    assert hasattr(analyzer, 'analyze_structure')
    assert hasattr(analyzer, 'is_peptide')


def test_batch_conversion_smoke():
    fixture = Path(__file__).parent / 'fixtures' / 'smiles2peptide_head_to_tail_smoke.csv'
    df = pd.read_csv(fixture)
    analyzer = load_peptide_analyzer()
    out_df, failed_df, summary = convert_dataframe(
        df,
        smiles_column='SMILES',
        id_column=None,
        analyzer=analyzer,
        max_rows=None,
    )

    assert len(out_df) == 3
    assert set(out_df.columns) >= {
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
    }
    assert summary['total_rows'] == 3
    assert summary['status_counts']['failed'] == 1
    # The review CSV includes partial parses as well as hard failures.
    assert len(failed_df) == int((out_df['parse_status'] != 'full').sum())
    assert int((failed_df['parse_status'] == 'failed').sum()) == 1
    assert summary['resolved_residue_totals']['counting_basis'] == 'resolved residues only'

    ok_rows = out_df[out_df['parse_status'] != 'failed']
    assert not ok_rows.empty
    assert ok_rows['peptide_length'].notna().all()
    for _, row in ok_rows.iterrows():
        residues = json.loads(row['residues'])
        detected_uaas = json.loads(row['detected_uaas'])
        assert int(row['peptide_length']) == len(residues)
        assert int(row['n_natural_resolved']) + int(row['n_non_natural_resolved']) == len(residues)
        assert int(row['n_detected_uaas']) == len(detected_uaas)
        assert row['three_letter_sequence']
        assert row['one_letter_sequence']
