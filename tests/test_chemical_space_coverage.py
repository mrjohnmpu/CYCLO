from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

from evaluation.metrics.chemical_space_coverage import (
    canonical_counts,
    canonicalize_smiles,
    coverage_grid,
    ecdf_table,
    fpsim2_available,
    morgan_fingerprint,
    nearest_neighbor_search,
    ratio_novelty,
    scaffold_smiles,
)


class ChemicalSpaceCoverageTests(unittest.TestCase):
    def test_canonical_counts_preserve_multiplicity(self) -> None:
        counts, invalid = canonical_counts(["CCO", "OCC", "invalid"])
        self.assertEqual(invalid, 1)
        self.assertEqual(counts[canonicalize_smiles("CCO")], 2)

    def test_morgan_matches_current_moses_definition(self) -> None:
        from utils.release_paths import add_moses_path
        add_moses_path()
        try:
            from moses.metrics.utils import fingerprint
        except ImportError:
            self.skipTest("MOSES installation/source tree is unavailable")

        smiles = "CC1CCCCC1N"
        ours = np.asarray(morgan_fingerprint(smiles), dtype=np.uint8)
        current = fingerprint(smiles, fp_type="morgan")
        np.testing.assert_array_equal(ours, current)

    def test_scaffold_definitions(self) -> None:
        benzene_murcko, benzene_generic = scaffold_smiles("c1ccccc1")
        toluene_murcko, toluene_generic = scaffold_smiles("Cc1ccccc1")
        self.assertEqual(benzene_murcko, toluene_murcko)
        self.assertEqual(benzene_generic, toluene_generic)
        self.assertEqual(scaffold_smiles("CCO"), (None, None))

    def test_threshold_direction_and_strictness(self) -> None:
        train_similarity = np.array([0.79, 0.80, 0.95])
        self.assertAlmostEqual(float((train_similarity < 0.80).mean()), 1 / 3)
        reference_similarity = np.array([0.39, 0.40, 0.80])
        grid = coverage_grid(reference_similarity, "M", "M", "head_to_tail")
        value_04 = grid.loc[np.isclose(grid["threshold"], 0.4), "coverage"].item()
        value_08 = grid.loc[np.isclose(grid["threshold"], 0.8), "coverage"].item()
        self.assertAlmostEqual(value_04, 2 / 3)
        self.assertAlmostEqual(value_08, 1 / 3)

    def test_ecdf_uses_unique_molecules(self) -> None:
        frame = pd.DataFrame(
            {
                "canonical_smiles": ["CC", "CCC", "CCCC"],
                "nearest_train_similarity": [0.8, 0.2, 0.5],
            }
        )
        result = ecdf_table(frame, "M", "M", "head_to_tail")
        self.assertEqual(result["nearest_train_similarity"].tolist(), [0.2, 0.5, 0.8])
        np.testing.assert_allclose(result["ecdf"], [1 / 3, 2 / 3, 1.0])

    def test_exact_novelty_is_set_based(self) -> None:
        self.assertAlmostEqual(ratio_novelty({"A", "B", "C"}, {"B", "D"}), 2 / 3)

    def test_reverse_coverage_is_not_snn(self) -> None:
        generated = pd.DataFrame({"canonical_smiles": ["CCO"]})
        reference = pd.DataFrame({"canonical_smiles": ["CCO", "c1ccccc1"]})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated_to_reference = nearest_neighbor_search(
                generated,
                reference,
                cache_dir=root / "g_to_r",
                backend="rdkit",
                n_jobs=1,
                chunk_size=2,
                radius=2,
                n_bits=1024,
                force=True,
            )
            reference_to_generated = nearest_neighbor_search(
                reference,
                generated,
                cache_dir=root / "r_to_g",
                backend="rdkit",
                n_jobs=1,
                chunk_size=2,
                radius=2,
                n_bits=1024,
                force=True,
            )
        self.assertAlmostEqual(generated_to_reference["similarity"].mean(), 1.0)
        self.assertLess(reference_to_generated["similarity"].mean(), 1.0)

    @unittest.skipUnless(fpsim2_available(), "FPSim2 is not installed")
    def test_fpsim2_matches_rdkit_top_one(self) -> None:
        target = pd.DataFrame(
            {"canonical_smiles": ["CCO", "c1ccccc1", "CC(=O)N", "C1CCCCC1"]}
        )
        queries = pd.DataFrame({"canonical_smiles": ["CCCO", "c1ccncc1", "CC(=O)NC"]})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rdkit_result = nearest_neighbor_search(
                queries,
                target,
                cache_dir=root / "rdkit",
                backend="rdkit",
                n_jobs=1,
                chunk_size=3,
                radius=2,
                n_bits=1024,
                force=True,
            )
            fpsim2_result = nearest_neighbor_search(
                queries,
                target,
                cache_dir=root / "fpsim2",
                backend="fpsim2",
                n_jobs=1,
                chunk_size=3,
                radius=2,
                n_bits=1024,
                force=True,
            )
        np.testing.assert_allclose(
            rdkit_result["similarity"], fpsim2_result["similarity"], atol=1e-7
        )
        for row in fpsim2_result.itertuples(index=False):
            query_fp = morgan_fingerprint(row.canonical_smiles)
            target_fp = morgan_fingerprint(row.nearest_target_smiles)
            self.assertAlmostEqual(
                DataStructs.TanimotoSimilarity(query_fp, target_fp), row.similarity, places=7
            )


if __name__ == "__main__":
    unittest.main()
