from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    RDKIT_AVAILABLE = False
    logger.warning("RDKit is not available; LoRA evaluation will be limited.")

TYPE_SMARTS = {
    "disulfide_bridge": "[S;X2][S;X2]",
    "head_to_tail": "[N;R][C;R](=O)",
    "sc_to_tail": "[C;R](=O)[O,N]",
    "linear": None,
}


class PeptideEvaluator:
    def evaluate(
        self,
        generated: list[str],
        expected_type: str,
        sample_size: int = 200,
    ) -> dict[str, float | int | None]:
        if not generated:
            return {}

        metrics: dict[str, float | int | None] = {
            "total": len(generated),
            "validity": self._validity(generated),
            "uniqueness": self._uniqueness(generated),
            "diversity": self._diversity(generated, sample_size),
            "type_accuracy": self._type_accuracy(generated, expected_type),
        }
        logger.info("Evaluation metrics for %s: %s", expected_type, metrics)
        return metrics

    def compare_all_types(
        self,
        generated_by_type: dict[str, list[str]],
        sample_size: int = 200,
    ) -> dict[str, dict[str, float | int | None]]:
        return {
            peptide_type: self.evaluate(smiles_values, peptide_type, sample_size=sample_size)
            for peptide_type, smiles_values in generated_by_type.items()
        }

    def _validity(self, smiles_values: list[str]) -> float:
        if not RDKIT_AVAILABLE:
            return -1.0
        valid_count = sum(
            1
            for smiles in smiles_values
            if smiles and Chem.MolFromSmiles(smiles) is not None
        )
        return valid_count / len(smiles_values)

    def _uniqueness(self, smiles_values: list[str]) -> float:
        if not RDKIT_AVAILABLE:
            return -1.0
        valid_smiles = [
            smiles
            for smiles in smiles_values
            if smiles and Chem.MolFromSmiles(smiles) is not None
        ]
        if not valid_smiles:
            return 0.0
        return len(set(valid_smiles)) / len(valid_smiles)

    def _diversity(self, smiles_values: list[str], sample_size: int) -> float:
        if not RDKIT_AVAILABLE:
            return -1.0
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_values]
        mols = [molecule for molecule in mols if molecule is not None]
        if len(mols) < 2:
            return 0.0

        fingerprints = [
            AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
            for molecule in mols
        ]
        sampled = random.sample(fingerprints, min(sample_size, len(fingerprints)))
        similarities = [
            DataStructs.TanimotoSimilarity(sampled[i], sampled[j])
            for i in range(len(sampled))
            for j in range(i + 1, len(sampled))
        ]
        if not similarities:
            return 0.0
        return float(1.0 - np.mean(similarities))

    def _type_accuracy(self, smiles_values: list[str], expected_type: str) -> Optional[float]:
        if not RDKIT_AVAILABLE:
            return None
        if expected_type == "linear":
            matches = sum(1 for smiles in smiles_values if self._is_linear(smiles))
            return matches / len(smiles_values)

        smarts = TYPE_SMARTS.get(expected_type)
        if smarts is None:
            return None

        query = Chem.MolFromSmarts(smarts)
        matches = sum(1 for smiles in smiles_values if self._has_feature(smiles, query))
        return matches / len(smiles_values)

    @staticmethod
    def _has_feature(smiles: str, query) -> bool:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return False
        return molecule.HasSubstructMatch(query)

    @staticmethod
    def _is_linear(smiles: str) -> bool:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return False
        return molecule.GetRingInfo().NumRings() == 0
