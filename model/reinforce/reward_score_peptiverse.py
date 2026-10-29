from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
from utils.release_paths import peptiverse_root
THIRD_PARTY_ROOT = peptiverse_root().parent
if str(THIRD_PARTY_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_ROOT))

DEFAULT_WEIGHT_ROOT = peptiverse_root()
DEFAULT_MANIFEST_PATH = DEFAULT_WEIGHT_ROOT / "best_models.txt"


class RewardTerm:
    name = "base_term"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = float(weight)

    def __call__(self, smiles_list: list[str], ctx: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError


class PermeabilityTerm(RewardTerm):
    name = "permeability_pampa"

    def __init__(self, weight: float = 1.0, threshold: float = -6.0, slope: float = 1.5) -> None:
        super().__init__(weight=weight)
        self.threshold = float(threshold)
        self.slope = float(slope)

    def __call__(self, smiles_list: list[str], ctx: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        predictor = ctx["peptiverse_predictor"]
        preds: list[float] = []
        valid_count = 0

        for smiles in smiles_list:
            try:
                output = predictor.predict_property("permeability_pampa", mode="smiles", input_str=smiles)
                value = output["score"]
                preds.append(float(value[0] if isinstance(value, (list, np.ndarray)) else value))
                valid_count += 1
            except Exception:
                preds.append(-10.0)

        preds_arr = np.array(preds, dtype=np.float32)
        normalized = 1.0 / (1.0 + np.exp(-self.slope * (preds_arr - self.threshold)))
        return normalized.astype(np.float32), {
            "perm_raw_mean": float(np.mean(preds_arr)) if len(preds_arr) else np.nan,
            "perm_valid_rate": float(valid_count / len(smiles_list)) if smiles_list else 0.0,
        }


class HemolysisTerm(RewardTerm):
    name = "hemolysis_safe"

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight=weight)

    def __call__(self, smiles_list: list[str], ctx: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        predictor = ctx["peptiverse_predictor"]
        safe_probs: list[float] = []

        for smiles in smiles_list:
            try:
                output = predictor.predict_property("hemolysis", mode="smiles", input_str=smiles)
                prob = output["score"]
                prob_value = float(prob[0] if isinstance(prob, (list, np.ndarray)) else prob)
                safe_probs.append(1.0 - prob_value)
            except Exception:
                safe_probs.append(0.0)

        scores = np.array(safe_probs, dtype=np.float32)
        return scores, {
            "hemo_safe_rate": float(np.mean(scores > 0.5)) if len(scores) else 0.0,
        }


class ToxicityTerm(RewardTerm):
    name = "toxicity_safe"

    def __init__(self, weight: float = 0.0) -> None:
        super().__init__(weight=weight)

    def __call__(self, smiles_list: list[str], ctx: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        predictor = ctx["peptiverse_predictor"]
        safe_probs: list[float] = []

        for smiles in smiles_list:
            try:
                output = predictor.predict_property("toxicity", mode="smiles", input_str=smiles)
                prob = output["score"]
                prob_value = float(prob[0] if isinstance(prob, (list, np.ndarray)) else prob)
                safe_probs.append(1.0 - prob_value)
            except Exception:
                safe_probs.append(0.0)

        scores = np.array(safe_probs, dtype=np.float32)
        return scores, {
            "tox_safe_mean": float(np.mean(scores)) if len(scores) else np.nan,
        }


class BindingAffinityTerm(RewardTerm):
    name = "binding_affinity"

    def __init__(self, weight: float = 0.0, center: float = 8.0, slope: float = 1.0) -> None:
        super().__init__(weight=weight)
        self.center = float(center)
        self.slope = float(slope)

    def __call__(self, smiles_list: list[str], ctx: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        predictor = ctx["peptiverse_predictor"]
        target_seq = ctx.get("target_protein")
        if not target_seq:
            zeros = np.zeros(len(smiles_list), dtype=np.float32)
            return zeros, {"affinity_skipped": 1.0}

        raw_scores: list[float] = []
        for smiles in smiles_list:
            try:
                output = predictor.predict_binding_affinity(mode="smiles", target_seq=target_seq, binder_str=smiles)
                value = output["affinity"]
                raw_scores.append(float(value[0] if isinstance(value, (list, np.ndarray)) else value))
            except Exception:
                raw_scores.append(2.0)

        raw_arr = np.array(raw_scores, dtype=np.float32)
        normalized = 1.0 / (1.0 + np.exp(-self.slope * (raw_arr - self.center)))
        return normalized.astype(np.float32), {
            "affinity_raw_mean": float(np.mean(raw_arr)) if len(raw_arr) else np.nan,
            "affinity_norm_mean": float(np.mean(normalized)) if len(normalized) else np.nan,
        }


class RewardScorer:
    def __init__(self, terms: list[RewardTerm], score_type: str = "sum") -> None:
        self.terms = terms
        self.score_type = score_type

    def score(self, smiles_list: list[str], ctx: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
        term_values: dict[str, np.ndarray] = {}
        meta_all: dict[str, Any] = {}

        for term in self.terms:
            values, meta = term(smiles_list, ctx)
            term_values[term.name] = values
            meta_all.update(meta or {})

        weighted_sum = np.zeros(len(smiles_list), dtype=np.float32)
        for term in self.terms:
            if term.weight != 0:
                weighted_sum += term.weight * term_values[term.name]

        avg_stats: dict[str, Any] = {"TotalScore": float(np.mean(weighted_sum)) if len(weighted_sum) else 0.0}
        for key, values in term_values.items():
            avg_stats[key] = float(np.mean(values)) if len(values) else 0.0
        avg_stats.update(meta_all)
        stats_df = pd.DataFrame(avg_stats, index=[0])
        return weighted_sum, stats_df, term_values


def load_peptiverse(manifest_path: str | Path = DEFAULT_MANIFEST_PATH, weight_root: str | Path = DEFAULT_WEIGHT_ROOT, device: str = "cuda"):
    from PeptiVerse.inference import PeptiVersePredictor

    predictor = PeptiVersePredictor(
        manifest_path=str(manifest_path),
        classifier_weight_root=str(weight_root),
        device=device,
    )
    print(f"[PeptiVerse] Loaded predictor on {device}", flush=True)
    return predictor
