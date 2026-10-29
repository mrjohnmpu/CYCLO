from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
import torch

from .mol_cache import DatasetCache


TEST2_ROOT = Path(__file__).resolve().parents[2]
if str(TEST2_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST2_ROOT))


def _resolve_peptiverse_root() -> Path:
    from utils.release_paths import peptiverse_root
    return peptiverse_root()


def _resolve_peptiverse_helper() -> Path:
    return TEST2_ROOT / "model" / "reinforce" / "reward_score_peptiverse.py"


PEPTIVERSE_ROOT = _resolve_peptiverse_root()
PEPTIVERSE_HELPER = _resolve_peptiverse_helper()
PEPTIVERSE_INFERENCE = PEPTIVERSE_ROOT / "inference.py"
DEFAULT_MANIFEST_PATH = PEPTIVERSE_ROOT / "best_models.txt"
DEFAULT_WEIGHT_ROOT = PEPTIVERSE_ROOT


CLASSIFICATION_TASKS = {
    "hemolysis": {"predictor_key": "hemolysis", "positive_name": "Non-hemolytic", "positive_from_label": "invert"},
    "non_fouling": {"predictor_key": "nf", "positive_name": "Non-fouling", "positive_from_label": "identity"},
    "toxicity": {"predictor_key": "toxicity", "positive_name": "Non-toxic", "positive_from_label": "invert"},
}

REGRESSION_TASKS = {
    "half_life": {"predictor_key": "halflife", "display_name": "Half-life"},
    "PAMPA": {"predictor_key": "permeability_pampa", "display_name": "PAMPA"},
    "Caco-2": {"predictor_key": "permeability_caco2", "display_name": "Caco-2"},
}


def load_peptiverse_inference_module() -> Any:
    spec = importlib.util.spec_from_file_location("peptiverse_inference", PEPTIVERSE_INFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load PeptiVerse inference module from {PEPTIVERSE_INFERENCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_peptiverse_helper() -> Any:
    spec = importlib.util.spec_from_file_location("reward_score_peptiverse", PEPTIVERSE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load PeptiVerse helper from {PEPTIVERSE_HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class PeptiVerseBatchPredictor:
    predictor: Any

    @classmethod
    def load(
        cls,
        *,
        manifest_path: Path = DEFAULT_MANIFEST_PATH,
        weight_root: Path = DEFAULT_WEIGHT_ROOT,
        device: str = "cuda",
    ) -> "PeptiVerseBatchPredictor":
        predictor = SmilesOnlyPeptiVersePredictor(
            manifest_path=manifest_path,
            weight_root=weight_root,
            device=device,
        )
        return cls(predictor=predictor)

    def predict_property(self, predictor_key: str, smiles: str) -> dict[str, Any]:
        return self.predictor.predict_property(predictor_key, input_str=smiles)

    def clear_runtime_cache(self) -> None:
        self.predictor.clear_runtime_cache()


class SmilesOnlyPeptiVersePredictor:
    def __init__(
        self,
        *,
        manifest_path: Path,
        weight_root: Path,
        device: str = "cuda",
        clm_name: str = "aaronfeller/PeptideCLM-23M-all",
        smiles_vocab: str = "tokenizer/new_vocab.txt",
        smiles_splits: str = "tokenizer/new_splits.txt",
    ) -> None:
        self.module = load_peptiverse_inference_module()
        self.root = Path(weight_root)
        self.training_root = self.root / "training_classifiers"
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.manifest = self.module.read_best_manifest_csv(manifest_path)
        self.smiles_embedder = self.module.SMILESEmbedder(
            self.device,
            vocab_path=str(self.root / smiles_vocab),
            splits_path=str(self.root / smiles_splits),
            clm_name=clm_name,
            use_cache=False,
        )
        self.models: dict[tuple[str, str], Any] = {}
        self.meta: dict[tuple[str, str], dict[str, Any]] = {}
        self._load_selected_smiles_models(
            [
                task["predictor_key"]
                for task in [*CLASSIFICATION_TASKS.values(), *REGRESSION_TASKS.values()]
            ]
        )

    def _resolve_dir(self, prop_key: str, model_name: str) -> Path:
        disk_prop = "half_life" if prop_key == "halflife" else prop_key
        base = self.training_root / disk_prop

        if prop_key == "halflife" and model_name == "xgb":
            special_dir = base / "xgb_smiles"
            if special_dir.exists():
                return special_dir

        candidates = [
            base / f"{model_name}_smiles",
            base / model_name,
        ]
        for model_dir in candidates:
            if model_dir.exists():
                return model_dir

        raise FileNotFoundError(f"Cannot find smiles model directory for {prop_key} {model_name}: {candidates}")

    def _load_selected_smiles_models(self, property_keys: list[str]) -> None:
        seen: set[str] = set()
        for prop_key in property_keys:
            if prop_key in seen:
                continue
            seen.add(prop_key)
            if prop_key not in self.manifest:
                raise KeyError(f"{prop_key} not found in PeptiVerse manifest")
            row = self.manifest[prop_key]
            model_name = self.module.canon_model(row.best_smiles)
            if model_name is None:
                raise ValueError(f"No smiles-side best model defined for {prop_key}")

            model_dir = self._resolve_dir(prop_key, model_name)
            kind, obj, artifact = self.module.load_artifact(model_dir, self.device)
            if kind in {"xgb", "joblib"}:
                model = obj
            else:
                arch = model_name
                if arch.startswith("transformer"):
                    arch = "transformer"
                elif arch.startswith("mlp"):
                    arch = "mlp"
                elif arch.startswith("cnn"):
                    arch = "cnn"
                model = self.module.build_torch_model_from_ckpt(arch, obj, self.device)

            key = (prop_key, "smiles")
            self.models[key] = model
            self.meta[key] = {
                "task_type": row.task_type,
                "threshold": row.thr_smiles,
                "artifact": str(artifact),
                "model_name": model_name,
                "kind": kind,
            }

    def clear_runtime_cache(self) -> None:
        # The upstream SMILESEmbedder caches GPU tensors per unique SMILES.
        # For offline batch inference over ~200k molecules this can snowball into
        # memory pressure and spurious downstream failures, so we keep cache-free
        # embeddings by default and also provide an explicit cleanup hook.
        self.smiles_embedder._cache_pooled.clear()
        self.smiles_embedder._cache_unpooled.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _get_features_for_model(self, prop_key: str, input_str: str) -> Any:
        key = (prop_key, "smiles")
        meta = self.meta[key]
        kind = meta.get("kind")
        if kind == "torch_ckpt":
            return self.smiles_embedder.unpooled(input_str)
        pooled = self.smiles_embedder.pooled(input_str)
        features = pooled.detach().cpu().numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        return features

    def predict_property(self, prop_key: str, input_str: str) -> dict[str, Any]:
        key = (prop_key, "smiles")
        if key not in self.models:
            raise KeyError(f"No smiles model loaded for {prop_key}")
        model = self.models[key]
        meta = self.meta[key]
        kind = meta.get("kind")
        task_type = str(meta["task_type"]).lower()
        threshold = meta.get("threshold")

        if kind == "torch_ckpt":
            X, M = self._get_features_for_model(prop_key, input_str)
            with torch.no_grad():
                value = model(X, M).squeeze().float().cpu().item()
            if task_type == "classifier":
                score = float(1.0 / (1.0 + np.exp(-value)))
                out = {"property": prop_key, "mode": "smiles", "score": score}
                if threshold is not None:
                    out["label"] = int(score >= float(threshold))
                    out["threshold"] = float(threshold)
                return out
            return {"property": prop_key, "mode": "smiles", "score": float(value)}

        features = self._get_features_for_model(prop_key, input_str)
        if kind == "xgb":
            dmat = self.module.xgb.DMatrix(features)
            pred = float(model.predict(dmat)[0])
            return {"property": prop_key, "mode": "smiles", "score": pred}

        if kind == "joblib":
            if task_type == "classifier":
                if hasattr(model, "predict_proba"):
                    pred = float(model.predict_proba(features)[:, 1][0])
                elif hasattr(model, "decision_function"):
                    logit = float(model.decision_function(features)[0])
                    pred = float(1.0 / (1.0 + np.exp(-logit)))
                else:
                    pred = float(model.predict(features)[0])
                out = {"property": prop_key, "mode": "smiles", "score": pred}
                if threshold is not None:
                    out["label"] = int(pred >= float(threshold))
                    out["threshold"] = float(threshold)
                return out
            pred = float(model.predict(features)[0])
            return {"property": prop_key, "mode": "smiles", "score": pred}

        raise RuntimeError(f"Unsupported PeptiVerse artifact kind: {kind}")


def _extract_label(record: dict[str, Any]) -> Any:
    return record.get("label")


def _to_positive_label(label: Any, mode: str) -> Any:
    if label is None or pd.isna(label):
        return np.nan
    label_int = int(label)
    if mode == "identity":
        return label_int
    if mode == "invert":
        return 1 - label_int
    raise ValueError(f"Unknown positive label conversion mode: {mode}")


def _predict_with_retry(
    predictor: PeptiVerseBatchPredictor,
    predictor_key: str,
    smiles: str,
) -> tuple[dict[str, Any] | None, str | None]:
    last_error: str | None = None
    for attempt in range(2):
        try:
            return predictor.predict_property(predictor_key, smiles), None
        except Exception as exc:
            last_error = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            predictor.clear_runtime_cache()
            if attempt == 0:
                continue
    return None, last_error


def run_peptiverse_predictions(cache: DatasetCache, predictor: PeptiVerseBatchPredictor) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in cache.valid_dataframe.to_dict(orient="records"):
        smiles = str(row["source_smiles"])
        output_row: dict[str, Any] = {
            "dataset_id": cache.spec.dataset_id,
            "dataset_label": cache.spec.dataset_label,
            "SMILES": smiles,
            "canonical_smiles": row.get("canonical_smiles"),
        }

        for task_name, task_info in CLASSIFICATION_TASKS.items():
            predictor_key = str(task_info["predictor_key"])
            result, error_text = _predict_with_retry(predictor, predictor_key, smiles)
            if result is not None:
                output_row[f"{task_name}_score"] = float(result["score"])
                output_row[f"{task_name}_label"] = _extract_label(result)
                output_row[f"{task_name}_positive"] = _to_positive_label(
                    _extract_label(result),
                    str(task_info["positive_from_label"]),
                )
                output_row[f"{task_name}_failed"] = False
                output_row[f"{task_name}_error"] = np.nan
            else:
                output_row[f"{task_name}_score"] = np.nan
                output_row[f"{task_name}_label"] = np.nan
                output_row[f"{task_name}_positive"] = np.nan
                output_row[f"{task_name}_failed"] = True
                output_row[f"{task_name}_error"] = error_text

        for task_name, task_info in REGRESSION_TASKS.items():
            predictor_key = str(task_info["predictor_key"])
            result, error_text = _predict_with_retry(predictor, predictor_key, smiles)
            if result is not None:
                output_row[f"{task_name}_value"] = float(result["score"])
                output_row[f"{task_name}_failed"] = False
                output_row[f"{task_name}_error"] = np.nan
            else:
                output_row[f"{task_name}_value"] = np.nan
                output_row[f"{task_name}_failed"] = True
                output_row[f"{task_name}_error"] = error_text

        rows.append(output_row)

    return pd.DataFrame(rows)


def summarize_peptiverse_predictions(prediction_table: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for (dataset_id, dataset_label), group in prediction_table.groupby(["dataset_id", "dataset_label"]):
        for task_name, task_info in CLASSIFICATION_TASKS.items():
            positive_col = f"{task_name}_positive"
            failed_col = f"{task_name}_failed"
            valid_values = pd.to_numeric(group[positive_col], errors="coerce").dropna().to_numpy(dtype=float)
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": dataset_label,
                    "property": task_info["positive_name"],
                    "property_type": "classification",
                    "count": int(valid_values.size),
                    "failure_rate": float(group[failed_col].mean()),
                    "positive_ratio": float(np.mean(valid_values)) if valid_values.size else np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            )

        for task_name, task_info in REGRESSION_TASKS.items():
            value_col = f"{task_name}_value"
            failed_col = f"{task_name}_failed"
            valid_values = pd.to_numeric(group[value_col], errors="coerce").dropna().to_numpy(dtype=float)
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": dataset_label,
                    "property": task_info["display_name"],
                    "property_type": "regression",
                    "count": int(valid_values.size),
                    "failure_rate": float(group[failed_col].mean()),
                    "positive_ratio": np.nan,
                    "mean": float(np.mean(valid_values)) if valid_values.size else np.nan,
                    "std": float(np.std(valid_values, ddof=0)) if valid_values.size else np.nan,
                    "median": float(np.median(valid_values)) if valid_values.size else np.nan,
                    "min": float(np.min(valid_values)) if valid_values.size else np.nan,
                    "max": float(np.max(valid_values)) if valid_values.size else np.nan,
                }
            )
    return pd.DataFrame(summary_rows)
