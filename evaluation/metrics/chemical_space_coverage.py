#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


TRUE_VALUES = {"1", "true", "t", "yes", "y", "valid"}
TOPOLOGY_DISPLAY = {
    "head_to_tail": "Head-to-tail",
    "sc_to_tail": "Side-chain-to-tail",
    "disulfide_bridge": "Disulfide-bridged",
}
SUMMARY_COLUMNS = [
    "model",
    "model_family",
    "topology",
    "exact_novelty",
    "NNNovelty_0.80",
    "NNNovelty_0.90",
    "NNNovelty_0.95",
    "SNN",
    "Coverage_0.4",
    "Coverage_0.6",
    "Coverage_0.8",
    "unique_Murcko",
    "Murcko_novelty",
    "unique_generic_scaffold",
    "generic_scaffold_novelty",
    "n_total",
    "n_valid",
    "n_unique_valid",
    "duplicate_rate_valid",
    "empty_Murcko",
    "empty_generic_scaffold",
    "source_path",
    "source_sha256",
]

_RDKIT_TARGET_FPS: list[Any] | None = None
_FPSIM_DB_PATH: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute topology-stratified memorization and chemical-space coverage metrics."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "compute", "plot", "all"), default="all")
    parser.add_argument("--search-backend", choices=("auto", "fpsim2", "rdkit"), default="auto")
    parser.add_argument("--n-jobs", type=int, default=max(1, min(16, multiprocessing.cpu_count())))
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-generated-per-dataset", type=int, default=None)
    parser.add_argument("--max-training-per-topology", type=int, default=None)
    parser.add_argument("--max-reference-per-topology", type=int, default=None)
    parser.add_argument("--formats", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def parse_bool_series(series: pd.Series) -> pd.Series:
    return normalize_text(series).str.lower().isin(TRUE_VALUES)


def source_rule_mask(frame: pd.DataFrame, rules: list[dict[str, Any]]) -> pd.Series:
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for rule in rules:
        column = rule["column"]
        if column not in frame.columns:
            raise KeyError(f"Missing validity column {column!r}")
        kind = rule.get("kind", "equals")
        if kind == "bool_true":
            mask &= parse_bool_series(frame[column])
        elif kind == "equals":
            accepted = {str(value).strip().lower() for value in rule.get("values", [])}
            mask &= normalize_text(frame[column]).str.lower().isin(accepted)
        else:
            raise ValueError(f"Unsupported validity rule kind: {kind}")
    return mask


def canonicalize_smiles(smiles: str) -> str | None:
    text = str(smiles).strip()
    if not text:
        return None
    try:
        molecule = Chem.MolFromSmiles(text, sanitize=True)
    except Exception:
        return None
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def morgan_fingerprint(smiles_or_mol: str | Chem.Mol, radius: int = 2, n_bits: int = 1024):
    molecule = (
        smiles_or_mol
        if isinstance(smiles_or_mol, Chem.Mol)
        else Chem.MolFromSmiles(str(smiles_or_mol), sanitize=True)
    )
    if molecule is None:
        raise ValueError(f"Cannot fingerprint invalid SMILES: {smiles_or_mol!r}")
    return AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius,
        nBits=n_bits,
        useChirality=False,
    )


def scaffold_smiles(smiles: str) -> tuple[str | None, str | None]:
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        return None, None
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None, None
    murcko = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
    generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    generic = (
        Chem.MolToSmiles(generic_mol, canonical=True, isomericSmiles=False)
        if generic_mol is not None and generic_mol.GetNumAtoms() > 0
        else None
    )
    return murcko or None, generic or None


def canonical_counts(smiles: Iterable[str]) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    invalid = 0
    for value in smiles:
        canonical = canonicalize_smiles(value)
        if canonical is None:
            invalid += 1
        else:
            counts[canonical] += 1
    return counts, invalid


def prepare_reference_source(
    path: Path,
    topology: str,
    output_path: Path,
    maximum: int | None,
) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=["Cyclization", "SMILES"], low_memory=False)
    subset = frame.loc[frame["Cyclization"].astype(str).eq(topology), "SMILES"]
    raw_rows_for_topology = len(subset)
    if maximum is not None and maximum > 0:
        subset = subset.iloc[:maximum]
    counts, invalid = canonical_counts(subset)
    prepared = pd.DataFrame(
        {
            "row_id": np.arange(1, len(counts) + 1, dtype=np.int64),
            "canonical_smiles": list(counts.keys()),
            "multiplicity": list(counts.values()),
        }
    )
    atomic_write_parquet(prepared, output_path)
    return {
        "topology": topology,
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "raw_rows_for_topology": int(raw_rows_for_topology),
        "sampled_rows": int(len(subset)),
        "invalid_rows": int(invalid),
        "unique_valid_rows": int(len(prepared)),
        "prepared_path": str(output_path),
        "prepared_sha256": sha256_strings(prepared["canonical_smiles"].tolist()),
    }


def prepare_generated_source(
    entry: dict[str, Any],
    output_path: Path,
    maximum: int | None,
) -> dict[str, Any]:
    source_path = Path(entry["path"])
    header = pd.read_csv(source_path, nrows=0)
    smiles_column = entry["smiles_column"]
    rules = entry.get("source_validity", [])
    required = {smiles_column, *(rule["column"] for rule in rules)}
    if "Cyclization" in header.columns:
        required.add("Cyclization")
    missing = required - set(header.columns)
    if missing:
        raise KeyError(f"{entry['dataset_id']} missing columns: {sorted(missing)}")

    frame = pd.read_csv(source_path, usecols=sorted(required), low_memory=False)
    total_rows = len(frame)
    if "Cyclization" in frame.columns:
        normalized_type = normalize_text(frame["Cyclization"])
        topology_mask = normalized_type.eq(entry["cyclization"])
        if not topology_mask.any():
            raise ValueError(
                f"{entry['dataset_id']} has no rows for topology {entry['cyclization']!r}"
            )
        frame = frame.loc[topology_mask].copy()
    source_mask = source_rule_mask(frame, rules)
    candidate_smiles = normalize_text(frame.loc[source_mask, smiles_column])
    if maximum is not None and maximum > 0:
        candidate_smiles = candidate_smiles.iloc[:maximum]
    counts, rdkit_invalid_after_source_filter = canonical_counts(candidate_smiles)
    prepared = pd.DataFrame(
        {
            "canonical_smiles": list(counts.keys()),
            "multiplicity": list(counts.values()),
        }
    )
    prepared.insert(0, "molecule_id", np.arange(1, len(prepared) + 1, dtype=np.int64))
    atomic_write_parquet(prepared, output_path)

    valid_rows = int(prepared["multiplicity"].sum())
    unique_valid = int(len(prepared))
    return {
        "dataset_id": entry["dataset_id"],
        "model": entry["model"],
        "model_family": entry["model_family"],
        "topology": entry["cyclization"],
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "n_total": int(total_rows),
        "n_source_filter_pass": int(source_mask.sum()),
        "n_rdkit_invalid_after_source_filter": int(rdkit_invalid_after_source_filter),
        "n_valid": valid_rows,
        "n_unique_valid": unique_valid,
        "duplicate_rate_valid": 1.0 - (unique_valid / valid_rows if valid_rows else 0.0),
        "prepared_path": str(output_path),
        "prepared_sha256": sha256_strings(prepared["canonical_smiles"].tolist()),
    }


def enrich_manifest_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = load_json(Path(config["source_manifest"]))
    sources = manifest.get("sources", [])
    allowed_ids = set(config["dataset_ids"])
    model_metadata = config["models"]
    entries: list[dict[str, Any]] = []
    for source in sources:
        dataset_id = source.get("dataset_id")
        if dataset_id not in allowed_ids:
            continue
        if source.get("category") != "basic_generation":
            raise ValueError(f"Dataset is not basic_generation: {dataset_id}")
        metadata = model_metadata.get(source["method"])
        if metadata is None:
            raise KeyError(f"Missing model metadata for {source['method']}")
        entries.append({**source, **metadata})

    found_ids = {entry["dataset_id"] for entry in entries}
    if found_ids != allowed_ids:
        raise ValueError(
            f"Dataset selection mismatch; missing={sorted(allowed_ids - found_ids)}, "
            f"unexpected={sorted(found_ids - allowed_ids)}"
        )
    order = {name: index for index, name in enumerate(config["model_order"])}
    topology_order = {name: index for index, name in enumerate(config["topologies"])}
    entries.sort(key=lambda item: (topology_order[item["cyclization"]], order[item["model"]]))
    expected = len(config["topologies"]) * len(config["model_order"])
    if len(entries) != expected:
        raise ValueError(f"Expected {expected} datasets, received {len(entries)}")
    return entries


def prepare_inputs(
    config: dict[str, Any],
    entries: list[dict[str, Any]],
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    prepared_root = output_root / "cache" / "prepared"
    records: list[dict[str, Any]] = []
    for role, source_key, maximum in (
        ("training", "training_csv", args.max_training_per_topology),
        ("reference", "reference_csv", args.max_reference_per_topology),
    ):
        source_path = Path(config[source_key])
        for topology in config["topologies"]:
            path = prepared_root / role / f"{topology}.parquet"
            record = prepare_reference_source(source_path, topology, path, maximum)
            record["role"] = role
            records.append(record)
            print(f"Prepared {role} {topology}: {record['unique_valid_rows']}", flush=True)

    generated_records: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        path = prepared_root / "generated" / f"{entry['dataset_id']}.parquet"
        record = prepare_generated_source(entry, path, args.max_generated_per_dataset)
        generated_records.append(record)
        print(
            f"Prepared generated [{index}/{len(entries)}] {entry['dataset_id']}: "
            f"{record['n_unique_valid']}",
            flush=True,
        )

    atomic_write_csv(pd.DataFrame(records), output_root / "tables" / "reference_source_summary.csv")
    atomic_write_csv(
        pd.DataFrame(generated_records), output_root / "tables" / "generated_source_summary.csv"
    )


def fpsim2_available() -> bool:
    try:
        import FPSim2  # noqa: F401
    except Exception:
        return False
    return True


def resolve_search_backend(requested: str) -> str:
    available = fpsim2_available()
    if requested == "fpsim2" and not available:
        raise RuntimeError("FPSim2 was explicitly requested but is not importable")
    if requested == "auto":
        return "fpsim2" if available else "rdkit"
    return requested


def create_fpsim2_database(
    target: pd.DataFrame,
    database_path: Path,
    radius: int,
    n_bits: int,
    n_jobs: int,
) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = database_path.with_name(
        f"{database_path.stem}.tmp.{os.getpid()}{database_path.suffix}"
    )
    if temporary.exists():
        temporary.unlink()
    executable = shutil.which("fpsim2-create-db")
    if executable:
        smiles_path = database_path.with_name(f"{database_path.stem}_source.smi")
        lines = [
            f"{row.canonical_smiles} {int(row.target_id)}"
            for row in target[["canonical_smiles", "target_id"]].itertuples(index=False)
        ]
        atomic_write_text(smiles_path, "\n".join(lines) + "\n")
        subprocess.run(
            [
                executable,
                str(smiles_path),
                str(temporary),
                "--fp_type",
                "Morgan",
                "--fp_params",
                json.dumps(
                    {"radius": radius, "fpSize": n_bits, "includeChirality": False}
                ),
                "--processes",
                str(n_jobs),
            ],
            check=True,
        )
    else:
        from FPSim2.io import create_db_file

        molecules = [
            [row.canonical_smiles, int(row.target_id)]
            for row in target[["canonical_smiles", "target_id"]].itertuples(index=False)
        ]
        create_db_file(
            mols_source=molecules,
            filename=str(temporary),
            mol_format="smiles",
            fp_type="Morgan",
            fp_params={"radius": radius, "fpSize": n_bits, "includeChirality": False},
        )
    temporary.replace(database_path)


def _init_fpsim_worker(database_path: str) -> None:
    global _FPSIM_DB_PATH
    _FPSIM_DB_PATH = database_path


def _fpsim_top_one(task: tuple[int, str]) -> tuple[int, int, float]:
    from FPSim2 import FPSim2Engine

    query_id, smiles = task
    if not hasattr(_fpsim_top_one, "engine"):
        setattr(_fpsim_top_one, "engine", FPSim2Engine(_FPSIM_DB_PATH))
    engine = getattr(_fpsim_top_one, "engine")
    hits = engine.top_k(smiles, k=1, threshold=0.0, metric="tanimoto", n_workers=1)
    if len(hits) == 0:
        return query_id, -1, 0.0
    return query_id, int(hits[0]["mol_id"]), float(hits[0]["coeff"])


def _rdkit_top_one(task: tuple[int, str]) -> tuple[int, int, float]:
    if _RDKIT_TARGET_FPS is None:
        raise RuntimeError("RDKit search worker was not initialized")
    query_id, smiles = task
    query_fp = morgan_fingerprint(smiles)
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, _RDKIT_TARGET_FPS)
    best_index = int(np.argmax(similarities))
    return query_id, best_index + 1, float(similarities[best_index])


def result_cache_key(
    queries: pd.DataFrame,
    target: pd.DataFrame,
    backend: str,
    radius: int,
    n_bits: int,
) -> str:
    payload = {
        "query": sha256_strings(queries["canonical_smiles"].tolist()),
        "target": sha256_strings(target["canonical_smiles"].tolist()),
        "backend": backend,
        "radius": radius,
        "n_bits": n_bits,
        "use_chirality": False,
        "rdkit": rdBase.rdkitVersion,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def nearest_neighbor_search(
    queries: pd.DataFrame,
    target: pd.DataFrame,
    *,
    cache_dir: Path,
    backend: str,
    n_jobs: int,
    chunk_size: int,
    radius: int,
    n_bits: int,
    force: bool,
) -> pd.DataFrame:
    if queries.empty or target.empty:
        raise ValueError("Nearest-neighbor search requires non-empty query and target sets")
    queries = queries[["canonical_smiles"]].drop_duplicates().reset_index(drop=True).copy()
    queries.insert(0, "query_id", np.arange(1, len(queries) + 1, dtype=np.int64))
    target = target[["canonical_smiles"]].drop_duplicates().reset_index(drop=True).copy()
    target.insert(0, "target_id", np.arange(1, len(target) + 1, dtype=np.int64))
    key = result_cache_key(queries, target, backend, radius, n_bits)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = cache_dir / "nearest_neighbors.parquet"
    metadata_path = cache_dir / "metadata.json"
    cache_valid = False
    if not force and metadata_path.is_file():
        metadata = load_json(metadata_path)
        cache_valid = metadata.get("cache_key") == key
        if cache_valid and metadata.get("status") == "complete" and final_path.is_file():
            return pd.read_parquet(final_path)
    if force or not cache_valid:
        atomic_write_text(
            metadata_path,
            json.dumps(
                {
                    "cache_key": key,
                    "backend": backend,
                    "status": "in_progress",
                    "n_queries": len(queries),
                    "n_targets": len(target),
                    "radius": radius,
                    "n_bits": n_bits,
                    "use_chirality": False,
                    "rdkit_version": rdBase.rdkitVersion,
                },
                indent=2,
            )
            + "\n",
        )

    database_path = cache_dir / "target_morgan_r2_1024.h5"
    if backend == "fpsim2" and (force or not cache_valid or not database_path.is_file()):
        create_fpsim2_database(target, database_path, radius, n_bits, n_jobs)

    chunk_paths: list[Path] = []
    chunks: list[tuple[int, int, Path]] = []
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        chunk_path = cache_dir / "chunks" / f"chunk_{start:09d}_{stop - 1:09d}.parquet"
        chunk_paths.append(chunk_path)
        if force or not cache_valid or not chunk_path.is_file():
            chunks.append((start, stop, chunk_path))

    global _RDKIT_TARGET_FPS
    executor_kwargs: dict[str, Any]
    worker = _fpsim_top_one
    if backend == "fpsim2":
        executor_kwargs = {
            "initializer": _init_fpsim_worker,
            "initargs": (str(database_path),),
        }
    else:
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("The RDKit fallback requires the Linux fork start method")
        _RDKIT_TARGET_FPS = [
            morgan_fingerprint(item, radius, n_bits) for item in target["canonical_smiles"]
        ]
        executor_kwargs = {"mp_context": multiprocessing.get_context("fork")}
        worker = _rdkit_top_one

    if chunks:
        with ProcessPoolExecutor(max_workers=n_jobs, **executor_kwargs) as executor:
            for start, stop, chunk_path in chunks:
                chunk = queries.iloc[start:stop]
                tasks = list(
                    chunk[["query_id", "canonical_smiles"]].itertuples(index=False, name=None)
                )
                rows = list(
                    executor.map(worker, tasks, chunksize=max(1, len(tasks) // n_jobs))
                )
                result = pd.DataFrame(
                    rows, columns=["query_id", "nearest_target_id", "similarity"]
                )
                result = result.merge(queries, on="query_id", how="left", validate="one_to_one")
                result = result.merge(
                    target.rename(
                        columns={
                            "target_id": "nearest_target_id",
                            "canonical_smiles": "nearest_target_smiles",
                        }
                    ),
                    on="nearest_target_id",
                    how="left",
                    validate="many_to_one",
                )
                atomic_write_parquet(result, chunk_path)
                print(f"Search {cache_dir.name}: {stop}/{len(queries)}", flush=True)
    _RDKIT_TARGET_FPS = None

    combined = pd.concat([pd.read_parquet(path) for path in chunk_paths], ignore_index=True)
    combined = combined.sort_values("query_id").reset_index(drop=True)
    atomic_write_parquet(combined, final_path)
    atomic_write_text(
        metadata_path,
        json.dumps(
            {
                "cache_key": key,
                "backend": backend,
                "status": "complete",
                "n_queries": len(queries),
                "n_targets": len(target),
                "radius": radius,
                "n_bits": n_bits,
                "use_chirality": False,
                "rdkit_version": rdBase.rdkitVersion,
            },
            indent=2,
        )
        + "\n",
    )
    return combined


def scaffold_table(smiles: Sequence[str], n_jobs: int) -> pd.DataFrame:
    tasks = list(smiles)
    if n_jobs == 1:
        values = [scaffold_smiles(item) for item in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=n_jobs) as pool:
            values = pool.map(scaffold_smiles, tasks, chunksize=max(1, len(tasks) // (n_jobs * 8)))
    return pd.DataFrame(
        {
            "canonical_smiles": tasks,
            "Murcko_scaffold": [item[0] for item in values],
            "generic_scaffold": [item[1] for item in values],
        }
    )


def ecdf_table(frame: pd.DataFrame, model: str, model_family: str, topology: str) -> pd.DataFrame:
    ordered = frame.sort_values("nearest_train_similarity").reset_index(drop=True)
    result = ordered[["canonical_smiles", "nearest_train_similarity"]].copy()
    result["ecdf"] = np.arange(1, len(result) + 1, dtype=float) / len(result)
    result.insert(0, "topology", topology)
    result.insert(0, "model_family", model_family)
    result.insert(0, "model", model)
    return result


def coverage_grid(similarities: np.ndarray, model: str, model_family: str, topology: str) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, 101)
    coverage = [(similarities >= threshold).mean() for threshold in thresholds]
    return pd.DataFrame(
        {
            "model": model,
            "model_family": model_family,
            "topology": topology,
            "threshold": thresholds,
            "coverage": coverage,
        }
    )


def exact_coverage_curve(
    similarities: np.ndarray, model: str, model_family: str, topology: str
) -> pd.DataFrame:
    thresholds = np.unique(np.concatenate(([0.0], similarities, [1.0])))
    ordered = np.sort(similarities)
    covered = len(ordered) - np.searchsorted(ordered, thresholds, side="left")
    return pd.DataFrame(
        {
            "model": model,
            "model_family": model_family,
            "topology": topology,
            "threshold": thresholds,
            "coverage": covered / len(ordered),
        }
    )


def ratio_novelty(generated: set[str], training: set[str]) -> float:
    return len(generated - training) / len(generated) if generated else float("nan")


def load_expected_snn(config: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for dataset_id, path_text in config.get("existing_snn_metrics", {}).items():
        path = Path(path_text)
        metrics = load_json(path)
        values[dataset_id] = float(metrics["SNN/Test"])
    return values


def compute_metrics(
    config: dict[str, Any],
    entries: list[dict[str, Any]],
    output_root: Path,
    args: argparse.Namespace,
) -> str:
    backend = resolve_search_backend(args.search_backend)
    print(f"Similarity backend: {backend}", flush=True)
    fingerprint = config["fingerprint"]
    radius = int(fingerprint["radius"])
    n_bits = int(fingerprint["n_bits"])
    prepared_root = output_root / "cache" / "prepared"
    input_summary = pd.read_csv(output_root / "tables" / "generated_source_summary.csv")
    expected_snn = load_expected_snn(config)
    is_smoke = any(
        value is not None
        for value in (
            args.max_generated_per_dataset,
            args.max_training_per_topology,
            args.max_reference_per_topology,
        )
    )

    summary_rows: list[dict[str, Any]] = []
    ecdf_frames: list[pd.DataFrame] = []
    coverage_grid_frames: list[pd.DataFrame] = []
    coverage_exact_frames: list[pd.DataFrame] = []
    snn_validation_rows: list[dict[str, Any]] = []

    for topology in config["topologies"]:
        topology_entries = [entry for entry in entries if entry["cyclization"] == topology]
        training = pd.read_parquet(prepared_root / "training" / f"{topology}.parquet")
        reference = pd.read_parquet(prepared_root / "reference" / f"{topology}.parquet")
        generated_frames = {
            entry["dataset_id"]: pd.read_parquet(
                prepared_root / "generated" / f"{entry['dataset_id']}.parquet"
            )
            for entry in topology_entries
        }
        generated_union = pd.concat(
            [frame[["canonical_smiles"]] for frame in generated_frames.values()], ignore_index=True
        ).drop_duplicates()

        union_to_train = nearest_neighbor_search(
            generated_union,
            training,
            cache_dir=output_root / "cache" / "search" / topology / "generated_union_to_training",
            backend=backend,
            n_jobs=args.n_jobs,
            chunk_size=args.chunk_size,
            radius=radius,
            n_bits=n_bits,
            force=args.force,
        ).rename(
            columns={
                "nearest_target_id": "nearest_train_row_id",
                "nearest_target_smiles": "nearest_train_smiles",
                "similarity": "nearest_train_similarity",
            }
        )
        union_to_reference = nearest_neighbor_search(
            generated_union,
            reference,
            cache_dir=output_root / "cache" / "search" / topology / "generated_union_to_reference",
            backend=backend,
            n_jobs=args.n_jobs,
            chunk_size=args.chunk_size,
            radius=radius,
            n_bits=n_bits,
            force=args.force,
        ).rename(
            columns={
                "nearest_target_id": "nearest_reference_row_id",
                "nearest_target_smiles": "nearest_reference_smiles",
                "similarity": "nearest_reference_similarity",
            }
        )

        training_scaffold_path = output_root / "cache" / "scaffolds" / f"training_{topology}.parquet"
        if args.force or not training_scaffold_path.is_file():
            training_scaffolds = scaffold_table(training["canonical_smiles"].tolist(), args.n_jobs)
            atomic_write_parquet(training_scaffolds, training_scaffold_path)
        else:
            training_scaffolds = pd.read_parquet(training_scaffold_path)
        training_murcko = set(training_scaffolds["Murcko_scaffold"].dropna())
        training_generic = set(training_scaffolds["generic_scaffold"].dropna())
        training_smiles = set(training["canonical_smiles"])

        for entry in topology_entries:
            dataset_id = entry["dataset_id"]
            generated = generated_frames[dataset_id]
            per_molecule = generated.merge(
                union_to_train[
                    [
                        "canonical_smiles",
                        "nearest_train_row_id",
                        "nearest_train_smiles",
                        "nearest_train_similarity",
                    ]
                ],
                on="canonical_smiles",
                how="left",
                validate="one_to_one",
            ).merge(
                union_to_reference[
                    [
                        "canonical_smiles",
                        "nearest_reference_row_id",
                        "nearest_reference_smiles",
                        "nearest_reference_similarity",
                    ]
                ],
                on="canonical_smiles",
                how="left",
                validate="one_to_one",
            )
            per_molecule["exact_training_match"] = per_molecule["canonical_smiles"].isin(
                training_smiles
            )
            per_molecule.insert(0, "topology", topology)
            per_molecule.insert(0, "model_family", entry["model_family"])
            per_molecule.insert(0, "model", entry["model"])

            generated_scaffold_path = (
                output_root / "scaffolds" / f"{dataset_id}_scaffolds.parquet"
            )
            if args.force or not generated_scaffold_path.is_file():
                generated_scaffolds = scaffold_table(
                    generated["canonical_smiles"].tolist(), args.n_jobs
                )
                generated_scaffolds.insert(0, "topology", topology)
                generated_scaffolds.insert(0, "model_family", entry["model_family"])
                generated_scaffolds.insert(0, "model", entry["model"])
                atomic_write_parquet(generated_scaffolds, generated_scaffold_path)
            else:
                generated_scaffolds = pd.read_parquet(generated_scaffold_path)
            per_molecule = per_molecule.merge(
                generated_scaffolds[
                    ["canonical_smiles", "Murcko_scaffold", "generic_scaffold"]
                ],
                on="canonical_smiles",
                how="left",
                validate="one_to_one",
            )
            atomic_write_parquet(
                per_molecule, output_root / "per_molecule" / f"{dataset_id}.parquet"
            )

            reference_to_generated = nearest_neighbor_search(
                reference,
                generated,
                cache_dir=output_root
                / "cache"
                / "search"
                / topology
                / f"reference_to_{dataset_id}",
                backend=backend,
                n_jobs=args.n_jobs,
                chunk_size=args.chunk_size,
                radius=radius,
                n_bits=n_bits,
                force=args.force,
            ).rename(
                columns={
                    "canonical_smiles": "reference_smiles",
                    "nearest_target_id": "nearest_generated_id",
                    "nearest_target_smiles": "nearest_generated_smiles",
                    "similarity": "nearest_generated_similarity",
                }
            )
            reference_to_generated.insert(0, "topology", topology)
            reference_to_generated.insert(0, "model_family", entry["model_family"])
            reference_to_generated.insert(0, "model", entry["model"])
            atomic_write_parquet(
                reference_to_generated,
                output_root / "per_molecule" / f"{dataset_id}_reference_coverage.parquet",
            )

            train_sim = per_molecule["nearest_train_similarity"].to_numpy(dtype=float)
            reference_sim = reference_to_generated["nearest_generated_similarity"].to_numpy(
                dtype=float
            )
            weights = per_molecule["multiplicity"].to_numpy(dtype=float)
            snn = float(np.average(per_molecule["nearest_reference_similarity"], weights=weights))
            generated_murcko = set(generated_scaffolds["Murcko_scaffold"].dropna())
            generated_generic = set(generated_scaffolds["generic_scaffold"].dropna())
            empty_murcko = int(generated_scaffolds["Murcko_scaffold"].isna().sum())
            empty_generic = int(generated_scaffolds["generic_scaffold"].isna().sum())
            empty_fraction = max(empty_murcko, empty_generic) / len(generated_scaffolds)
            if empty_fraction > float(config["max_empty_scaffold_fraction"]):
                raise RuntimeError(
                    f"Unexpected empty scaffold fraction for {dataset_id}: {empty_fraction:.6f}"
                )

            source_row = input_summary.loc[input_summary["dataset_id"].eq(dataset_id)].iloc[0]
            summary_rows.append(
                {
                    "model": entry["model"],
                    "model_family": entry["model_family"],
                    "topology": topology,
                    "exact_novelty": float((~per_molecule["exact_training_match"]).mean()),
                    "NNNovelty_0.80": float((train_sim < 0.80).mean()),
                    "NNNovelty_0.90": float((train_sim < 0.90).mean()),
                    "NNNovelty_0.95": float((train_sim < 0.95).mean()),
                    "SNN": snn,
                    "Coverage_0.4": float((reference_sim >= 0.4).mean()),
                    "Coverage_0.6": float((reference_sim >= 0.6).mean()),
                    "Coverage_0.8": float((reference_sim >= 0.8).mean()),
                    "unique_Murcko": len(generated_murcko),
                    "Murcko_novelty": ratio_novelty(generated_murcko, training_murcko),
                    "unique_generic_scaffold": len(generated_generic),
                    "generic_scaffold_novelty": ratio_novelty(
                        generated_generic, training_generic
                    ),
                    "n_total": int(source_row["n_total"]),
                    "n_valid": int(source_row["n_valid"]),
                    "n_unique_valid": int(source_row["n_unique_valid"]),
                    "duplicate_rate_valid": float(source_row["duplicate_rate_valid"]),
                    "empty_Murcko": empty_murcko,
                    "empty_generic_scaffold": empty_generic,
                    "source_path": source_row["source_path"],
                    "source_sha256": source_row["source_sha256"],
                }
            )
            ecdf_frames.append(
                ecdf_table(per_molecule, entry["model"], entry["model_family"], topology)
            )
            coverage_grid_frames.append(
                coverage_grid(reference_sim, entry["model"], entry["model_family"], topology)
            )
            coverage_exact_frames.append(
                exact_coverage_curve(
                    reference_sim, entry["model"], entry["model_family"], topology
                )
            )

            expected = expected_snn.get(dataset_id)
            difference = abs(snn - expected) if expected is not None else float("nan")
            snn_validation_rows.append(
                {
                    "dataset_id": dataset_id,
                    "model": entry["model"],
                    "topology": topology,
                    "computed_snn": snn,
                    "existing_snn": expected,
                    "absolute_difference": difference,
                    "within_tolerance": (
                        True
                        if is_smoke or expected is None
                        else difference <= float(config["snn_validation_tolerance"])
                    ),
                }
            )
            print(f"Computed {dataset_id}", flush=True)

    summary = pd.DataFrame(summary_rows)[SUMMARY_COLUMNS]
    atomic_write_csv(summary, output_root / "tables" / "chemical_space_coverage_summary.csv")
    atomic_write_csv(
        pd.concat(ecdf_frames, ignore_index=True),
        output_root / "curves" / "nearest_training_similarity_ecdf.csv",
    )
    atomic_write_csv(
        pd.concat(coverage_grid_frames, ignore_index=True),
        output_root / "curves" / "reference_coverage_curve_grid.csv",
    )
    atomic_write_parquet(
        pd.concat(coverage_exact_frames, ignore_index=True),
        output_root / "curves" / "reference_coverage_curve_exact.parquet",
    )
    validation = pd.DataFrame(snn_validation_rows)
    atomic_write_csv(validation, output_root / "tables" / "snn_regression_validation.csv")
    if not is_smoke and not validation["within_tolerance"].all():
        failures = validation.loc[~validation["within_tolerance"]]
        raise RuntimeError(
            "SNN regression validation failed:\n" + failures.to_string(index=False)
        )
    return backend


def run_plots(config: dict[str, Any], output_root: Path, formats: list[str] | None) -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from evaluation.plots.chemical_space_coverage_plots import generate_all_figures

    generate_all_figures(
        output_root=output_root,
        model_order=config["model_order"],
        topology_order=config["topologies"],
        formats=formats or config["figure_formats"],
    )


def write_run_record(
    config_path: Path,
    config: dict[str, Any],
    output_root: Path,
    command: str,
    backend: str | None,
    elapsed: float,
) -> None:
    try:
        import FPSim2

        fpsim2_version = getattr(FPSim2, "__version__", "installed")
    except Exception:
        fpsim2_version = "not installed"
    lines = [
        "# Memorization and chemical-space coverage run record",
        "",
        f"- Completed: {datetime.now().astimezone().isoformat()}",
        f"- Host: {socket.gethostname()}",
        f"- Python: {sys.executable}",
        f"- Python version: {platform.python_version()}",
        f"- RDKit: {rdBase.rdkitVersion}",
        f"- FPSim2: {fpsim2_version}",
        f"- Search backend: {backend or 'not used'}",
        f"- Config: {config_path}",
        f"- Output root: {output_root}",
        f"- Command: `{command}`",
        f"- Elapsed seconds: {elapsed:.3f}",
        "",
        "## Metric contract",
        "",
        "- Morgan fingerprint: radius=2, nBits=1024, useChirality=False.",
        "- Memorization, NN novelty, and scaffolds use canonical valid unique generated molecules.",
        "- SNN is generated-to-reference and is weighted by original valid-sample multiplicity.",
        "- Coverage is reference-to-generated and uses unique generated molecules.",
        "- NNNovelty uses strict similarity < threshold; Coverage uses similarity >= threshold.",
        "- Topologies are processed independently and are never pooled.",
        "",
        "## Source files",
        "",
        f"- Training: {config['training_csv']}",
        f"- Reference: {config['reference_csv']}",
        f"- Generated source manifest: {config['source_manifest']}",
        "",
        "## Machine-readable outputs",
        "",
        f"- {output_root / 'tables' / 'chemical_space_coverage_summary.csv'}",
        f"- {output_root / 'tables' / 'snn_regression_validation.csv'}",
        f"- {output_root / 'curves' / 'nearest_training_similarity_ecdf.csv'}",
        f"- {output_root / 'curves' / 'reference_coverage_curve_grid.csv'}",
        f"- {output_root / 'curves' / 'reference_coverage_curve_exact.parquet'}",
    ]
    atomic_write_text(output_root / "run_record.md", "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    if args.n_jobs < 1 or args.chunk_size < 1:
        raise ValueError("--n-jobs and --chunk-size must be positive")
    config_path = args.config.resolve()
    config = load_json(config_path)
    fingerprint = config.get("fingerprint", {})
    if (
        fingerprint.get("type") != "Morgan"
        or int(fingerprint.get("radius", -1)) != 2
        or int(fingerprint.get("n_bits", -1)) != 1024
        or bool(fingerprint.get("use_chirality", True))
    ):
        raise ValueError(
            "The analysis requires the paper SNN fingerprint: Morgan radius=2, "
            "nBits=1024, useChirality=False"
        )
    output_root = (args.output_root or Path(config["output_root"])).resolve()
    for directory in (
        "cache",
        "per_molecule",
        "curves",
        "scaffolds",
        "tables",
        "figures",
        "logs",
    ):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    entries = enrich_manifest_entries(config)
    atomic_write_text(
        output_root / "resolved_config.json",
        json.dumps({**config, "resolved_sources": entries}, indent=2) + "\n",
    )
    RDLogger.DisableLog("rdApp.*")

    started = time.time()
    backend: str | None = None
    if args.stage in {"prepare", "all"}:
        prepare_inputs(config, entries, output_root, args)
    if args.stage in {"compute", "all"}:
        backend = compute_metrics(config, entries, output_root, args)
    if args.stage in {"plot", "all"}:
        run_plots(config, output_root, args.formats)
    elapsed = time.time() - started
    write_run_record(
        config_path,
        config,
        output_root,
        " ".join(shlex.quote(part) for part in sys.argv),
        backend,
        elapsed,
    )
    print(f"Completed: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
