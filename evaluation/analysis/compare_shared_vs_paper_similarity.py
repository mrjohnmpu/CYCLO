#!/usr/bin/env python3
"""Compare Shared and Paper-exact generations without mixing topologies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from scipy import sparse
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]


TRUE_VALUES = {"1", "true", "t", "yes", "y", "valid"}
METHOD_ORDER = ("shared", "paper")
DIRECTION_LABELS = {
    "shared_to_paper": "Shared to Paper-exact",
    "paper_to_shared": "Paper-exact to Shared",
}
COLORS = {
    "shared_to_paper": "#0072B2",
    "paper_to_shared": "#D55E00",
}
PARSER: Any | None = None
BUILD_TOPOLOGY: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Shared token+LoRA and Paper-exact token+LoRA within each "
            "cyclization topology."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "compute", "plot", "all"), default="all"
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--parse-chunk-size", type=int, default=32)
    parser.add_argument("--search-chunk-size", type=int, default=None)
    parser.add_argument("--sequence-block-size", type=int, default=None)
    parser.add_argument("--formats", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-topoacc-gate",
        action="store_true",
        help="Diagnostic escape hatch. Never use for the formal manuscript result.",
    )
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    def resolve(value: str) -> str:
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())

    for name in ("output_root", "peptide_parser_root", "chemical_space_module"):
        config[name] = resolve(config[name])
    for method in config["methods"].values():
        method["path_template"] = resolve(method["path_template"])
    return config


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def git_value(repo: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def topology_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    records = config.get("topologies", [])
    if not isinstance(records, list) or not records:
        raise ValueError("Config must contain a non-empty topologies list")
    identifiers = [str(item["id"]) for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Topology identifiers must be unique")
    return records


def topology_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in topology_records(config)}


def source_path(config: dict[str, Any], method: str, topology: str) -> Path:
    return Path(config["methods"][method]["path_template"].format(topology=topology))


def validate_config(config: dict[str, Any]) -> None:
    if set(config.get("methods", {})) != set(METHOD_ORDER):
        raise ValueError(f"Methods must be exactly {METHOD_ORDER}")
    if config["morgan"].get("use_chirality") is not False:
        raise ValueError("The manuscript SNN definition requires use_chirality=false")
    if config["topology_validation"].get("denominator") != "rdkit_valid":
        raise ValueError("TopoAcc denominator must be rdkit_valid")
    if config["topology_validation"].get("classifier") != "peptide_parser_mutually_exclusive":
        raise ValueError("Unsupported topology classifier")
    filter_policy = config.get("analysis_filter", {})
    if filter_policy.get("structure") not in {
        None,
        "rdkit_valid",
        "rdkit_valid_parser_success_topology_correct",
    }:
        raise ValueError("Unsupported structure filter policy")
    if filter_policy.get("residue") not in {
        None,
        "rdkit_valid_parser_success",
        "rdkit_valid_parser_success_topology_correct",
    }:
        raise ValueError("Unsupported residue filter policy")
    ngrams = [int(value) for value in config["sequence"]["ngram_sizes"]]
    if ngrams != [1, 2, 3]:
        raise ValueError("This analysis requires equal-weight 1/2/3-gram features")
    for topology in topology_records(config):
        expected = topology.get("expected_topoacc", {})
        if set(expected) != set(METHOD_ORDER):
            raise ValueError(f"Missing expected TopoAcc values for {topology['id']}")


def resolve_settings(
    config: dict[str, Any], args: argparse.Namespace
) -> tuple[Path, int, int, int, list[str]]:
    output_root = args.output_root or Path(config["output_root"])
    n_jobs = args.n_jobs or int(config["runtime"]["n_jobs"])
    search_chunk_size = args.search_chunk_size or int(
        config["runtime"]["fingerprint_chunk_size"]
    )
    sequence_block_size = args.sequence_block_size or int(
        config["sequence"]["query_block_size"]
    )
    formats = args.formats or list(config["figure"]["formats"])
    if n_jobs <= 0 or search_chunk_size <= 0 or sequence_block_size <= 0:
        raise ValueError("Runtime sizes must be positive")
    if args.max_rows_per_dataset is not None and args.max_rows_per_dataset <= 0:
        raise ValueError("--max-rows-per-dataset must be positive")
    return output_root, n_jobs, search_chunk_size, sequence_block_size, formats


def _init_parser_worker(peptide_parser_root: str) -> None:
    global PARSER, BUILD_TOPOLOGY
    if peptide_parser_root not in sys.path:
        sys.path.insert(0, peptide_parser_root)
    from loguru import logger

    logger.remove()
    from peptide_parser.parser import Parser
    from peptide_parser.topology import build_topology

    PARSER = Parser()
    BUILD_TOPOLOGY = build_topology
    RDLogger.DisableLog("rdApp.*")


def _canonicalize(smiles: str) -> tuple[Chem.Mol | None, str]:
    try:
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception:
        molecule = None
    if molecule is None:
        return None, ""
    return molecule, Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _classify_topology(topology: dict[str, Any]) -> tuple[str, str]:
    body = topology.get("topology", {})
    molecule = topology.get("molecule", {})
    bridges = body.get("bridges", []) if isinstance(body.get("bridges", []), list) else []
    cyclic_backbone = bool(body.get("cyclic_backbone", False))
    has_disulfide = bool(molecule.get("has_disulfide", False)) or any(
        str(bridge.get("type", "")).strip().lower() == "disulfide"
        or bool(bridge.get("has_disulfide", False))
        for bridge in bridges
        if isinstance(bridge, dict)
    )
    has_terminal = any(
        str(bridge.get("role", "")).strip().lower() == "terminal_cyclization"
        for bridge in bridges
        if isinstance(bridge, dict)
    )
    flags = {
        "head_to_tail": cyclic_backbone,
        "sc_to_tail": has_terminal,
        "disulfide_bridge": has_disulfide,
    }
    positive = [name for name, present in flags.items() if present]
    if len(positive) == 1:
        return positive[0], "exclusive_match"
    if len(positive) > 1:
        return "mixed", "+".join(positive)
    return "other", "no_target_topology_feature"


def _parse_worker(task: tuple[int, str, bool]) -> dict[str, Any]:
    source_index, smiles, source_valid = task
    row: dict[str, Any] = {
        "source_index": int(source_index),
        "input_smiles": smiles,
        "source_valid": bool(source_valid),
        "rdkit_valid": False,
        "canonical_smiles": "",
        "parse_success": False,
        "parse_error": "",
        "monomer_id_sequence": "",
        "peptide_length": 0,
        "predicted_topology": "failed",
        "topology_reason": "rdkit_invalid",
        "parser_class": "",
    }
    molecule, canonical = _canonicalize(smiles)
    if molecule is None:
        row["parse_error"] = "rdkit_invalid"
        return row
    row["rdkit_valid"] = True
    row["canonical_smiles"] = canonical
    if PARSER is None or BUILD_TOPOLOGY is None:
        raise RuntimeError("Parser worker was not initialized")
    try:
        molecule_ser, match_df = PARSER.parse_mol(molecule, name=f"row_{source_index:07d}")
        reason = molecule_ser.get("reason_for_exclusion")
        if reason is not None and not (isinstance(reason, float) and math.isnan(reason)):
            row["parse_error"] = f"reason_for_exclusion:{reason}"
            row["topology_reason"] = "parser_excluded"
            return row
        topology = BUILD_TOPOLOGY(molecule_ser, match_df)
        residues = topology.get("topology", {}).get("residues", [])
        monomers = []
        for residue in residues:
            value = residue.get("monomer_id") or residue.get("name") or ""
            value = str(value).strip()
            if value:
                monomers.append(value)
        if not monomers:
            row["parse_error"] = "no_residues_detected"
            row["topology_reason"] = "parser_no_residues"
            return row
        predicted, topology_reason = _classify_topology(topology)
        row.update(
            {
                "parse_success": True,
                "monomer_id_sequence": "|".join(monomers),
                "peptide_length": len(monomers),
                "predicted_topology": predicted,
                "topology_reason": topology_reason,
                "parser_class": str(molecule_ser.get("class") or ""),
            }
        )
    except Exception as exc:  # Per-row failures remain auditable.
        row["parse_error"] = f"{type(exc).__name__}:{exc}"
        row["topology_reason"] = "parser_exception"
    return row


def parse_dataset(
    *,
    config: dict[str, Any],
    method: str,
    topology: str,
    output_root: Path,
    max_rows: int | None,
    n_jobs: int,
    chunksize: int,
    force: bool,
    parser_commit: str,
) -> pd.DataFrame:
    path = source_path(config, method, topology)
    if not path.is_file():
        raise FileNotFoundError(path)
    cache_path = output_root / "cache" / "parsed" / f"{method}_{topology}.parquet"
    metadata_path = cache_path.with_suffix(".metadata.json")
    cache_key = {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "max_rows": max_rows,
        "parser_commit": parser_commit,
        "classifier": config["topology_validation"]["classifier"],
    }
    if not force and cache_path.is_file() and metadata_path.is_file():
        metadata = load_json(metadata_path)
        if metadata.get("cache_key") == cache_key and metadata.get("status") == "complete":
            return pd.read_parquet(cache_path)

    header = pd.read_csv(path, nrows=0)
    required = {"SMILES"}
    missing = required - set(header.columns)
    if missing:
        raise KeyError(f"{path} missing columns {sorted(missing)}")
    usecols = ["SMILES"] + (["Valid"] if "Valid" in header.columns else [])
    frame = pd.read_csv(path, usecols=usecols, nrows=max_rows, low_memory=False)
    smiles = frame["SMILES"].fillna("").astype(str).str.strip()
    source_valid = (
        frame["Valid"].map(parse_bool)
        if "Valid" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    tasks = [
        (int(source_index), text, bool(valid))
        for source_index, text, valid in zip(frame.index, smiles, source_valid)
    ]
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_init_parser_worker,
        initargs=(str(config["peptide_parser_root"]),),
    ) as executor:
        rows = list(executor.map(_parse_worker, tasks, chunksize=max(1, chunksize)))
    result = pd.DataFrame(rows).sort_values("source_index").reset_index(drop=True)
    atomic_write_parquet(result, cache_path)
    atomic_write_json(
        metadata_path,
        {
            "status": "complete",
            "cache_key": cache_key,
            "rows": len(result),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(
        f"Parsed {method}/{topology}: {len(result)} rows in "
        f"{time.monotonic() - started:.1f}s",
        flush=True,
    )
    return result


def _format_rate(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def prepare_inputs(
    config: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    n_jobs: int,
) -> tuple[pd.DataFrame, bool]:
    parser_root = Path(config["peptide_parser_root"])
    parser_commit = git_value(parser_root, "rev-parse", "HEAD")
    summaries: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    digits = int(config["topology_validation"]["round_digits"])
    require_topology_correct = bool(
        config.get("analysis_filter", {}).get("require_topology_correct", True)
    )
    gate_required = require_topology_correct and bool(
        config["topology_validation"].get("require_match_for_full_run", True)
    )
    topology_lookup = topology_map(config)
    for topology in topology_lookup:
        for method in METHOD_ORDER:
            frame = parse_dataset(
                config=config,
                method=method,
                topology=topology,
                output_root=output_root,
                max_rows=args.max_rows_per_dataset,
                n_jobs=n_jobs,
                chunksize=args.parse_chunk_size,
                force=args.force,
                parser_commit=parser_commit,
            )
            rdkit_valid = frame["rdkit_valid"].astype(bool)
            parse_success = frame["parse_success"].astype(bool)
            correct = rdkit_valid & parse_success & frame["predicted_topology"].eq(topology)
            denominator = int(rdkit_valid.sum())
            observed = float(correct.sum() / denominator) if denominator else float("nan")
            expected = float(topology_lookup[topology]["expected_topoacc"][method])
            source_disagreement = int(
                (frame["source_valid"].astype(bool) != frame["rdkit_valid"].astype(bool)).sum()
            )
            summaries.append(
                {
                    "method": method,
                    "method_label": config["methods"][method]["label"],
                    "cyclization": topology,
                    "source_path": str(source_path(config, method, topology)),
                    "total_rows": int(len(frame)),
                    "source_valid": int(frame["source_valid"].sum()),
                    "rdkit_valid": denominator,
                    "source_vs_rdkit_disagreement": source_disagreement,
                    "parse_success": int(parse_success.sum()),
                    "topology_correct": int(correct.sum()),
                    "topoacc_observed": observed,
                    "topoacc_expected": expected,
                    "topoacc_matches_4dp": _format_rate(observed, digits)
                    == _format_rate(expected, digits),
                    "topoacc_gate_evaluated": (
                        require_topology_correct and args.max_rows_per_dataset is None
                    ),
                    "analysis_requires_topology_correct": require_topology_correct,
                    "analysis_structure_rows": int(rdkit_valid.sum())
                    if not require_topology_correct
                    else int(correct.sum()),
                    "analysis_residue_rows": int((rdkit_valid & parse_success).sum())
                    if not require_topology_correct
                    else int(correct.sum()),
                    "excluded_rdkit_invalid": int((~rdkit_valid).sum()),
                    "excluded_parse_failed_after_rdkit": int((rdkit_valid & ~parse_success).sum()),
                    "excluded_topology_mismatch": int(
                        (rdkit_valid & parse_success & ~frame["predicted_topology"].eq(topology)).sum()
                    ),
                    "unique_topology_correct_canonical_smiles": int(
                        frame.loc[correct, "canonical_smiles"].nunique()
                    ),
                    "unique_topology_correct_monomer_sequences": int(
                        frame.loc[correct, "monomer_id_sequence"].nunique()
                    ),
                    "parsed_cache": str(
                        output_root / "cache" / "parsed" / f"{method}_{topology}.parquet"
                    ),
                }
            )
            if gate_required and args.max_rows_per_dataset is None and _format_rate(
                observed, digits
            ) != _format_rate(expected, digits):
                counts = frame.loc[rdkit_valid, "predicted_topology"].value_counts(dropna=False)
                differences.append(
                    {
                        "method": method,
                        "cyclization": topology,
                        "expected_topoacc": expected,
                        "observed_topoacc": observed,
                        "expected_4dp": _format_rate(expected, digits),
                        "observed_4dp": _format_rate(observed, digits),
                        "rdkit_valid_denominator": denominator,
                        "topology_correct": int(correct.sum()),
                        "prediction_counts_json": json.dumps(
                            {str(k): int(v) for k, v in counts.items()}, sort_keys=True
                        ),
                    }
                )

    summary = pd.DataFrame(summaries)
    atomic_write_csv(summary, output_root / "tables" / "input_filter_summary.csv")
    difference_frame = pd.DataFrame(differences)
    if not difference_frame.empty:
        atomic_write_csv(
            difference_frame, output_root / "tables" / "topoacc_differences.csv"
        )
    gate_passed = not differences or args.skip_topoacc_gate or not gate_required
    if args.max_rows_per_dataset is not None:
        gate_passed = True
    if not gate_passed:
        atomic_write_text(
            output_root / "PIPELINE_BLOCKED_TOPOACC.txt",
            "Formal similarity computation was blocked because recomputed TopoAcc "
            "did not match the supplied values at four decimal places. See "
            "tables/topoacc_differences.csv.\n",
        )
    return summary, gate_passed


def require_topoacc_gate(
    output_root: Path, args: argparse.Namespace, config: dict[str, Any]
) -> None:
    if args.max_rows_per_dataset is not None or args.skip_topoacc_gate:
        return
    if not config.get("analysis_filter", {}).get("require_topology_correct", True):
        return
    if not config["topology_validation"].get("require_match_for_full_run", True):
        return
    path = output_root / "tables" / "input_filter_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Formal compute requires a completed topology audit: {path}"
        )
    summary = pd.read_csv(path)
    required = {"topoacc_matches_4dp", "topoacc_gate_evaluated"}
    missing = required - set(summary.columns)
    if missing:
        raise KeyError(f"Topology audit is missing columns: {sorted(missing)}")
    matched = summary["topoacc_matches_4dp"].map(parse_bool)
    evaluated = summary["topoacc_gate_evaluated"].map(parse_bool)
    expected_rows = len(METHOD_ORDER) * len(topology_records(config))
    if len(summary) != expected_rows or not matched.all() or not evaluated.all():
        raise RuntimeError(
            "Formal compute is blocked: all six full-data TopoAcc checks must match "
            "the supplied values at four decimal places"
        )


def canonical_rotation(tokens: Sequence[str]) -> tuple[str, ...]:
    values = tuple(tokens)
    if not values:
        return values
    return min(values[index:] + values[:index] for index in range(len(values)))


def exact_sequence_key(sequence: str, topology: str) -> str:
    tokens = tuple(token for token in str(sequence).split("|") if token)
    if topology == "head_to_tail":
        tokens = canonical_rotation(tokens)
    return "|".join(tokens)


def sequence_ngrams(tokens: Sequence[str], n: int, cyclic: bool) -> list[tuple[str, ...]]:
    values = tuple(tokens)
    length = len(values)
    if not values or n <= 0:
        return []
    if cyclic:
        return [tuple(values[(start + offset) % length] for offset in range(n)) for start in range(length)]
    if length < n:
        return []
    return [values[start : start + n] for start in range(length - n + 1)]


def prepared_set(
    frame: pd.DataFrame,
    topology: str,
    *,
    kind: str,
    require_topology_correct: bool,
) -> pd.DataFrame:
    mask = frame["rdkit_valid"].astype(bool)
    if kind == "sequence" or require_topology_correct:
        mask &= frame["parse_success"].astype(bool)
    if require_topology_correct:
        mask &= frame["predicted_topology"].eq(topology)
    selected = frame.loc[mask].copy()
    if kind == "structure":
        selected = selected.loc[selected["canonical_smiles"].ne("")]
        return selected[["source_index", "canonical_smiles"]].drop_duplicates(
            "canonical_smiles", keep="first"
        ).reset_index(drop=True)
    selected = selected.loc[selected["monomer_id_sequence"].ne("")]
    selected["sequence_key"] = selected["monomer_id_sequence"].map(
        lambda value: exact_sequence_key(value, topology)
    )
    return selected[
        ["source_index", "monomer_id_sequence", "sequence_key"]
    ].drop_duplicates("sequence_key", keep="first").reset_index(drop=True)


def exact_overlap(
    left: set[str], right: set[str], *, topology: str, representation: str
) -> dict[str, Any]:
    intersection = len(left & right)
    union = len(left | right)
    return {
        "cyclization": topology,
        "representation": representation,
        "shared_unique": len(left),
        "paper_unique": len(right),
        "intersection": intersection,
        "union": union,
        "shared_covered_by_paper": intersection / len(left) if left else float("nan"),
        "paper_covered_by_shared": intersection / len(right) if right else float("nan"),
        "jaccard": intersection / union if union else float("nan"),
    }


def dynamic_import(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    # ProcessPoolExecutor pickles workers by module name, so the dynamically
    # loaded project utility must be importable from the parent registry.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sequence_feature_matrices(
    shared: pd.DataFrame,
    paper: pd.DataFrame,
    topology: str,
    ngram_sizes: Sequence[int],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    sequences = [
        tuple(token for token in str(value).split("|") if token)
        for value in pd.concat(
            [shared["monomer_id_sequence"], paper["monomer_id_sequence"]],
            ignore_index=True,
        )
    ]
    split = len(shared)
    blocks: list[sparse.csr_matrix] = []
    cyclic = topology == "head_to_tail"
    for n in ngram_sizes:
        feature_rows = [sequence_ngrams(tokens, int(n), cyclic) for tokens in sequences]
        vocabulary = {
            feature: index
            for index, feature in enumerate(
                sorted({feature for row in feature_rows for feature in row})
            )
        }
        row_indices: list[int] = []
        col_indices: list[int] = []
        values: list[float] = []
        for row_index, features in enumerate(feature_rows):
            counts: dict[int, int] = {}
            for feature in features:
                column = vocabulary[feature]
                counts[column] = counts.get(column, 0) + 1
            for column, count in counts.items():
                row_indices.append(row_index)
                col_indices.append(column)
                values.append(float(count))
        matrix = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(sequences), len(vocabulary)),
            dtype=np.float32,
        )
        blocks.append(normalize(matrix, norm="l2", axis=1, copy=False))
    weight = np.float32(1.0 / math.sqrt(len(blocks)))
    combined = sparse.hstack([block * weight for block in blocks], format="csr")
    return combined[:split].tocsr(), combined[split:].tocsr()


def sparse_top_one(
    queries: sparse.csr_matrix,
    targets: sparse.csr_matrix,
    query_table: pd.DataFrame,
    target_table: pd.DataFrame,
    *,
    block_size: int,
    topology: str,
    direction: str,
) -> pd.DataFrame:
    if queries.shape[0] == 0 or targets.shape[0] == 0:
        raise ValueError("Sequence nearest-neighbor search requires non-empty sets")
    target_transpose = targets.transpose().tocsr()
    pieces: list[pd.DataFrame] = []
    for start in range(0, queries.shape[0], block_size):
        stop = min(start + block_size, queries.shape[0])
        similarities = (queries[start:stop] @ target_transpose).toarray()
        nearest_index = similarities.argmax(axis=1)
        nearest_score = similarities[np.arange(stop - start), nearest_index]
        query_slice = query_table.iloc[start:stop].reset_index(drop=True)
        nearest = target_table.iloc[nearest_index].reset_index(drop=True)
        pieces.append(
            pd.DataFrame(
                {
                    "cyclization": topology,
                    "direction": direction,
                    "query_source_index": query_slice["source_index"].to_numpy(),
                    "query_sequence": query_slice["sequence_key"].to_numpy(),
                    "nearest_target_source_index": nearest["source_index"].to_numpy(),
                    "nearest_target_sequence": nearest["sequence_key"].to_numpy(),
                    "similarity": np.clip(nearest_score.astype(float), 0.0, 1.0),
                }
            )
        )
        print(
            f"Sequence search {topology}/{direction}: {stop}/{queries.shape[0]}",
            flush=True,
        )
    return pd.concat(pieces, ignore_index=True)


def distribution_summary(values: Iterable[float], prefix: str, thresholds: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"Invalid or empty similarity array for {prefix}")
    result: dict[str, Any] = {
        f"{prefix}_n": int(array.size),
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_q05": float(np.quantile(array, 0.05)),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
        f"{prefix}_q95": float(np.quantile(array, 0.95)),
    }
    for threshold in thresholds:
        label = str(threshold).replace(".", "p")
        result[f"{prefix}_ge_{label}"] = float(np.mean(array >= threshold - 1e-12))
    return result


def compute_similarity(
    config: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    n_jobs: int,
    search_chunk_size: int,
    sequence_block_size: int,
) -> None:
    chemical = dynamic_import(Path(config["chemical_space_module"]), "chemical_space_coverage")
    backend = chemical.resolve_search_backend(config["morgan"]["search_backend"])
    thresholds = [float(value) for value in config["thresholds"]]
    exact_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    require_topology_correct = bool(
        config.get("analysis_filter", {}).get("require_topology_correct", True)
    )
    for topology_record in topology_records(config):
        topology = str(topology_record["id"])
        parsed = {
            method: pd.read_parquet(
                output_root / "cache" / "parsed" / f"{method}_{topology}.parquet"
            )
            for method in METHOD_ORDER
        }
        structures = {
            method: prepared_set(
                parsed[method],
                topology,
                kind="structure",
                require_topology_correct=require_topology_correct,
            )
            for method in METHOD_ORDER
        }
        sequences = {
            method: prepared_set(
                parsed[method],
                topology,
                kind="sequence",
                require_topology_correct=require_topology_correct,
            )
            for method in METHOD_ORDER
        }
        exact_rows.append(
            exact_overlap(
                set(structures["shared"]["canonical_smiles"]),
                set(structures["paper"]["canonical_smiles"]),
                topology=topology,
                representation="canonical_smiles",
            )
        )
        exact_rows.append(
            exact_overlap(
                set(sequences["shared"]["sequence_key"]),
                set(sequences["paper"]["sequence_key"]),
                topology=topology,
                representation="monomer_id_sequence",
            )
        )

        structural_results: dict[str, pd.DataFrame] = {}
        for query_method, target_method, direction in (
            ("shared", "paper", "shared_to_paper"),
            ("paper", "shared", "paper_to_shared"),
        ):
            result = chemical.nearest_neighbor_search(
                structures[query_method][["canonical_smiles"]],
                structures[target_method][["canonical_smiles"]],
                cache_dir=output_root
                / "cache"
                / "morgan"
                / topology
                / direction,
                backend=backend,
                n_jobs=n_jobs,
                chunk_size=search_chunk_size,
                radius=int(config["morgan"]["radius"]),
                n_bits=int(config["morgan"]["n_bits"]),
                force=args.force,
            )
            result.insert(0, "direction", direction)
            result.insert(0, "cyclization", topology)
            structural_results[direction] = result
            atomic_write_parquet(
                result,
                output_root / "per_molecule" / f"{topology}_{direction}_structural.parquet",
            )

        shared_matrix, paper_matrix = sequence_feature_matrices(
            sequences["shared"],
            sequences["paper"],
            topology,
            [int(value) for value in config["sequence"]["ngram_sizes"]],
        )
        sequence_results = {
            "shared_to_paper": sparse_top_one(
                shared_matrix,
                paper_matrix,
                sequences["shared"],
                sequences["paper"],
                block_size=sequence_block_size,
                topology=topology,
                direction="shared_to_paper",
            ),
            "paper_to_shared": sparse_top_one(
                paper_matrix,
                shared_matrix,
                sequences["paper"],
                sequences["shared"],
                block_size=sequence_block_size,
                topology=topology,
                direction="paper_to_shared",
            ),
        }
        for direction, result in sequence_results.items():
            atomic_write_parquet(
                result,
                output_root / "per_molecule" / f"{topology}_{direction}_residue.parquet",
            )

        summary: dict[str, Any] = {
            "cyclization": topology,
            "shared_unique_structures": len(structures["shared"]),
            "paper_unique_structures": len(structures["paper"]),
            "shared_unique_sequences": len(sequences["shared"]),
            "paper_unique_sequences": len(sequences["paper"]),
        }
        for direction in DIRECTION_LABELS:
            summary.update(
                distribution_summary(
                    structural_results[direction]["similarity"],
                    f"structure_{direction}",
                    thresholds,
                )
            )
            summary.update(
                distribution_summary(
                    sequence_results[direction]["similarity"],
                    f"residue_{direction}",
                    thresholds,
                )
            )
        summary["structure_symmetric_mean"] = float(
            np.mean(
                [
                    summary["structure_shared_to_paper_mean"],
                    summary["structure_paper_to_shared_mean"],
                ]
            )
        )
        summary["residue_symmetric_mean"] = float(
            np.mean(
                [
                    summary["residue_shared_to_paper_mean"],
                    summary["residue_paper_to_shared_mean"],
                ]
            )
        )
        summary_rows.append(summary)
        print(f"Completed topology: {topology}", flush=True)

    atomic_write_csv(
        pd.DataFrame(exact_rows), output_root / "tables" / "exact_overlap_summary.csv"
    )
    atomic_write_csv(
        pd.DataFrame(summary_rows), output_root / "tables" / "similarity_summary.csv"
    )


def ecdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: Path, formats: Sequence[str], dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    requested = {extension.lower().lstrip(".") for extension in formats}
    if dpi != 600:
        raise ValueError("Nature raster exports are fixed at 600 dpi")
    if "svg" in requested:
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    if "pdf" in requested:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    if "png" in requested:
        fig.savefig(
            stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white"
        )
    if requested.intersection({"tif", "tiff"}):
        fig.savefig(
            stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white"
        )


def plot_ecdf_panels(
    config: dict[str, Any],
    output_root: Path,
    *,
    modality: str,
    formats: Sequence[str],
) -> None:
    configure_matplotlib()
    topology_items = topology_records(config)
    if len(topology_items) == 1:
        fig, axes_grid = plt.subplots(
            1, 1, figsize=(3.5039, 2.3), sharex=True, sharey=True, squeeze=False
        )
    else:
        fig, axes_grid = plt.subplots(
            1,
            len(topology_items),
            figsize=(7.0866, 2.15),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
    axes = axes_grid[0]
    for panel_index, (axis, topology_record) in enumerate(zip(axes, topology_items)):
        topology = str(topology_record["id"])
        suffix = "structural" if modality == "structure" else "residue"
        for direction in DIRECTION_LABELS:
            frame = pd.read_parquet(
                output_root / "per_molecule" / f"{topology}_{direction}_{suffix}.parquet"
            )
            x, y = ecdf(frame["similarity"].to_numpy())
            axis.plot(
                x,
                y,
                color=COLORS[direction],
                linewidth=1.25,
                label=DIRECTION_LABELS[direction],
            )
        axis.set_title(str(topology_record["display"]), pad=4)
        axis.set_xlim(0, 1.0)
        axis.set_ylim(0, 1.0)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(False)
        axis.text(
            0.02,
            0.98,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
        )
    xlabel = (
        "Nearest-neighbour Morgan-Tanimoto similarity"
        if modality == "structure"
        else "Nearest-neighbour residue 1-3-gram cosine similarity"
    )
    fig.supxlabel(xlabel, y=0.015, fontsize=7)
    fig.supylabel("Cumulative fraction", x=0.012, fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    if len(topology_items) == 1:
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.80)
    else:
        fig.subplots_adjust(left=0.09, right=0.995, bottom=0.22, top=0.78, wspace=0.18)
    save_figure(
        fig,
        output_root
        / "figures"
        / ("structural_similarity_ecdf" if modality == "structure" else "residue_sequence_similarity_ecdf"),
        formats,
        int(config["figure"]["dpi"]),
    )
    plt.close(fig)


def plot_results(
    config: dict[str, Any], output_root: Path, formats: Sequence[str]
) -> None:
    plot_ecdf_panels(config, output_root, modality="structure", formats=formats)
    plot_ecdf_panels(config, output_root, modality="residue", formats=formats)


def run_internal_tests() -> None:
    values = ("A", "B", "C", "D")
    rotated = ("C", "D", "A", "B")
    reversed_values = tuple(reversed(values))
    if canonical_rotation(values) != canonical_rotation(rotated):
        raise AssertionError("Cyclic exact keys must be rotation invariant")
    if canonical_rotation(values) == canonical_rotation(reversed_values):
        raise AssertionError("Cyclic exact keys must not be reversal invariant")
    if exact_sequence_key("A|B|C|D", "sc_to_tail") == exact_sequence_key(
        "C|D|A|B", "sc_to_tail"
    ):
        raise AssertionError("Linear topologies must not be rotation invariant")
    left = pd.DataFrame(
        {"source_index": [0, 1], "monomer_id_sequence": ["A|B|C|D", "A|X|C|D"]}
    )
    left["sequence_key"] = left["monomer_id_sequence"].map(
        lambda value: exact_sequence_key(value, "head_to_tail")
    )
    matrix, same = sequence_feature_matrices(left, left.copy(), "head_to_tail", [1, 2, 3])
    result = sparse_top_one(
        matrix,
        same,
        left,
        left,
        block_size=2,
        topology="head_to_tail",
        direction="self_test",
    )
    if not np.allclose(result["similarity"].to_numpy(), 1.0, atol=1e-6):
        raise AssertionError("Sequence self-comparison must return 1.0")
    overlap = exact_overlap(
        set(left["sequence_key"]),
        set(left["sequence_key"]),
        topology="head_to_tail",
        representation="self_test",
    )
    if overlap["jaccard"] != 1.0:
        raise AssertionError("Exact self-overlap must return Jaccard 1.0")


def write_hash_manifest(config: dict[str, Any], output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for topology in topology_map(config):
        for method in METHOD_ORDER:
            path = source_path(config, method, topology)
            rows.append(
                {
                    "role": "input",
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "sha256_manifest.csv":
            continue
        rows.append(
            {
                "role": "output",
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    atomic_write_csv(pd.DataFrame(rows), output_root / "sha256_manifest.csv")


def write_run_record(
    *,
    config_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    started_at: str,
    status: str,
) -> None:
    parser_root = Path(config["peptide_parser_root"])
    require_topology_correct = bool(
        config.get("analysis_filter", {}).get("require_topology_correct", True)
    )
    topology_names = ", ".join(str(item["id"]) for item in topology_records(config))
    filter_text = (
        "RDKit-valid, parser-successful and topology-correct records"
        if require_topology_correct
        else (
            "RDKit-valid records for structure similarity; RDKit-valid and "
            "parser-successful records for residue similarity"
        )
    )
    lines = [
        "# Shared vs Paper-exact Similarity Analysis",
        "",
        "## Runtime",
        f"- Started: {started_at}",
        f"- Updated: {iso_now()}",
        f"- Status: {status}",
        f"- Host: {socket.gethostname()}",
        f"- Python: {sys.executable}",
        f"- Python version: {platform.python_version()}",
        f"- RDKit version: {rdBase.rdkitVersion}",
        f"- peptide_parser commit: {git_value(parser_root, 'rev-parse', 'HEAD')}",
        f"- Config: {config_path.resolve()}",
        f"- Command: `{shlex.join(sys.argv)}`",
        "",
        "## Analysis Contract",
        f"- Configured topology set: {topology_names}.",
        f"- Main-set filter: {filter_text}.",
        "- Morgan fingerprints use radius 2, 1024 bits and no chirality.",
        "- Residue similarity is the equal-weight mean of normalized monomer-ID 1/2/3-gram cosine similarities.",
        "- Head-to-tail n-grams wrap across the closure and are rotation invariant; reverse orientation is not merged.",
        "- Results are descriptive for one training seed; no pseudo-replicate significance test is performed.",
    ]
    atomic_write_text(output_root / "run_record.md", "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config = resolve_config(load_json(args.config))
    validate_config(config)
    output_root, n_jobs, search_chunk_size, sequence_block_size, formats = resolve_settings(
        config, args
    )
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = iso_now()
    resolved = dict(config)
    resolved["resolved_output_root"] = str(output_root)
    resolved["resolved_n_jobs"] = n_jobs
    resolved["resolved_search_chunk_size"] = search_chunk_size
    resolved["resolved_sequence_block_size"] = sequence_block_size
    resolved["resolved_formats"] = formats
    resolved["max_rows_per_dataset"] = args.max_rows_per_dataset
    atomic_write_json(output_root / "resolved_config.json", resolved)
    run_internal_tests()
    gate_passed = True
    try:
        if args.stage in {"prepare", "all"}:
            _, gate_passed = prepare_inputs(config, args, output_root, n_jobs)
            if not gate_passed:
                raise RuntimeError(
                    "TopoAcc validation failed; formal similarity computation was stopped"
                )
        if args.stage in {"compute", "all"}:
            require_topoacc_gate(output_root, args, config)
            compute_similarity(
                config,
                args,
                output_root,
                n_jobs,
                search_chunk_size,
                sequence_block_size,
            )
        if args.stage in {"plot", "all"}:
            plot_results(config, output_root, formats)
        write_hash_manifest(config, output_root)
        write_run_record(
            config_path=args.config,
            config=config,
            args=args,
            output_root=output_root,
            started_at=started_at,
            status="complete",
        )
        atomic_write_text(output_root / "pipeline_complete.txt", f"completed_at={iso_now()}\n")
    except Exception:
        write_run_record(
            config_path=args.config,
            config=config,
            args=args,
            output_root=output_root,
            started_at=started_at,
            status="failed_or_blocked",
        )
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
