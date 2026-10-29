from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from model.shared.pipeline_utils import build_condition_token

logger = logging.getLogger(__name__)

PEPTIDE_TYPES = [
    "linear",
    "head_to_tail",
    "sc_to_tail",
    "disulfide_bridge",
]
TYPE_PREFIX_TOKENS = {
    peptide_type: build_condition_token(peptide_type)
    for peptide_type in PEPTIDE_TYPES
}
CYCLIZATION_COL_CANDIDATES = ["Cyclization", "cyclization"]
SMILES_COL_CANDIDATES = ["SMILES", "smiles"]


def canonicalize_peptide_types(peptide_types: Sequence[str] | None) -> list[str]:
    if peptide_types is None:
        return list(PEPTIDE_TYPES)

    normalized_types: list[str] = []
    seen: set[str] = set()
    for raw_type in peptide_types:
        peptide_type = normalize_type(raw_type)
        if peptide_type is None:
            raise ValueError(f"Unsupported peptide type: {raw_type}")
        if peptide_type not in seen:
            normalized_types.append(peptide_type)
            seen.add(peptide_type)

    if not normalized_types:
        raise ValueError("At least one peptide type must be provided.")
    return normalized_types


def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    lower_to_actual = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        matched = lower_to_actual.get(candidate.lower())
        if matched is not None:
            return matched
    raise ValueError(
        f"Unable to find any of {list(candidates)} in CSV columns {list(df.columns)}"
    )


def normalize_type(raw_type: str | None) -> Optional[str]:
    if raw_type is None:
        return None
    normalized = (
        str(raw_type)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if normalized in TYPE_PREFIX_TOKENS:
        return normalized
    return None


def add_condition_tokens_to_tokenizer(tokenizer, peptide_types: Sequence[str] | None = None) -> int:
    target_types = canonicalize_peptide_types(peptide_types)
    tokens = [TYPE_PREFIX_TOKENS[peptide_type] for peptide_type in target_types]
    existing_vocab = set(tokenizer.get_vocab().keys())
    tokens_to_add = [token for token in tokens if token not in existing_vocab]

    if not tokens_to_add:
        logger.info("Condition tokens already exist in tokenizer vocabulary.")
        return 0

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens_to_add}
    )
    logger.info("Added %s condition tokens: %s", num_added, tokens_to_add)
    return num_added


class TypeSpecificDataset(Dataset):
    def __init__(
        self,
        smiles_list: Sequence[str],
        peptide_type: str,
        tokenizer,
        max_length: int = 576,
    ):
        if peptide_type not in PEPTIDE_TYPES:
            raise ValueError(
                f"Unsupported peptide type '{peptide_type}'. Expected one of {PEPTIDE_TYPES}."
            )

        self.peptide_type = peptide_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix_token = TYPE_PREFIX_TOKENS[peptide_type]
        self.examples: list[dict[str, torch.Tensor]] = []

        self._build(smiles_list)

    def _build(self, smiles_list: Sequence[str]) -> None:
        skipped = 0
        eos_token = self.tokenizer.eos_token or ""
        for smiles in smiles_list:
            if not isinstance(smiles, str) or not smiles.strip():
                skipped += 1
                continue

            text = f"{self.prefix_token}{smiles.strip()}{eos_token}"
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        logger.info(
            "Built dataset for %s: %s samples (%s skipped)",
            self.peptide_type,
            len(self.examples),
            skipped,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class PeptideDataProcessor:
    def __init__(self, tokenizer, max_length: int = 576):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_and_split(
        self,
        csv_path: str,
        peptide_types: Sequence[str] | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_samples_per_type: int | None = None,
    ) -> tuple[dict[str, dict[str, TypeSpecificDataset]], dict[str, dict[str, int]]]:
        target_types = canonicalize_peptide_types(peptide_types)
        dataframe = pd.read_csv(csv_path)
        cyclization_col = find_column(dataframe, CYCLIZATION_COL_CANDIDATES)
        smiles_col = find_column(dataframe, SMILES_COL_CANDIDATES)

        dataframe = dataframe[[cyclization_col, smiles_col]].dropna(subset=[cyclization_col, smiles_col]).copy()
        dataframe[smiles_col] = dataframe[smiles_col].astype(str).str.strip()
        dataframe["_type"] = dataframe[cyclization_col].apply(normalize_type)
        dataframe = dataframe.dropna(subset=["_type"])
        dataframe = dataframe[dataframe[smiles_col].str.len() > 0]

        splits: dict[str, dict[str, TypeSpecificDataset]] = {}
        split_summaries: dict[str, dict[str, int]] = {}

        for peptide_type in target_types:
            smiles_values = dataframe.loc[
                dataframe["_type"] == peptide_type, smiles_col
            ].tolist()
            rng = random.Random(f"{seed}:{peptide_type}")
            rng.shuffle(smiles_values)

            if max_samples_per_type is not None:
                smiles_values = smiles_values[:max_samples_per_type]

            train_count, val_count, test_count = self._resolve_split_counts(
                len(smiles_values),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
            train_smiles = smiles_values[:train_count]
            val_smiles = smiles_values[train_count : train_count + val_count]
            test_smiles = smiles_values[train_count + val_count :]

            splits[peptide_type] = {
                "train": TypeSpecificDataset(
                    train_smiles,
                    peptide_type,
                    self.tokenizer,
                    self.max_length,
                ),
                "val": TypeSpecificDataset(
                    val_smiles,
                    peptide_type,
                    self.tokenizer,
                    self.max_length,
                ),
                "test": TypeSpecificDataset(
                    test_smiles,
                    peptide_type,
                    self.tokenizer,
                    self.max_length,
                ),
            }
            split_summaries[peptide_type] = {
                "total": len(smiles_values),
                "train": train_count,
                "val": val_count,
                "test": test_count,
            }
            logger.info(
                "Prepared splits for %s: total=%s train=%s val=%s test=%s",
                peptide_type,
                len(smiles_values),
                train_count,
                val_count,
                test_count,
            )

        return splits, split_summaries

    @staticmethod
    def _resolve_split_counts(
        total_count: int,
        *,
        train_ratio: float,
        val_ratio: float,
    ) -> tuple[int, int, int]:
        if total_count < 3:
            raise ValueError(
                "Each peptide type needs at least 3 samples to create non-empty train/val/test splits. "
                f"Received {total_count}."
            )
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        if not 0 < val_ratio < 1:
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
        if train_ratio + val_ratio >= 1:
            raise ValueError(
                f"train_ratio + val_ratio must be < 1, got {train_ratio + val_ratio}"
            )

        train_count = max(1, int(round(total_count * train_ratio)))
        val_count = max(1, int(round(total_count * val_ratio)))
        test_count = total_count - train_count - val_count

        while test_count < 1:
            if train_count >= val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                raise ValueError(
                    f"Unable to create non-empty train/val/test splits from {total_count} samples."
                )
            test_count = total_count - train_count - val_count

        return train_count, val_count, test_count
