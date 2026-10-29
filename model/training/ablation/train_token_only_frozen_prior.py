from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shlex
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from peft import TrainableTokensConfig, get_peft_model
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    EarlyStoppingCallback,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.shared.pipeline_utils import load_tokenizer, parse_args_with_config, seed_everything
from model.training.lora.dataset import (
    CYCLIZATION_COL_CANDIDATES,
    SMILES_COL_CANDIDATES,
    TYPE_PREFIX_TOKENS,
    PeptideDataProcessor,
    canonicalize_peptide_types,
    find_column,
    normalize_type,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

EXPECTED_FULL_SPLITS = {
    "head_to_tail": {"total": 9999, "train": 7999, "val": 1000, "test": 1000},
    "sc_to_tail": {"total": 10001, "train": 8001, "val": 1000, "test": 1000},
    "disulfide_bridge": {"total": 10000, "train": 8000, "val": 1000, "test": 1000},
}


class TokenConditionDataset(Dataset):
    def __init__(self, smiles_values, peptide_type, tokenizer, max_length):
        if max_length < 3:
            raise ValueError("max_length must be at least 3")
        condition_token = TYPE_PREFIX_TOKENS[peptide_type]
        condition_token_id = tokenizer.convert_tokens_to_ids(condition_token)
        if condition_token_id == tokenizer.unk_token_id:
            raise ValueError(f"Condition token is not registered: {condition_token}")
        if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define EOS and PAD token IDs.")

        self.examples = []
        for smiles in smiles_values:
            smiles_ids = tokenizer.encode(
                smiles,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length - 2,
            )
            input_ids = [condition_token_id, *smiles_ids, tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            padding = max_length - len(input_ids)
            input_ids.extend([tokenizer.pad_token_id] * padding)
            attention_mask.extend([0] * padding)
            labels = [
                token_id if mask else -100
                for token_id, mask in zip(input_ids, attention_mask)
            ]
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def resolve_repo_path(path_str: str, *, allow_model_hub: bool = False) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    candidate = PROJECT_ROOT / path
    if candidate.exists() or not allow_model_hub:
        return str(candidate.resolve())
    return path_str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train topology condition tokens while keeping the pretrained prior frozen."
    )
    parser.add_argument("--base_model_path", type=str, required=False)
    parser.add_argument("--tokenizer_path", type=str, required=False)
    parser.add_argument("--data_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--peptide_types", nargs="*", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--early_stop_patience", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_samples_per_type", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--audit_frozen_parameters", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    args = parse_args_with_config(build_parser())
    missing = [
        key
        for key in ("base_model_path", "tokenizer_path", "data_path", "output_dir")
        if not getattr(args, key, None)
    ]
    if missing:
        raise ValueError("Missing required settings: " + ", ".join(missing))
    args.peptide_types = canonicalize_peptide_types(args.peptide_types)
    if set(args.peptide_types) != set(EXPECTED_FULL_SPLITS):
        raise ValueError(
            "Token-only ablation requires exactly head_to_tail, sc_to_tail, and disulfide_bridge."
        )
    args.base_model_path = resolve_repo_path(args.base_model_path)
    args.tokenizer_path = resolve_repo_path(args.tokenizer_path, allow_model_hub=True)
    args.data_path = resolve_repo_path(args.data_path)
    args.output_dir = resolve_repo_path(args.output_dir)
    return args


def prepare_splits(args, tokenizer):
    dataframe = pd.read_csv(args.data_path)
    cyclization_col = find_column(dataframe, CYCLIZATION_COL_CANDIDATES)
    smiles_col = find_column(dataframe, SMILES_COL_CANDIDATES)
    dataframe = dataframe[[cyclization_col, smiles_col]].dropna().copy()
    dataframe[smiles_col] = dataframe[smiles_col].astype(str).str.strip()
    dataframe["_type"] = dataframe[cyclization_col].apply(normalize_type)
    dataframe = dataframe.dropna(subset=["_type"])
    dataframe = dataframe[dataframe[smiles_col].str.len() > 0]

    datasets = {}
    summaries = {}
    manifest_rows = []
    for peptide_type in args.peptide_types:
        smiles_values = dataframe.loc[
            dataframe["_type"] == peptide_type, smiles_col
        ].tolist()
        random.Random(f"{args.split_seed}:{peptide_type}").shuffle(smiles_values)
        if args.max_samples_per_type is not None:
            smiles_values = smiles_values[: args.max_samples_per_type]

        train_count, val_count, test_count = PeptideDataProcessor._resolve_split_counts(
            len(smiles_values),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        split_values = {
            "train": smiles_values[:train_count],
            "val": smiles_values[train_count : train_count + val_count],
            "test": smiles_values[train_count + val_count :],
        }
        datasets[peptide_type] = {
            split: TokenConditionDataset(values, peptide_type, tokenizer, args.max_length)
            for split, values in split_values.items()
        }
        summaries[peptide_type] = {
            "total": len(smiles_values),
            "train": train_count,
            "val": val_count,
            "test": test_count,
        }
        for split, values in split_values.items():
            manifest_rows.extend(
                {
                    "peptide_type": peptide_type,
                    "split": split,
                    "split_position": index,
                    "SMILES": smiles,
                }
                for index, smiles in enumerate(values)
            )

    if args.max_samples_per_type is None and summaries != EXPECTED_FULL_SPLITS:
        raise RuntimeError(
            f"Full-data split mismatch. Expected {EXPECTED_FULL_SPLITS}, received {summaries}."
        )
    return datasets, summaries, pd.DataFrame(manifest_rows)


def frozen_parameter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit_dataset_prefixes(datasets, peptide_types, condition_token_ids) -> None:
    expected_ids = dict(zip(peptide_types, condition_token_ids))
    for peptide_type in peptide_types:
        expected_id = expected_ids[peptide_type]
        for split, dataset in datasets[peptide_type].items():
            mismatches = sum(
                int(example["input_ids"][0].item() != expected_id)
                for example in dataset
            )
            if mismatches:
                raise RuntimeError(
                    f"Condition-token audit failed for {peptide_type}/{split}: "
                    f"{mismatches} mismatches."
                )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite only for an intentional rerun."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_dir / "tokenizer"
    adapter_dir = output_dir / "adapter"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_path, model_max_length=args.max_length)
    condition_tokens = [TYPE_PREFIX_TOKENS[value] for value in args.peptide_types]
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": condition_tokens}
    )
    if added != len(condition_tokens):
        raise RuntimeError(
            f"Expected to add {len(condition_tokens)} condition tokens, added {added}."
        )
    condition_token_ids = tokenizer.convert_tokens_to_ids(condition_tokens)
    if len(set(condition_token_ids)) != len(condition_tokens):
        raise RuntimeError(f"Condition token IDs are not unique: {condition_token_ids}")
    tokenizer.save_pretrained(tokenizer_dir)

    datasets, split_summaries, split_manifest = prepare_splits(args, tokenizer)
    audit_dataset_prefixes(datasets, args.peptide_types, condition_token_ids)
    split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    write_json(output_dir / "split_summary.json", split_summaries)
    train_dataset = ConcatDataset(
        [datasets[value]["train"] for value in args.peptide_types]
    )
    val_dataset = ConcatDataset(
        [datasets[value]["val"] for value in args.peptide_types]
    )

    base_model = GPT2LMHeadModel.from_pretrained(args.base_model_path)
    base_model.resize_token_embeddings(len(tokenizer))
    token_config = TrainableTokensConfig(token_indices=condition_token_ids)
    model = get_peft_model(base_model, token_config)

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    trainable_count = sum(param.numel() for _, param in trainable)
    expected_count = len(condition_token_ids) * int(base_model.config.n_embd)
    unexpected_names = [name for name, _ in trainable if "trainable_tokens" not in name]
    if trainable_count != expected_count or unexpected_names:
        raise RuntimeError(
            "Frozen-prior audit failed before training: "
            f"expected={expected_count}, actual={trainable_count}, unexpected={unexpected_names}."
        )
    logger.info("Trainable token IDs: %s", condition_token_ids)
    logger.info("Trainable parameters: %s", [name for name, _ in trainable])
    logger.info("Trainable parameter count: %s", trainable_count)

    use_step_schedule = args.max_steps > 0
    schedule_kwargs = {
        "eval_strategy": "steps" if use_step_schedule else "epoch",
        "save_strategy": "steps" if use_step_schedule else "epoch",
    }
    if use_step_schedule:
        schedule_kwargs.update(
            {
                "max_steps": args.max_steps,
                "eval_steps": args.max_steps,
                "save_steps": args.max_steps,
            }
        )

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum_steps,
        logging_strategy="steps",
        logging_steps=max(1, args.log_steps),
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        label_names=["labels"],
        run_name="token_only_frozen_prior_seed42",
        **schedule_kwargs,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
    )

    frozen_hash_before = frozen_parameter_hash(model) if args.audit_frozen_parameters else None
    train_output = trainer.train()
    frozen_hash_after = frozen_parameter_hash(model) if args.audit_frozen_parameters else None
    if frozen_hash_before is not None and frozen_hash_before != frozen_hash_after:
        raise RuntimeError("Frozen prior parameters changed during training.")

    trainer.model.save_pretrained(adapter_dir)
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_model_path": args.base_model_path,
        "base_tokenizer_path": args.tokenizer_path,
        "tokenizer_path": str(tokenizer_dir),
        "adapter_path": str(adapter_dir),
        "data_path": args.data_path,
        "peptide_types": args.peptide_types,
        "condition_tokens": condition_tokens,
        "condition_token_ids": condition_token_ids,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "split_summary": split_summaries,
        "trainable_parameters": trainable_count,
        "expected_trainable_parameters": expected_count,
        "trainable_parameter_names": [name for name, _ in trainable],
        "train_loss": train_output.training_loss,
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "frozen_parameter_hash_before": frozen_hash_before,
        "frozen_parameter_hash_after": frozen_hash_after,
        "command": " ".join(shlex.quote(value) for value in sys.argv),
    }
    write_json(output_dir / "training_summary.json", summary)
    logger.info("Saved token-only adapter to %s", adapter_dir)


if __name__ == "__main__":
    main()
