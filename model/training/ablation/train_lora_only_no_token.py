from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shlex
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
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
from model.training.lora.dataset import TYPE_PREFIX_TOKENS, canonicalize_peptide_types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

EXPECTED_SPLITS = {
    "head_to_tail": {"train": 7999, "val": 1000, "test": 1000},
    "sc_to_tail": {"train": 8001, "val": 1000, "test": 1000},
    "disulfide_bridge": {"train": 8000, "val": 1000, "test": 1000},
}
EXPECTED_TRAINABLE_PARAMETERS = {
    "head_to_tail": 2359296,
    "sc_to_tail": 4718592,
    "disulfide_bridge": 2359296,
}


class NoTokenDataset(Dataset):
    def __init__(self, smiles_values, tokenizer, max_length):
        if max_length < 3:
            raise ValueError("max_length must be at least 3")
        if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define BOS and EOS token IDs.")
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define a PAD token ID.")

        self.examples = []
        for smiles in smiles_values:
            smiles_ids = tokenizer.encode(
                smiles,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length - 2,
            )
            input_ids = [tokenizer.bos_token_id, *smiles_ids, tokenizer.eos_token_id]
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one topology-specific LoRA per class without condition tokens."
    )
    parser.add_argument("--base_model_path", type=str, required=False)
    parser.add_argument("--tokenizer_path", type=str, required=False)
    parser.add_argument("--split_manifest_path", type=str, required=False)
    parser.add_argument("--expected_manifest_sha256", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--peptide_types", nargs="*", default=None)
    parser.add_argument("--lora_settings", default=None)
    parser.add_argument("--expected_vocab_size", type=int, default=2140)
    parser.add_argument("--expected_bos_token_id", type=int, default=1)
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
    parser.add_argument("--max_train_samples_per_type", type=int, default=None)
    parser.add_argument("--max_val_samples_per_type", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--audit_frozen_parameters", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    args = parse_args_with_config(build_parser())
    missing = [
        key
        for key in (
            "base_model_path",
            "tokenizer_path",
            "split_manifest_path",
            "expected_manifest_sha256",
            "output_dir",
            "lora_settings",
        )
        if not getattr(args, key, None)
    ]
    if missing:
        raise ValueError("Missing required settings: " + ", ".join(missing))
    args.peptide_types = canonicalize_peptide_types(args.peptide_types)
    if set(args.peptide_types) != set(EXPECTED_SPLITS):
        raise ValueError(
            "LoRA-only ablation requires exactly head_to_tail, sc_to_tail, and disulfide_bridge."
        )
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_manifest(args):
    manifest_path = Path(args.split_manifest_path)
    actual_sha256 = sha256_file(manifest_path)
    if actual_sha256 != args.expected_manifest_sha256:
        raise RuntimeError(
            f"Split manifest SHA-256 mismatch: expected {args.expected_manifest_sha256}, "
            f"received {actual_sha256}."
        )
    dataframe = pd.read_csv(manifest_path)
    required = {"peptide_type", "split", "split_position", "SMILES"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    dataframe = dataframe.sort_values(
        ["peptide_type", "split", "split_position"], kind="stable"
    )

    counts = {}
    for peptide_type in args.peptide_types:
        counts[peptide_type] = {
            split: int(
                ((dataframe["peptide_type"] == peptide_type) & (dataframe["split"] == split)).sum()
            )
            for split in ("train", "val", "test")
        }
    if counts != EXPECTED_SPLITS:
        raise RuntimeError(f"Split count mismatch: expected {EXPECTED_SPLITS}, received {counts}.")
    return dataframe, counts, actual_sha256


def audit_tokenizer(tokenizer, args) -> None:
    if len(tokenizer) != args.expected_vocab_size:
        raise RuntimeError(
            f"Tokenizer vocabulary changed: expected {args.expected_vocab_size}, received {len(tokenizer)}."
        )
    if tokenizer.bos_token_id != args.expected_bos_token_id:
        raise RuntimeError(
            f"Unexpected BOS token ID: expected {args.expected_bos_token_id}, "
            f"received {tokenizer.bos_token_id}."
        )
    present = [token for token in TYPE_PREFIX_TOKENS.values() if token in tokenizer.get_vocab()]
    if present:
        raise RuntimeError(f"Topology tokens must not be present: {present}")


def build_dataset(dataframe, peptide_type, split, tokenizer, max_length, max_samples):
    rows = dataframe[
        (dataframe["peptide_type"] == peptide_type) & (dataframe["split"] == split)
    ]
    if max_samples is not None:
        rows = rows.iloc[:max_samples]
    dataset = NoTokenDataset(rows["SMILES"].tolist(), tokenizer, max_length)
    mismatches = sum(
        int(example["input_ids"][0].item() != tokenizer.bos_token_id)
        for example in dataset
    )
    if mismatches:
        raise RuntimeError(
            f"BOS prefix audit failed for {peptide_type}/{split}: {mismatches} mismatches."
        )
    return dataset


def frozen_parameter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def train_type(args, dataframe, tokenizer, peptide_type, output_dir):
    seed_everything(args.seed)
    settings = args.lora_settings[peptide_type]
    adapter_dir = output_dir / f"adapter_{peptide_type}"
    checkpoint_dir = adapter_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = build_dataset(
        dataframe,
        peptide_type,
        "train",
        tokenizer,
        args.max_length,
        args.max_train_samples_per_type,
    )
    val_dataset = build_dataset(
        dataframe,
        peptide_type,
        "val",
        tokenizer,
        args.max_length,
        args.max_val_samples_per_type,
    )

    base_model = GPT2LMHeadModel.from_pretrained(args.base_model_path)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        bias="none",
        r=int(settings["r"]),
        lora_alpha=int(settings["lora_alpha"]),
        lora_dropout=float(settings["lora_dropout"]),
        target_modules=list(settings["target_modules"]),
        modules_to_save=None,
        trainable_token_indices=None,
    )
    model = get_peft_model(base_model, lora_config)
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    trainable_count = sum(param.numel() for _, param in trainable)
    unexpected = [
        name
        for name, _ in trainable
        if "lora_" not in name or "wte" in name or "lm_head" in name or "trainable_tokens" in name
    ]
    expected_count = EXPECTED_TRAINABLE_PARAMETERS[peptide_type]
    if trainable_count != expected_count or unexpected:
        raise RuntimeError(
            f"LoRA-only parameter audit failed for {peptide_type}: expected={expected_count}, "
            f"actual={trainable_count}, unexpected={unexpected}."
        )
    logger.info("[%s] trainable parameter count: %s", peptide_type, trainable_count)

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
        run_name=f"lora_only_no_token_{peptide_type}_seed42",
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
        raise RuntimeError(f"Frozen prior changed while training {peptide_type}.")

    trainer.model.save_pretrained(adapter_dir)
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "peptide_type": peptide_type,
        "base_model_path": args.base_model_path,
        "tokenizer_path": args.tokenizer_path,
        "split_manifest_path": args.split_manifest_path,
        "seed": args.seed,
        "lora_config": settings,
        "trainable_parameters": trainable_count,
        "expected_trainable_parameters": expected_count,
        "trainable_parameter_names": [name for name, _ in trainable],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_loss": train_output.training_loss,
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "frozen_parameter_hash_before": frozen_hash_before,
        "frozen_parameter_hash_after": frozen_hash_after,
    }
    write_json(adapter_dir / "training_summary.json", summary)

    del trainer
    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite only for an intentional rerun."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe, split_summary, manifest_sha256 = load_and_validate_manifest(args)
    tokenizer = load_tokenizer(args.tokenizer_path, model_max_length=args.max_length)
    audit_tokenizer(tokenizer, args)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    write_json(output_dir / "split_summary.json", split_summary)

    summaries = {}
    for peptide_type in args.peptide_types:
        summaries[peptide_type] = train_type(
            args, dataframe, tokenizer, peptide_type, output_dir
        )
    write_json(
        output_dir / "training_run_summary.json",
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "manifest_sha256": manifest_sha256,
            "split_summary": split_summary,
            "seed": args.seed,
            "command": " ".join(shlex.quote(value) for value in sys.argv),
            "adapters": summaries,
        },
    )


if __name__ == "__main__":
    main()
