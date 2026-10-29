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
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, GPT2LMHeadModel, Trainer, TrainingArguments, default_data_collator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.shared.pipeline_utils import load_tokenizer, parse_args_with_config, seed_everything
from model.training.lora.dataset import TYPE_PREFIX_TOKENS, canonicalize_peptide_types

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_SPLITS = {
    "head_to_tail": {"train": 7999, "val": 1000, "test": 1000},
    "sc_to_tail": {"train": 8001, "val": 1000, "test": 1000},
    "disulfide_bridge": {"train": 8000, "val": 1000, "test": 1000},
}


class StrictTokenDataset(Dataset):
    def __init__(self, smiles_values, token_id, tokenizer, max_length):
        if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define EOS and PAD token IDs.")
        self.examples = []
        for smiles in smiles_values:
            smiles_ids = tokenizer.encode(
                smiles, add_special_tokens=False, truncation=True, max_length=max_length - 2
            )
            input_ids = [token_id, *smiles_ids, tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            padding = max_length - len(input_ids)
            input_ids.extend([tokenizer.pad_token_id] * padding)
            attention_mask.extend([0] * padding)
            labels = [value if mask else -100 for value, mask in zip(input_ids, attention_mask)]
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


def build_parser():
    parser = argparse.ArgumentParser(description="Full-finetune topology-specific GPT-2 models with strict tokens.")
    parser.add_argument("--base_model_path", required=False)
    parser.add_argument("--tokenizer_path", required=False)
    parser.add_argument("--split_manifest_path", required=False)
    parser.add_argument("--expected_manifest_sha256", required=False)
    parser.add_argument("--output_dir", required=False)
    parser.add_argument("--peptide_types", nargs="*", default=None)
    parser.add_argument("--expected_base_vocab_size", type=int, default=2140)
    parser.add_argument("--expected_condition_token_ids", default=None)
    parser.add_argument("--expected_trainable_parameters", type=int, default=87488256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
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
    parser.add_argument("--audit_parameter_updates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args():
    args = parse_args_with_config(build_parser())
    required = (
        "base_model_path", "tokenizer_path", "split_manifest_path",
        "expected_manifest_sha256", "output_dir", "expected_condition_token_ids",
    )
    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise ValueError("Missing required settings: " + ", ".join(missing))
    args.peptide_types = canonicalize_peptide_types(args.peptide_types)
    if set(args.peptide_types) != set(EXPECTED_SPLITS):
        raise ValueError("Full-finetune ablation requires exactly the three cyclic topologies.")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(args):
    actual_hash = sha256_file(args.split_manifest_path)
    if actual_hash != args.expected_manifest_sha256:
        raise RuntimeError(f"Manifest hash mismatch: {actual_hash}")
    dataframe = pd.read_csv(args.split_manifest_path)
    required = {"peptide_type", "split", "split_position", "SMILES"}
    if not required.issubset(dataframe.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(required - set(dataframe.columns))}")
    dataframe = dataframe.sort_values(["peptide_type", "split", "split_position"], kind="stable")
    counts = {
        peptide_type: {
            split: int(((dataframe["peptide_type"] == peptide_type) & (dataframe["split"] == split)).sum())
            for split in ("train", "val", "test")
        }
        for peptide_type in args.peptide_types
    }
    if counts != EXPECTED_SPLITS:
        raise RuntimeError(f"Split mismatch: expected {EXPECTED_SPLITS}, received {counts}")
    return dataframe, counts, actual_hash


def prepare_tokenizer(args):
    tokenizer = load_tokenizer(args.tokenizer_path, model_max_length=args.max_length)
    if len(tokenizer) != args.expected_base_vocab_size:
        raise RuntimeError(f"Unexpected base vocabulary size: {len(tokenizer)}")
    tokens = [TYPE_PREFIX_TOKENS[value] for value in args.peptide_types]
    added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    if added != len(tokens):
        raise RuntimeError(f"Expected to add {len(tokens)} tokens, added {added}")
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    expected = [int(args.expected_condition_token_ids[value]) for value in args.peptide_types]
    if token_ids != expected:
        raise RuntimeError(f"Token IDs mismatch: expected {expected}, received {token_ids}")
    return tokenizer, dict(zip(args.peptide_types, token_ids))


def build_dataset(dataframe, peptide_type, split, token_id, tokenizer, args, max_samples):
    rows = dataframe[(dataframe["peptide_type"] == peptide_type) & (dataframe["split"] == split)]
    if max_samples is not None:
        rows = rows.iloc[:max_samples]
    dataset = StrictTokenDataset(rows["SMILES"].tolist(), token_id, tokenizer, args.max_length)
    mismatches = sum(int(item["input_ids"][0].item() != token_id) for item in dataset)
    if mismatches:
        raise RuntimeError(f"Condition-token audit failed for {peptide_type}/{split}: {mismatches}")
    return dataset


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def update_audit_hashes(model):
    block_digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.startswith("transformer.h."):
            block_digest.update(name.encode("utf-8"))
            block_digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return {
        "transformer_blocks": block_digest.hexdigest(),
        "tied_wte_lm_head": tensor_hash(model.transformer.wte.weight),
    }


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def train_type(args, dataframe, tokenizer, token_ids, peptide_type, output_dir):
    seed_everything(args.seed)
    token_id = token_ids[peptide_type]
    model_dir = output_dir / f"model_{peptide_type}"
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = build_dataset(
        dataframe, peptide_type, "train", token_id, tokenizer, args,
        args.max_train_samples_per_type,
    )
    val_dataset = build_dataset(
        dataframe, peptide_type, "val", token_id, tokenizer, args,
        args.max_val_samples_per_type,
    )

    model = GPT2LMHeadModel.from_pretrained(args.base_model_path)
    model.resize_token_embeddings(len(tokenizer))
    for parameter in model.parameters():
        parameter.requires_grad = True
    if model.transformer.wte.weight.data_ptr() != model.lm_head.weight.data_ptr():
        raise RuntimeError("GPT-2 input embeddings and LM head are expected to be tied.")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    special_names = [name for name, _ in model.named_parameters() if "lora_" in name or "trainable_tokens" in name]
    if parameter_count != args.expected_trainable_parameters or trainable_count != parameter_count or special_names:
        raise RuntimeError(
            f"Full-finetune parameter audit failed: total={parameter_count}, trainable={trainable_count}, special={special_names}"
        )
    logger.info("[%s] token_id=%s trainable_parameters=%s", peptide_type, token_id, trainable_count)

    step_schedule = args.max_steps > 0
    schedule = {
        "eval_strategy": "steps" if step_schedule else "epoch",
        "save_strategy": "steps" if step_schedule else "epoch",
    }
    if step_schedule:
        schedule.update({"max_steps": args.max_steps, "eval_steps": args.max_steps, "save_steps": args.max_steps})
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
        run_name=f"full_finetune_strict_token_{peptide_type}_seed42",
        **schedule,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
    )
    hashes_before = update_audit_hashes(model) if args.audit_parameter_updates else None
    train_output = trainer.train()
    hashes_after = update_audit_hashes(model) if args.audit_parameter_updates else None
    if hashes_before is not None:
        unchanged = [name for name in hashes_before if hashes_before[name] == hashes_after[name]]
        if unchanged:
            raise RuntimeError(f"Expected full-finetune parameter groups did not change: {unchanged}")

    trainer.save_model(str(model_dir))
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "peptide_type": peptide_type,
        "condition_token": TYPE_PREFIX_TOKENS[peptide_type],
        "condition_token_id": token_id,
        "seed": args.seed,
        "trainable_parameters": trainable_count,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_loss": train_output.training_loss,
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "parameter_hashes_before": hashes_before,
        "parameter_hashes_after": hashes_after,
    }
    write_json(model_dir / "training_summary.json", summary)
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe, splits, manifest_hash = load_manifest(args)
    tokenizer, token_ids = prepare_tokenizer(args)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    write_json(output_dir / "split_summary.json", splits)
    summaries = {}
    for peptide_type in args.peptide_types:
        summaries[peptide_type] = train_type(args, dataframe, tokenizer, token_ids, peptide_type, output_dir)
    write_json(
        output_dir / "training_run_summary.json",
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "manifest_sha256": manifest_hash,
            "split_summary": splits,
            "condition_token_ids": token_ids,
            "seed": args.seed,
            "command": " ".join(shlex.quote(value) for value in sys.argv),
            "models": summaries,
        },
    )


if __name__ == "__main__":
    main()
