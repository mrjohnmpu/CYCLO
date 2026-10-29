from __future__ import annotations

import json
import logging
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
from peft import get_peft_model
from transformers import (
    EarlyStoppingCallback,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from model.training.lora.dataset import (
    PEPTIDE_TYPES,
    TypeSpecificDataset,
    add_condition_tokens_to_tokenizer,
)
from model.training.lora.lora_config import LoRAConfigFactory

logger = logging.getLogger(__name__)


def is_main_process() -> bool:
    return True


class LoRAPeptideTrainer:
    def __init__(
        self,
        *,
        base_model_path: str,
        tokenizer,
        output_dir: str,
        seed: int,
        base_tokenizer_path: str,
        command_args: Sequence[str] | None = None,
    ):
        self.base_model_path = base_model_path
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_dir = self.output_dir / "tokenizer"
        self.seed = seed
        self.base_tokenizer_path = base_tokenizer_path
        self.command_args = list(command_args or sys.argv)

    def prepare_tokenizer(self, peptide_types: Sequence[str]) -> None:
        add_condition_tokens_to_tokenizer(self.tokenizer, peptide_types)
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(self.tokenizer_dir))
        logger.info("Saved shared tokenizer to %s", self.tokenizer_dir)

    def train_types(
        self,
        *,
        data_splits: dict[str, dict[str, TypeSpecificDataset]],
        split_summaries: dict[str, dict[str, int]],
        peptide_types: Sequence[str],
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        warmup_ratio: float,
        grad_accum_steps: int,
        early_stop_patience: int,
        save_total_limit: int,
        log_steps: int,
        weight_decay: float,
        lora_r: int | None = None,
        lora_alpha: int | None = None,
        lora_dropout: float | None = None,
        target_modules: Sequence[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        self.prepare_tokenizer(peptide_types)
        results: dict[str, dict[str, Any]] = {}

        for peptide_type in peptide_types:
            datasets = data_splits[peptide_type]
            summary = self.train_single_type(
                peptide_type=peptide_type,
                train_dataset=datasets["train"],
                val_dataset=datasets["val"],
                test_dataset=datasets["test"],
                split_summary=split_summaries[peptide_type],
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                grad_accum_steps=grad_accum_steps,
                early_stop_patience=early_stop_patience,
                save_total_limit=save_total_limit,
                log_steps=log_steps,
                weight_decay=weight_decay,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            results[peptide_type] = summary

        summary_path = self.output_dir / "training_run_summary.json"
        summary_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        logger.info("Saved run summary to %s", summary_path)
        return results

    def train_single_type(
        self,
        *,
        peptide_type: str,
        train_dataset: TypeSpecificDataset,
        val_dataset: TypeSpecificDataset,
        test_dataset: TypeSpecificDataset,
        split_summary: dict[str, int],
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        warmup_ratio: float,
        grad_accum_steps: int,
        early_stop_patience: int,
        save_total_limit: int,
        log_steps: int,
        weight_decay: float,
        lora_r: int | None,
        lora_alpha: int | None,
        lora_dropout: float | None,
        target_modules: Sequence[str] | None,
    ) -> dict[str, Any]:
        if peptide_type not in PEPTIDE_TYPES:
            raise ValueError(f"Unsupported peptide type: {peptide_type}")

        adapter_dir = self.output_dir / f"adapter_{peptide_type}"
        checkpoint_dir = adapter_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting LoRA training for %s", peptide_type)
        base_model = GPT2LMHeadModel.from_pretrained(self.base_model_path)
        base_model.resize_token_embeddings(len(self.tokenizer))

        lora_config, hyperparameters = LoRAConfigFactory.build_config(
            peptide_type,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        peft_model = get_peft_model(base_model, lora_config)

        trainable_params, total_params = peft_model.get_nb_trainable_parameters()
        logger.info(
            "[%s] trainable params: %s / %s",
            peptide_type,
            trainable_params,
            total_params,
        )

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=max(1, batch_size),
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            gradient_accumulation_steps=grad_accum_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=log_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            seed=self.seed,
            data_seed=self.seed,
            disable_tqdm=False,
            label_names=["labels"],
            run_name=f"lora_{peptide_type}",
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stop_patience,
                    early_stopping_threshold=0.0,
                )
            ],
        )

        train_output = trainer.train()
        val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

        summary = {
            "peptide_type": peptide_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "base_model_path": self.base_model_path,
            "base_tokenizer_path": self.base_tokenizer_path,
            "tokenizer_path": str(self.tokenizer_dir),
            "adapter_path": str(adapter_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "seed": self.seed,
            "split_summary": split_summary,
            "lora_config": hyperparameters.to_dict(),
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "train_loss": train_output.training_loss,
            "best_eval_loss": trainer.state.best_metric,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "command": " ".join(shlex.quote(arg) for arg in self.command_args),
        }

        trainer.model.save_pretrained(str(adapter_dir))
        summary_path = adapter_dir / "training_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        logger.info("Saved adapter and summary for %s to %s", peptide_type, adapter_dir)

        del trainer
        del peft_model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return summary
