from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.shared.pipeline_utils import load_tokenizer, parse_args_with_config, seed_everything
from model.training.lora.dataset import PEPTIDE_TYPES, PeptideDataProcessor, canonicalize_peptide_types
from model.training.lora.trainer import LoRAPeptideTrainer, is_main_process

logging.basicConfig(
    level=logging.INFO if is_main_process() else logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_repo_path(path_str: str, *, allow_model_hub: bool = False) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    project_candidate = PROJECT_ROOT / path
    model_candidate = PROJECT_ROOT / "model" / path

    if str(path_str).startswith("../"):
        return str(model_candidate.resolve())
    if str(path_str).startswith(("./", "data/", "output/", "configs/", "evaluation/", "model/", "scripts/")):
        return str(project_candidate.resolve())
    if project_candidate.exists():
        return str(project_candidate.resolve())
    if allow_model_hub:
        return path_str
    return str(project_candidate.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train type-specific LoRA adapters for cyclic peptide generation.")
    parser.add_argument("--base_model_path", type=str, default="output/best_model_with_mask_trainer/checkpoint-177198")
    parser.add_argument("--tokenizer_path", type=str, default="output/best_model_with_mask_trainer/checkpoint-177198")
    parser.add_argument("--data_path", type=str, default="data/restored_validation.csv")
    parser.add_argument("--output_dir", type=str, default='runs/lora_type_adapters_all')
    parser.add_argument("--peptide_types", nargs="*", default=None)
    parser.add_argument("--target_type", type=str, default=None)
    parser.add_argument("--all_types", nargs="*", default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples_per_type", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=576)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--target_modules", nargs="*", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_args_with_config(parser)
    if args.lr is not None:
        args.learning_rate = args.lr

    if args.peptide_types:
        requested_types = args.peptide_types
    elif args.all_types:
        requested_types = args.all_types
    elif args.target_type:
        requested_types = [args.target_type]
    else:
        requested_types = list(PEPTIDE_TYPES)

    args.peptide_types = canonicalize_peptide_types(requested_types)
    args.base_model_path = resolve_repo_path(args.base_model_path)
    args.tokenizer_path = resolve_repo_path(args.tokenizer_path, allow_model_hub=True)
    args.data_path = resolve_repo_path(args.data_path)
    args.output_dir = resolve_repo_path(args.output_dir)
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    logger.info("Training args: %s", vars(args))

    tokenizer = load_tokenizer(args.tokenizer_path, model_max_length=args.max_length)
    processor = PeptideDataProcessor(tokenizer=tokenizer, max_length=args.max_length)
    data_splits, split_summaries = processor.load_and_split(
        csv_path=args.data_path,
        peptide_types=args.peptide_types,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples_per_type=args.max_samples_per_type,
    )

    trainer = LoRAPeptideTrainer(
        base_model_path=args.base_model_path,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        seed=args.seed,
        base_tokenizer_path=args.tokenizer_path,
        command_args=sys.argv,
    )
    trainer.train_types(
        data_splits=data_splits,
        split_summaries=split_summaries,
        peptide_types=args.peptide_types,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        grad_accum_steps=args.grad_accum_steps,
        early_stop_patience=args.early_stop_patience,
        save_total_limit=args.save_total_limit,
        log_steps=args.log_steps,
        weight_decay=args.weight_decay,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    logger.info("LoRA training finished.")


if __name__ == "__main__":
    main()
