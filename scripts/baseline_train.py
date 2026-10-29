#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.moses_baselines import runtime


def build_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a baseline peptide generator.")
    parser.add_argument("--model", required=True, choices=runtime.MODEL_CHOICES)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--val_csv", type=Path, required=True)
    parser.add_argument("--cyclization_type", type=str, required=True)
    parser.add_argument("--run_root", type=Path, default=runtime.DEFAULT_RUN_ROOT)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=runtime.DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=runtime.DEFAULT_SEED)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--save_frequency", type=int, default=20)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--model")
    known, _ = probe.parse_known_args(argv)

    parser = build_common_parser()
    if known.model:
        try:
            model_name = runtime.normalize_model_name(known.model)
        except ValueError as exc:
            parser.error(str(exc))
        runtime.add_model_train_args(parser, model_name)

    args = parser.parse_args(argv)
    args.model = runtime.normalize_model_name(args.model)
    return args


def main() -> None:
    args = parse_args()
    runtime.prepare_device(args.device)
    runtime.seed_everything(args.seed)

    run_dir = runtime.ensure_new_run_dir(
        runtime.resolve_run_dir(args.run_root, args.model, args.cyclization_type, args.run_name)
    )
    run_paths = runtime.ensure_run_dirs(run_dir, ("checkpoints", "logs", "artifacts"))

    train_smiles = runtime.load_filtered_smiles(args.train_csv, args.cyclization_type, limit=args.train_limit)
    val_smiles = runtime.load_filtered_smiles(args.val_csv, args.cyclization_type, limit=args.val_limit)

    config_dict = vars(args).copy()
    config_dict.update(
        {
            "model": args.model,
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "train_csv": str(Path(args.train_csv).resolve()),
            "val_csv": str(Path(args.val_csv).resolve()),
            "model_save": str(run_paths["model_path"]),
            "vocab_save": str(run_paths["vocab_path"]),
            "config_save": str(run_paths["config_pt_path"]),
            "log_file": str(run_paths["train_log_path"]),
        }
    )
    runtime.save_model_config(config_dict, run_paths)

    config = runtime.to_namespace(config_dict)
    trainer = runtime.build_trainer(args.model, config)
    vocabulary = trainer.get_vocabulary(train_smiles)
    torch.save(vocabulary, run_paths["vocab_path"])

    model = runtime.build_model(args.model, vocabulary, config).to(args.device)
    start = time.time()
    trainer.fit(model, train_smiles, val_smiles)
    elapsed = time.time() - start

    model = model.to("cpu")
    torch.save(model.state_dict(), run_paths["model_path"])

    manifest = {
        "model": args.model,
        "cyclization_type": args.cyclization_type,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "train_csv": str(Path(args.train_csv).resolve()),
        "val_csv": str(Path(args.val_csv).resolve()),
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "num_train_rows": len(train_smiles),
        "num_val_rows": len(val_smiles),
        "seed": args.seed,
        "device": args.device,
        "save_frequency": args.save_frequency,
        "elapsed_seconds": elapsed,
        "checkpoint_paths": {
            "model": str(run_paths["model_path"]),
            "vocab": str(run_paths["vocab_path"]),
            "config": str(run_paths["config_pt_path"]),
        },
        "log_file": str(run_paths["train_log_path"]),
    }
    runtime.save_json(run_paths["manifest_path"], manifest)

    print(f"Saved run directory: {run_dir}")
    print(f"Model checkpoint: {run_paths['model_path']}")
    print(f"Vocab checkpoint: {run_paths['vocab_path']}")
    print(f"Model config: {run_paths['config_pt_path']}")
    print(f"Train manifest: {run_paths['manifest_path']}")


if __name__ == "__main__":
    main()
