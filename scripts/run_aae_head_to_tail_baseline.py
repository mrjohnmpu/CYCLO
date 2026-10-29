#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path('runs/baselines/aae_head_to_tail_vs_lora_20260422')
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "restored_train.csv"
DEFAULT_VAL_CSV = PROJECT_ROOT / "data" / "restored_validation.csv"
DEFAULT_LORA_CSV = Path(
    'runs/lora/generation_eval_n30000/head_to_tail_n30000.csv'
)
DEFAULT_METRICS_SCRIPT = PROJECT_ROOT / "evaluation" / "metrics" / "compute_moses_metrics.py"
SUMMARY_FIELDS = [
    ("Validity", "validity"),
    ("Uniqueness@10000", "unique@10000"),
    ("Uniqueness@20000", "unique@20000"),
    ("Uniqueness@30000", "unique@30000"),
    ("Novelty", "Novelty"),
    ("IntDiv", "IntDiv"),
    ("IntDiv2", "IntDiv2"),
    ("SNN/Test", "SNN/Test"),
    ("FCD/Test", "FCD/Test"),
    ("num_total", "num_total"),
    ("num_valid", "num_valid"),
]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.release_paths import add_moses_path
MOSES_ROOT = add_moses_path()

from baselines.moses_baselines.aae import AAE, AAETrainer  # noqa: E402

RDLogger.DisableLog("rdApp.*")


class SafeAAE(AAE):
    """AAE wrapper that keeps packed-sequence length tensors on CPU."""

    def encoder_forward(self, x, lengths, *args, **kwargs):
        if torch.is_tensor(lengths):
            lengths = lengths.detach().cpu()
        return self.encoder(x, lengths, *args, **kwargs)

    def decoder_forward(self, x, lengths, states, is_latent_states=False):
        if torch.is_tensor(lengths):
            lengths = lengths.detach().cpu()
        return self.decoder(x, lengths, states, is_latent_states=is_latent_states)

    def sample(self, n_batch, max_len=100):
        with torch.no_grad():
            samples = []
            lengths = torch.zeros(n_batch, dtype=torch.long, device=self.device)
            states = self.sample_latent(n_batch)
            prevs = torch.empty(n_batch, 1, dtype=torch.long, device=self.device).fill_(self.vocabulary.bos)
            one_lens = torch.ones(n_batch, dtype=torch.long)
            is_end = torch.zeros(n_batch, dtype=torch.bool, device=self.device)

            for i in range(max_len):
                logits, _, states = self.decoder(prevs, one_lens, states, i == 0)
                logits = torch.softmax(logits, 2)
                shape = logits.shape[:-1]
                logits = logits.contiguous().view(-1, logits.shape[-1])
                currents = torch.distributions.Categorical(logits).sample()
                currents = currents.view(shape)

                is_end[currents.view(-1) == self.vocabulary.eos] = True
                if bool(torch.all(is_end)):
                    break

                currents[is_end, :] = self.vocabulary.pad
                samples.append(currents.cpu())
                lengths[~is_end] += 1
                prevs = currents

            if samples:
                samples = torch.cat(samples, dim=-1)
                return [self.tensor2string(t[:l]) for t, l in zip(samples, lengths.tolist())]
            return ["" for _ in range(n_batch)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate an AAE head-to-tail baseline.")
    parser.add_argument("--stage", choices=["smoke", "full", "sample", "all"], default="smoke")
    parser.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val_csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--lora_csv", type=Path, default=DEFAULT_LORA_CSV)
    parser.add_argument("--cyclization_type", type=str, default="head_to_tail")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--max_len", type=int, default=576)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics_n_jobs", type=int, default=8)
    parser.add_argument("--save_frequency", type=int, default=20)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_layout(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "generated": root / "generated",
        "metrics": root / "metrics",
        "summary": root / "summary",
        "artifacts": root / "artifacts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    (paths["metrics"] / "aae").mkdir(parents=True, exist_ok=True)
    (paths["metrics"] / "lora").mkdir(parents=True, exist_ok=True)
    return paths


def load_filtered_smiles(csv_path: Path, cyclization_type: str, limit: int | None = None) -> list[str]:
    smiles: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "Cyclization" not in (reader.fieldnames or []) or "SMILES" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain Cyclization and SMILES columns")
        for row in reader:
            if row.get("Cyclization") != cyclization_type:
                continue
            value = str(row.get("SMILES", "")).strip()
            if not value:
                continue
            smiles.append(value)
            if limit is not None and len(smiles) >= limit:
                break
    if not smiles:
        raise ValueError(f"No SMILES found in {csv_path} for cyclization_type={cyclization_type}")
    return smiles


def compute_length_stats(smiles_list: list[str], max_len: int) -> dict[str, float | int]:
    lengths = sorted(len(item) for item in smiles_list)
    total = len(lengths)

    def percentile(p: float) -> int:
        index = min(total - 1, max(0, int((total - 1) * p)))
        return lengths[index]

    return {
        "count": total,
        "min": lengths[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": lengths[-1],
        "mean": round(sum(lengths) / total, 4),
        "num_equal_max_len": sum(length == max_len for length in lengths),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def build_config(args: argparse.Namespace, paths: dict[str, Path], *, train_epochs: int, n_batch: int) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_size=32,
        encoder_hidden_size=512,
        encoder_num_layers=1,
        encoder_bidirectional=True,
        encoder_dropout=0.0,
        decoder_hidden_size=512,
        decoder_num_layers=2,
        decoder_dropout=0.0,
        latent_size=128,
        discriminator_layers=[640, 256],
        pretrain_epochs=0,
        train_epochs=train_epochs,
        n_batch=n_batch,
        lr=1e-3,
        step_size=20,
        gamma=0.5,
        n_jobs=1,
        n_workers=1,
        discriminator_steps=1,
        weight_decay=0,
        model_save=str(paths["checkpoints"] / "aae_model.pt"),
        save_frequency=args.save_frequency,
        log_file=str(paths["logs"] / "train_log.csv"),
        config_save=str(paths["checkpoints"] / "aae_config.pt"),
        vocab_save=str(paths["checkpoints"] / "aae_vocab.pt"),
        seed=args.seed,
        cyclization_type=args.cyclization_type,
    )


def train_model(train_smiles: list[str], val_smiles: list[str], config: SimpleNamespace, device_str: str) -> None:
    seed_everything(config.seed)
    trainer = AAETrainer(config)
    vocabulary = trainer.get_vocabulary(train_smiles)
    torch.save(vars(config), config.config_save)
    torch.save(vocabulary, config.vocab_save)

    model = SafeAAE(vocabulary, config).to(device_str)
    trainer.fit(model, train_smiles, val_smiles)
    model = model.to("cpu")
    torch.save(model.state_dict(), config.model_save)


def load_trained_model(paths: dict[str, Path], device_str: str) -> tuple[SafeAAE, SimpleNamespace]:
    config_dict = torch.load(paths["checkpoints"] / "aae_config.pt", map_location="cpu", weights_only=False)
    config = SimpleNamespace(**config_dict)
    vocabulary = torch.load(paths["checkpoints"] / "aae_vocab.pt", map_location="cpu", weights_only=False)
    model = SafeAAE(vocabulary, config)
    state_dict = torch.load(paths["checkpoints"] / "aae_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device_str)
    model.eval()
    return model, config


def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def sample_to_csv(
    model: SafeAAE,
    output_csv: Path,
    *,
    num_samples: int,
    batch_size: int,
    max_len: int,
    cyclization_type: str,
    model_path: Path,
) -> dict[str, float | int]:
    records: list[dict[str, object]] = []
    generated = 0
    while generated < num_samples:
        current_batch = min(batch_size, num_samples - generated)
        batch_samples = model.sample(current_batch, max_len=max_len)
        for smiles in batch_samples:
            cleaned = str(smiles).strip()
            records.append(
                {
                    "SMILES": cleaned,
                    "Valid": is_valid_smiles(cleaned),
                    "Cyclization": cyclization_type,
                    "model_path": str(model_path),
                }
            )
        generated += len(batch_samples)

    dataframe = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)
    lengths = [len(str(item)) for item in dataframe["SMILES"].tolist()]
    return {
        "num_rows": int(len(dataframe)),
        "num_valid": int(dataframe["Valid"].sum()),
        "validity": float(dataframe["Valid"].mean()),
        "min_length": int(min(lengths)),
        "p50_length": int(sorted(lengths)[len(lengths) // 2]),
        "max_length": int(max(lengths)),
        "num_equal_max_len": int(sum(length == max_len for length in lengths)),
    }


def compute_metrics(
    *,
    generated_csv: Path,
    output_dir: Path,
    experiment_name: str,
    train_csv: Path,
    reference_csv: Path,
    cyclization_type: str,
    device: str,
    n_jobs: int,
) -> Path:
    env = os.environ.copy()
    pythonpath_parts = [str(MOSES_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    command = [
        sys.executable,
        str(DEFAULT_METRICS_SCRIPT),
        "--generated_csv",
        str(generated_csv),
        "--output_dir",
        str(output_dir),
        "--experiment_name",
        experiment_name,
        "--train_csv",
        str(train_csv),
        "--test_csv",
        str(reference_csv),
        "--cyclization_type",
        cyclization_type,
        "--device",
        device,
        "--n_jobs",
        str(n_jobs),
        "--report_prefix",
        "metrics",
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)
    return output_dir / "metrics.json"


def load_metrics(metrics_json: Path) -> dict:
    return json.loads(metrics_json.read_text(encoding="utf-8"))


def build_summary_row(label: str, source_csv: Path, metrics: dict) -> dict[str, object]:
    row: dict[str, object] = {"Model": label, "source_csv": str(source_csv)}
    for column_name, metric_key in SUMMARY_FIELDS:
        row[column_name] = metrics.get(metric_key)
    return row


def write_summary(output_dir: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    csv_path = output_dir / "aae_vs_lora_head_to_tail.csv"
    md_path = output_dir / "aae_vs_lora_head_to_tail.md"
    dataframe.to_csv(csv_path, index=False)

    columns = list(dataframe.columns)
    lines = ["# AAE vs LoRA Head-to-Tail Summary", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in dataframe.iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def validate_inputs(args: argparse.Namespace) -> None:
    for path in [args.train_csv, args.val_csv, args.lora_csv, DEFAULT_METRICS_SCRIPT]:
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")
    if args.max_len < 1:
        raise ValueError("--max_len must be positive")
    if args.num_samples < 1:
        raise ValueError("--num_samples must be positive")


SMOKE_TRAIN_LIMIT = 2048
SMOKE_VAL_LIMIT = 256
SMOKE_TRAIN_EPOCHS = 1
SMOKE_BATCH_SIZE = 64
SMOKE_NUM_SAMPLES = 256
FULL_TRAIN_EPOCHS = 120
FULL_BATCH_SIZE = 256


def run_smoke(args: argparse.Namespace) -> None:
    smoke_root = args.output_root / "smoke"
    paths = ensure_layout(smoke_root)
    train_smiles = load_filtered_smiles(args.train_csv, args.cyclization_type, limit=SMOKE_TRAIN_LIMIT)
    val_smiles = load_filtered_smiles(args.val_csv, args.cyclization_type, limit=SMOKE_VAL_LIMIT)
    write_json(paths["artifacts"] / "smoke_train_length_stats.json", compute_length_stats(train_smiles, args.max_len))
    write_json(paths["artifacts"] / "smoke_val_length_stats.json", compute_length_stats(val_smiles, args.max_len))

    config = build_config(args, paths, train_epochs=SMOKE_TRAIN_EPOCHS, n_batch=SMOKE_BATCH_SIZE)
    train_model(train_smiles, val_smiles, config, args.device)
    model, _ = load_trained_model(paths, args.device)
    sample_stats = sample_to_csv(
        model,
        paths["generated"] / "aae_head_to_tail_smoke_n256.csv",
        num_samples=SMOKE_NUM_SAMPLES,
        batch_size=SMOKE_BATCH_SIZE,
        max_len=args.max_len,
        cyclization_type=args.cyclization_type,
        model_path=paths["checkpoints"] / "aae_model.pt",
    )
    write_json(paths["artifacts"] / "smoke_generation_stats.json", sample_stats)
    compute_metrics(
        generated_csv=paths["generated"] / "aae_head_to_tail_smoke_n256.csv",
        output_dir=paths["metrics"] / "aae",
        experiment_name="aae_head_to_tail_smoke",
        train_csv=args.train_csv,
        reference_csv=args.val_csv,
        cyclization_type=args.cyclization_type,
        device=args.device,
        n_jobs=args.metrics_n_jobs,
    )


def run_full_train(args: argparse.Namespace) -> dict[str, Path]:
    paths = ensure_layout(args.output_root)
    train_smiles = load_filtered_smiles(args.train_csv, args.cyclization_type)
    val_smiles = load_filtered_smiles(args.val_csv, args.cyclization_type)
    write_json(paths["artifacts"] / "train_length_stats.json", compute_length_stats(train_smiles, args.max_len))
    write_json(paths["artifacts"] / "val_length_stats.json", compute_length_stats(val_smiles, args.max_len))
    config = build_config(args, paths, train_epochs=FULL_TRAIN_EPOCHS, n_batch=FULL_BATCH_SIZE)
    train_model(train_smiles, val_smiles, config, args.device)
    return paths


def run_sampling_and_summary(args: argparse.Namespace) -> None:
    paths = ensure_layout(args.output_root)
    model, config = load_trained_model(paths, args.device)
    output_csv = paths["generated"] / "aae_head_to_tail_n30000.csv"
    sample_stats = sample_to_csv(
        model,
        output_csv,
        num_samples=args.num_samples,
        batch_size=config.n_batch,
        max_len=args.max_len,
        cyclization_type=args.cyclization_type,
        model_path=paths["checkpoints"] / "aae_model.pt",
    )
    write_json(paths["artifacts"] / "generation_stats.json", sample_stats)

    aae_metrics_json = compute_metrics(
        generated_csv=output_csv,
        output_dir=paths["metrics"] / "aae",
        experiment_name="aae_head_to_tail_full",
        train_csv=args.train_csv,
        reference_csv=args.val_csv,
        cyclization_type=args.cyclization_type,
        device=args.device,
        n_jobs=args.metrics_n_jobs,
    )
    lora_metrics_json = compute_metrics(
        generated_csv=args.lora_csv,
        output_dir=paths["metrics"] / "lora",
        experiment_name="lora_head_to_tail_reference",
        train_csv=args.train_csv,
        reference_csv=args.val_csv,
        cyclization_type=args.cyclization_type,
        device=args.device,
        n_jobs=args.metrics_n_jobs,
    )

    summary_rows = [
        build_summary_row("AAE", output_csv, load_metrics(aae_metrics_json)),
        build_summary_row("LoRA", args.lora_csv, load_metrics(lora_metrics_json)),
    ]
    csv_path, md_path = write_summary(paths["summary"], summary_rows)
    write_json(
        paths["artifacts"] / "run_manifest.json",
        {
            "stage": "sample",
            "generated_csv": str(output_csv),
            "aae_metrics_json": str(aae_metrics_json),
            "lora_metrics_json": str(lora_metrics_json),
            "summary_csv": str(csv_path),
            "summary_md": str(md_path),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    start = time.time()
    print(f"Running stage={args.stage} on device={args.device}")
    print(f"Output root: {args.output_root}")

    if args.stage == "smoke":
        run_smoke(args)
    elif args.stage == "full":
        run_full_train(args)
    elif args.stage == "sample":
        run_sampling_and_summary(args)
    elif args.stage == "all":
        run_full_train(args)
        run_sampling_and_summary(args)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")

    elapsed = time.time() - start
    print(f"Completed stage={args.stage} in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
