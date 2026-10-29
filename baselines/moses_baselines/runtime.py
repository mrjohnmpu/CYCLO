from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "baselines" / "runs"
DEFAULT_DEVICE = "cuda:1"
DEFAULT_MAX_LEN = 576
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_SAMPLES = 30000
DEFAULT_SEED = 42
TYPE_COL = "Cyclization"
SMILES_COL = "SMILES"
MODEL_CHOICES = ("aae", "vae", "char_rnn", "rnn")


def ensure_python_paths() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from utils.release_paths import add_moses_path
    add_moses_path()


ensure_python_paths()

from baselines.moses_baselines.aae import AAE as RawAAE, AAETrainer, aae_parser  # noqa: E402
from baselines.moses_baselines.char_rnn import CharRNN, CharRNNTrainer, char_rnn_parser  # noqa: E402
from baselines.moses_baselines.vae import VAE, VAETrainer, vae_parser  # noqa: E402

RDLogger.DisableLog("rdApp.*")


class SafeAAE(RawAAE):
    """AAE wrapper that keeps packed-sequence lengths on CPU."""

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

            for _ in range(max_len):
                logits, _, states = self.decoder(prevs, one_lens, states, not samples)
                logits = torch.softmax(logits, 2)
                shape = logits.shape[:-1]
                logits = logits.contiguous().view(-1, logits.shape[-1])
                currents = torch.distributions.Categorical(logits).sample().view(shape)

                is_end[currents.view(-1) == self.vocabulary.eos] = True
                if bool(torch.all(is_end)):
                    break

                currents[is_end, :] = self.vocabulary.pad
                samples.append(currents.cpu())
                lengths[~is_end] += 1
                prevs = currents

            if not samples:
                return ["" for _ in range(n_batch)]

            samples = torch.cat(samples, dim=-1)
            return [self.tensor2string(t[:l]) for t, l in zip(samples, lengths.tolist())]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    aliases: tuple[str, ...]
    model_cls: type
    trainer_cls: type
    parser_factory: Callable[[Any], Any]
    sample_arg_name: str


_MODEL_SPECS = {
    "aae": ModelSpec(
        name="aae",
        aliases=(),
        model_cls=SafeAAE,
        trainer_cls=AAETrainer,
        parser_factory=aae_parser,
        sample_arg_name="max_len",
    ),
    "vae": ModelSpec(
        name="vae",
        aliases=(),
        model_cls=VAE,
        trainer_cls=VAETrainer,
        parser_factory=vae_parser,
        sample_arg_name="max_len",
    ),
    "char_rnn": ModelSpec(
        name="char_rnn",
        aliases=("rnn",),
        model_cls=CharRNN,
        trainer_cls=CharRNNTrainer,
        parser_factory=char_rnn_parser,
        sample_arg_name="max_length",
    ),
}
_ALIAS_TO_MODEL = {alias: spec.name for spec in _MODEL_SPECS.values() for alias in spec.aliases}


def normalize_model_name(name: str) -> str:
    normalized = str(name).strip().lower()
    normalized = _ALIAS_TO_MODEL.get(normalized, normalized)
    if normalized not in _MODEL_SPECS:
        raise ValueError(f"Unsupported model '{name}'. Expected one of {list(MODEL_CHOICES)}")
    return normalized


def get_model_spec(name: str) -> ModelSpec:
    return _MODEL_SPECS[normalize_model_name(name)]


def add_model_train_args(parser, model_name: str):
    spec = get_model_spec(model_name)
    return spec.parser_factory(parser)


def prepare_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type.startswith("cuda"):
        torch.cuda.set_device(device.index or 0)
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(str(smiles).strip()) is not None


def load_filtered_smiles(
    csv_path: str | Path,
    cyclization_type: str | None,
    *,
    smiles_col: str = SMILES_COL,
    type_col: str = TYPE_COL,
    limit: int | None = None,
) -> list[str]:
    path = Path(csv_path).resolve()
    dataframe = pd.read_csv(path)
    if smiles_col not in dataframe.columns:
        raise ValueError(f"{path} is missing required column '{smiles_col}'")
    if cyclization_type is not None and type_col in dataframe.columns:
        dataframe = dataframe[dataframe[type_col].astype(str) == str(cyclization_type)]
    smiles = dataframe[smiles_col].dropna().astype(str).str.strip()
    smiles = smiles[smiles != ""]
    if limit is not None:
        smiles = smiles.iloc[:limit]
    values = smiles.tolist()
    if not values:
        raise ValueError(f"No SMILES found in {path} after filtering cyclization_type={cyclization_type}")
    return values


def make_run_name(model_name: str, cyclization_type: str, run_name: str | None = None) -> str:
    if run_name:
        return run_name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{normalize_model_name(model_name)}_{cyclization_type}_{timestamp}"


def resolve_run_dir(run_root: str | Path, model_name: str, cyclization_type: str, run_name: str | None = None) -> Path:
    normalized_model = normalize_model_name(model_name)
    resolved_name = make_run_name(normalized_model, cyclization_type, run_name)
    return Path(run_root).resolve() / normalized_model / str(cyclization_type) / resolved_name


def get_run_paths(run_dir: str | Path) -> dict[str, Path]:
    root = Path(run_dir).resolve()
    checkpoints_dir = root / "checkpoints"
    logs_dir = root / "logs"
    artifacts_dir = root / "artifacts"
    generated_dir = root / "generated"
    metrics_dir = root / "metrics"
    return {
        "run_dir": root,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "artifacts_dir": artifacts_dir,
        "generated_dir": generated_dir,
        "metrics_dir": metrics_dir,
        "model_path": checkpoints_dir / "model.pt",
        "vocab_path": checkpoints_dir / "vocab.pt",
        "config_pt_path": checkpoints_dir / "model_config.pt",
        "train_log_path": logs_dir / "train_log.csv",
        "manifest_path": artifacts_dir / "train_manifest.json",
        "model_config_json_path": artifacts_dir / "model_config.json",
    }


def ensure_run_dirs(run_dir: str | Path, sections: tuple[str, ...]) -> dict[str, Path]:
    paths = get_run_paths(run_dir)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    section_to_dir_key = {
        "checkpoints": "checkpoints_dir",
        "logs": "logs_dir",
        "artifacts": "artifacts_dir",
        "generated": "generated_dir",
        "metrics": "metrics_dir",
    }
    for section in sections:
        paths[section_to_dir_key[section]].mkdir(parents=True, exist_ok=True)
    return paths


def ensure_new_run_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory already exists and is not empty: {path}")
    return path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_model_config(config_dict: dict[str, Any], run_paths: dict[str, Path]) -> None:
    serializable = to_jsonable(config_dict)
    torch.save(serializable, run_paths["config_pt_path"])
    save_json(run_paths["model_config_json_path"], serializable)


def build_model(model_name: str, vocabulary, config: SimpleNamespace):
    spec = get_model_spec(model_name)
    return spec.model_cls(vocabulary, config)


def build_trainer(model_name: str, config: SimpleNamespace):
    spec = get_model_spec(model_name)
    return spec.trainer_cls(config)


def to_namespace(config_dict: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**config_dict)


def load_trained_run(run_dir: str | Path, device: str):
    run_paths = get_run_paths(run_dir)
    manifest = load_json(run_paths["manifest_path"])
    model_name = normalize_model_name(manifest["model"])
    config_dict = torch.load(run_paths["config_pt_path"], map_location="cpu", weights_only=False)
    config = to_namespace(config_dict)
    vocabulary = torch.load(run_paths["vocab_path"], map_location="cpu", weights_only=False)
    model = build_model(model_name, vocabulary, config)
    state_dict = torch.load(run_paths["model_path"], map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    prepare_device(device)
    model = model.to(device)
    model.eval()
    return manifest, config_dict, config, vocabulary, model


def sample_batch(model_name: str, model, n_batch: int, max_len: int) -> list[str]:
    spec = get_model_spec(model_name)
    return model.sample(n_batch, **{spec.sample_arg_name: max_len})


def build_generation_records(
    smiles_list: list[str],
    *,
    model_name: str,
    run_name: str,
    run_dir: str,
    cyclization_type: str,
) -> list[dict[str, Any]]:
    records = []
    for smiles in smiles_list:
        cleaned = str(smiles).strip()
        records.append(
            {
                "SMILES": cleaned,
                "Valid": is_valid_smiles(cleaned),
                "Cyclization": cyclization_type,
                "model_name": model_name,
                "run_name": run_name,
                "run_dir": run_dir,
            }
        )
    return records


def default_output_name(model_name: str, cyclization_type: str, num_samples: int) -> str:
    return f"{normalize_model_name(model_name)}_{cyclization_type}_n{num_samples}"
