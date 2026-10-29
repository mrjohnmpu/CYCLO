from pathlib import Path
import os
import random
import json
from typing import Optional, Union

import numpy as np
import torch
from peft import PeftModel
from rdkit import Chem, RDLogger
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_FINETUNE_STATE_FILENAME = "full_model_state.pt"
CHECKPOINT_METADATA_FILENAME = "checkpoint_metadata.json"


def parse_args_with_config(parser):
    parser.add_argument("--config", type=str, default=None, help="Optional path to a JSON config file.")
    config_args, _ = parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        parser.set_defaults(**config)
    args = parser.parse_args()
    # Config/CLI paths are repository-relative, independent of the caller's cwd.
    for name, value in vars(args).items():
        is_path = name.endswith(("_path", "_dir", "_root", "_csv"))
        if is_path and isinstance(value, str) and value and "://" not in value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                local_prefixes = ("./", "../", "data/", "runs/", "results/", "outputs/", "output/", "assets/", "third_party/", "model/", "configs/")
                if name == "tokenizer_path" and not value.startswith(local_prefixes) and not (PROJECT_ROOT / path).exists():
                    continue
                base = PROJECT_ROOT / "model" if value.startswith("../") else PROJECT_ROOT
                setattr(args, name, str((base / path).resolve()))
    return args


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_tokenizer_padding(tokenizer: PreTrainedTokenizerFast) -> PreTrainedTokenizerFast:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_tokenizer(tokenizer_path: str, *, model_max_length: Optional[int] = None) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if model_max_length is not None:
        tokenizer.model_max_length = model_max_length
    return ensure_tokenizer_padding(tokenizer)


def build_condition_token(condition_type: str) -> str:
    return f"<{condition_type}>"


def configure_condition_token(tokenizer: PreTrainedTokenizerFast, condition_type: str) -> int:
    condition_token = build_condition_token(condition_type)
    if condition_token not in tokenizer.get_vocab():
        raise ValueError(f"Condition token '{condition_token}' not found in tokenizer vocabulary.")
    condition_token_id = tokenizer.convert_tokens_to_ids(condition_token)
    tokenizer.bos_token = condition_token
    tokenizer.bos_token_id = condition_token_id
    return condition_token_id


def load_peft_model(
    base_model_path: str,
    adapter_path: str,
    tokenizer: PreTrainedTokenizerFast,
    *,
    device: Union[str, torch.device] = "cuda",
    trainable: bool = False,
    full_finetune: bool = False,
):
    base_model = GPT2LMHeadModel.from_pretrained(base_model_path)
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=trainable)
    model.to(device)

    if trainable:
        if full_finetune:
            for param in model.parameters():
                param.requires_grad = True
        model.train()
    else:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    return model


def load_merged_model(
    base_model_path: str,
    adapter_path: str,
    tokenizer: PreTrainedTokenizerFast,
    *,
    device: Union[str, torch.device] = "cuda",
    trainable: bool = False,
):
    base_model = GPT2LMHeadModel.from_pretrained(base_model_path)
    base_model.resize_token_embeddings(len(tokenizer))

    peft_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=trainable)
    merged_model = peft_model.merge_and_unload()
    merged_model.to(device)

    if trainable:
        for param in merged_model.parameters():
            param.requires_grad = True
        merged_model.train()
    else:
        for param in merged_model.parameters():
            param.requires_grad = False
        merged_model.eval()

    return merged_model


def load_checkpoint_metadata(checkpoint_path: Union[str, os.PathLike]) -> dict:
    metadata_path = Path(checkpoint_path) / CHECKPOINT_METADATA_FILENAME
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_checkpoint_mode(checkpoint_path: Union[str, os.PathLike], checkpoint_mode: str = "auto") -> str:
    if checkpoint_mode != "auto":
        return checkpoint_mode

    checkpoint_dir = Path(checkpoint_path)
    metadata = load_checkpoint_metadata(checkpoint_dir)
    if metadata.get("checkpoint_mode"):
        return metadata["checkpoint_mode"]
    if (checkpoint_dir / FULL_FINETUNE_STATE_FILENAME).exists():
        return "full_finetune"
    if (checkpoint_dir / "model.safetensors").exists():
        return "full_finetune"
    return "adapter"


def load_checkpoint_model(
    base_model_path: str,
    checkpoint_path: str,
    tokenizer: PreTrainedTokenizerFast,
    *,
    device: Union[str, torch.device] = "cuda",
    trainable: bool = False,
    checkpoint_mode: str = "auto",
    full_finetune: bool = False,
):
    resolved_mode = resolve_checkpoint_mode(checkpoint_path, checkpoint_mode=checkpoint_mode)
    if resolved_mode in {"adapter", "lora_finetune"}:
        return load_peft_model(
            base_model_path,
            checkpoint_path,
            tokenizer,
            device=device,
            trainable=trainable,
            full_finetune=full_finetune,
        )

    if resolved_mode != "full_finetune":
        raise ValueError(f"Unsupported checkpoint_mode: {resolved_mode}")

    checkpoint_dir = Path(checkpoint_path)
    metadata = load_checkpoint_metadata(checkpoint_dir)
    train_mode = metadata.get("train_mode")
    if train_mode == "full_finetune_merged":
        model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
        model.resize_token_embeddings(len(tokenizer))
    else:
        state_path = checkpoint_dir / FULL_FINETUNE_STATE_FILENAME
        if not state_path.exists():
            raise FileNotFoundError(f"Full-finetune state file not found: {state_path}")
        model = load_peft_model(
            base_model_path,
            checkpoint_path,
            tokenizer,
            device=device,
            trainable=trainable,
            full_finetune=full_finetune,
        )
        state_dict = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    model.to(device)

    if trainable:
        for param in model.parameters():
            param.requires_grad = True
        model.train()
    else:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    return model


def is_valid_smiles(smiles: str) -> bool:
    if not smiles or len(smiles.strip()) == 0:
        return False
    try:
        RDLogger.DisableLog("rdApp.error")
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False
    finally:
        RDLogger.EnableLog("rdApp.error")
