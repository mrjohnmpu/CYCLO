from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Sequence

from peft import LoraConfig, TaskType

from model.training.lora.dataset import PEPTIDE_TYPES

logger = logging.getLogger(__name__)

DEFAULT_TARGET_MODULES = ("c_attn", "c_proj", "c_fc")
DEFAULT_MODULES_TO_SAVE = ("wte",)

_TYPE_LORA_CONFIGS = {
    "linear": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ("c_attn", "c_proj"),
    },
    "head_to_tail": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": DEFAULT_TARGET_MODULES,
    },
    "sc_to_tail": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": DEFAULT_TARGET_MODULES,
    },
    "disulfide_bridge": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": DEFAULT_TARGET_MODULES,
    },
}


@dataclass(frozen=True)
class LoRAHyperparameters:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    modules_to_save: tuple[str, ...] = DEFAULT_MODULES_TO_SAVE

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        data["modules_to_save"] = list(self.modules_to_save)
        return data


class LoRAConfigFactory:
    @staticmethod
    def get_hyperparameters(
        peptide_type: str,
        *,
        r: int | None = None,
        lora_alpha: int | None = None,
        lora_dropout: float | None = None,
        target_modules: Sequence[str] | None = None,
        modules_to_save: Sequence[str] | None = None,
    ) -> LoRAHyperparameters:
        if peptide_type not in PEPTIDE_TYPES:
            raise ValueError(
                f"Unsupported peptide type '{peptide_type}'. Expected one of {PEPTIDE_TYPES}."
            )

        defaults = _TYPE_LORA_CONFIGS[peptide_type]
        resolved = LoRAHyperparameters(
            r=r if r is not None else defaults["r"],
            lora_alpha=lora_alpha if lora_alpha is not None else defaults["lora_alpha"],
            lora_dropout=(
                lora_dropout if lora_dropout is not None else defaults["lora_dropout"]
            ),
            target_modules=tuple(target_modules or defaults["target_modules"]),
            modules_to_save=tuple(modules_to_save or DEFAULT_MODULES_TO_SAVE),
        )
        logger.info("LoRA hyperparameters for %s: %s", peptide_type, resolved.to_dict())
        return resolved

    @staticmethod
    def build_config(
        peptide_type: str,
        *,
        r: int | None = None,
        lora_alpha: int | None = None,
        lora_dropout: float | None = None,
        target_modules: Sequence[str] | None = None,
        modules_to_save: Sequence[str] | None = None,
    ) -> tuple[LoraConfig, LoRAHyperparameters]:
        hyperparameters = LoRAConfigFactory.get_hyperparameters(
            peptide_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            r=hyperparameters.r,
            lora_alpha=hyperparameters.lora_alpha,
            lora_dropout=hyperparameters.lora_dropout,
            target_modules=list(hyperparameters.target_modules),
            modules_to_save=list(hyperparameters.modules_to_save),
        )
        return config, hyperparameters
