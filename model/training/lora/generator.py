from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

from model.shared.generation_utils import build_generation_records
from model.shared.pipeline_utils import (
    build_condition_token,
    configure_condition_token,
    load_peft_model,
)
from model.training.lora.dataset import PEPTIDE_TYPES

logger = logging.getLogger(__name__)


class LoRAPeptideGenerator:
    def __init__(
        self,
        *,
        base_model_path: str,
        tokenizer,
        adapter_base_dir: str,
        device: Optional[str] = None,
        use_cache: bool = False,
    ):
        self.base_model_path = base_model_path
        self.tokenizer = tokenizer
        self.adapter_base_dir = Path(adapter_base_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cache = use_cache
        self.current_model = None
        self.current_type: str | None = None
        self._adapter_cache: dict[str, Any] = {}

    def list_available_adapters(self) -> list[str]:
        if not self.adapter_base_dir.exists():
            return []
        return sorted(
            entry.name.replace("adapter_", "")
            for entry in self.adapter_base_dir.iterdir()
            if entry.is_dir() and entry.name.startswith("adapter_")
        )

    def load_adapter(self, peptide_type: str):
        if peptide_type not in PEPTIDE_TYPES:
            raise ValueError(f"Unsupported peptide type: {peptide_type}")
        if self.current_type == peptide_type and self.current_model is not None:
            return self.current_model
        if self.use_cache and peptide_type in self._adapter_cache:
            self.current_model = self._adapter_cache[peptide_type]
            self.current_type = peptide_type
            return self.current_model

        adapter_path = self.adapter_base_dir / f"adapter_{peptide_type}"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

        if not self.use_cache and self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_type = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model = load_peft_model(
            self.base_model_path,
            str(adapter_path),
            self.tokenizer,
            device=self.device,
            trainable=False,
        )
        if self.use_cache:
            self._adapter_cache[peptide_type] = model
        self.current_model = model
        self.current_type = peptide_type
        logger.info("Loaded adapter for %s from %s", peptide_type, adapter_path)
        return model

    @torch.no_grad()
    def generate(
        self,
        peptide_type: str,
        *,
        num_sequences: int,
        batch_size: int,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> list[dict[str, Any]]:
        model = self.load_adapter(peptide_type)
        records: list[dict[str, Any]] = []
        start_token_id = configure_condition_token(self.tokenizer, peptide_type)

        while len(records) < num_sequences:
            current_batch_size = min(batch_size, num_sequences - len(records))
            input_ids = torch.full(
                (current_batch_size, 1),
                start_token_id,
                dtype=torch.long,
                device=self.device,
            )
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            batch_records, _ = build_generation_records(
                decoded,
                condition_type=peptide_type,
                tokenizer=self.tokenizer,
                include_raw_output=False,
            )
            for batch_record in batch_records:
                records.append(
                    {
                        "type": peptide_type,
                        "condition_token": build_condition_token(peptide_type),
                        "smiles": batch_record["SMILES"],
                        "valid": batch_record["Valid"],
                    }
                )
        return records[:num_sequences]

    def batch_generate_all_types(self, **kwargs) -> dict[str, list[dict[str, Any]]]:
        results: dict[str, list[dict[str, Any]]] = {}
        for peptide_type in self.list_available_adapters():
            results[peptide_type] = self.generate(peptide_type, **kwargs)
        return results
