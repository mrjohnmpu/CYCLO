from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from rdkit import Chem, RDLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.property_pipeline.peptiverse_batch import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    DEFAULT_WEIGHT_ROOT,
    PeptiVerseBatchPredictor,
)
from model.shared.generation_utils import (  # noqa: E402
    clean_generated_sequence,
    prepare_condition_token,
)
from model.shared.pipeline_utils import (  # noqa: E402
    CHECKPOINT_METADATA_FILENAME,
    FULL_FINETUNE_STATE_FILENAME,
    is_valid_smiles,
    load_tokenizer,
    parse_args_with_config,
    seed_everything,
)
from trl.experimental.ppo import (  # noqa: E402
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer as _ExperimentalPPOTrainer,
)
from trl.experimental.ppo.ppo_trainer import (  # noqa: E402
    masked_mean,
    masked_whiten,
)
from trl.experimental.utils import first_true_indices  # noqa: E402
from trl.trainer.utils import selective_log_softmax  # noqa: E402

RDLogger.DisableLog("rdApp.*")
INVALID_LOGPROB = 1.0
TRACE_COLUMNS = [
    "step",
    "mean_reward_total",
    "std_reward_total",
    "valid_rate",
    "unique_rate_valid",
    "mean_perm_raw",
    "perm_pass_rate",
    "mean_hemo_prob",
    "mean_hemo_safe",
    "hemo_safe_rate",
    "mean_response_length",
    "kl",
    "policy_loss",
    "value_loss",
    "entropy",
]
SAMPLE_COLUMNS = [
    "SMILES",
    "Valid",
    "Cyclization",
    "perm_raw",
    "hemo_prob",
    "hemo_safe",
    "reward_perm",
    "reward_hemo",
    "reward_total",
    "step",
]


def resolve_repo_path(path_str: str | None, *, allow_model_hub: bool = False) -> str | None:
    if path_str is None:
        return None
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


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Unable to parse boolean value: {value}")


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


SUPPORTED_CONDITION_TYPES = {
    "head_to_tail",
    "sc_to_tail",
    "disulfide_bridge",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full-finetune PPO with configurable PeptiVerse property rewards.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default='runs/pretrain/final_model',
    )
    parser.add_argument(
        "--adapter_init_path",
        type=str,
        default='runs/lora/adapter_head_to_tail',
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default='runs/lora/tokenizer',
    )
    parser.add_argument("--condition_type", type=str, default="head_to_tail")
    parser.add_argument("--checkpoint_mode", type=str, default="full_finetune")
    parser.add_argument("--output_dir", type=str, default='runs')
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--total_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--mini_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--kl_coef", type=float, default=0.04)
    parser.add_argument("--target_kl", type=float, default=0.05)
    parser.add_argument("--vf_coef", type=float, default=0.1)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--cliprange_value", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--save_trainer_state", type=str_to_bool, default=False)
    parser.add_argument("--sample_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--probe_samples", type=int, default=100)
    parser.add_argument("--sample_num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--reward_use_perm", type=str_to_bool, default=True)
    parser.add_argument("--reward_use_hemo", type=str_to_bool, default=True)
    parser.add_argument("--reward_invalid_penalty", type=float, default=-0.6)
    parser.add_argument("--reward_perm_threshold", type=float, default=-6.0)
    parser.add_argument("--reward_perm_sigmoid_center", type=float, default=-6.0)
    parser.add_argument("--reward_hemo_decay_rate", type=float, default=3.0)
    parser.add_argument("--reward_hemo_floor", type=float, default=0.05)
    parser.add_argument("--reward_hemo_safe_threshold", type=float, default=0.15)
    parser.add_argument("--peptiverse_manifest_path", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--peptiverse_weight_root", type=str, default=str(DEFAULT_WEIGHT_ROOT))
    parser.add_argument("--peptiverse_device", type=str, default="cuda:0")
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_args_with_config(parser)
    args.base_model_path = resolve_repo_path(args.base_model_path)
    args.adapter_init_path = resolve_repo_path(args.adapter_init_path)
    args.tokenizer_path = resolve_repo_path(args.tokenizer_path, allow_model_hub=True)
    args.output_dir = resolve_repo_path(args.output_dir)
    args.peptiverse_manifest_path = resolve_repo_path(args.peptiverse_manifest_path)
    args.peptiverse_weight_root = resolve_repo_path(args.peptiverse_weight_root)
    if args.condition_type not in SUPPORTED_CONDITION_TYPES:
        raise ValueError(
            f"Unsupported condition_type={args.condition_type!r}. "
            f"Expected one of {sorted(SUPPORTED_CONDITION_TYPES)}."
        )
    if args.checkpoint_mode != "full_finetune":
        raise ValueError("--checkpoint_mode must be 'full_finetune' for train_ppo_v3_full_finetune.py")
    if args.max_length < 2:
        raise ValueError("--max_length must be >= 2")
    if args.total_steps < 1:
        raise ValueError("--total_steps must be >= 1")
    if args.batch_size < 1 or args.mini_batch_size < 1:
        raise ValueError("--batch_size and --mini_batch_size must be >= 1")
    if args.batch_size % args.mini_batch_size != 0:
        raise ValueError("--batch_size must be divisible by --mini_batch_size")
    if args.mini_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("--mini_batch_size must be divisible by --gradient_accumulation_steps")
    if args.probe_samples < 1 or args.sample_num_samples < 1:
        raise ValueError("--probe_samples and --sample_num_samples must be >= 1")
    if not args.reward_use_perm and not args.reward_use_hemo:
        raise ValueError("At least one reward component must be enabled.")
    return args


def sigmoid_permeability_reward(perm_val: float, center: float = -6.0) -> float:
    """Sigmoid mapping of permeability to [0, 1].  Higher perm = higher reward.

    Examples (center=-6):
        perm_val=-8.0 → 0.12   (very poor)
        perm_val=-6.0 → 0.50   (threshold)
        perm_val=-4.5 → 0.82   (near-optimal)
        perm_val=-3.0 → 0.95   (excellent)
    """
    return 1.0 / (1.0 + math.exp(-(perm_val - center)))


def exp_hemo_reward(
    hemo_prob: float,
    decay_rate: float = 3.0,
    floor: float = 0.05,
) -> float:
    """Exponential-decay mapping of hemo_prob to [floor, 1].  Lower hemo = higher reward.

    Examples (decay_rate=3, floor=0.05):
        hemo_prob=0.00 → 1.000  (perfectly safe)
        hemo_prob=0.10 → 0.741
        hemo_prob=0.20 → 0.549
        hemo_prob=0.46 → 0.252  (typical initial model)
        hemo_prob=1.00 → 0.050  (worst case, floored)
    """
    return max(floor, math.exp(-decay_rate * hemo_prob))


def calculate_total_reward(r_perm: float, r_hemo: float) -> float:
    """Geometric mean of individual [0, 1] sub-rewards.

    Implements "veto" semantics: if either sub-reward is near 0 the total
    reward collapses, preventing one-sided optimisation.
    """
    return math.sqrt(max(0.0, r_perm) * max(0.0, r_hemo))


@dataclass
class SequenceMetrics:
    smiles: list[str]
    valid: list[bool]
    perm_raw: list[float]
    hemo_prob: list[float]
    hemo_safe: list[float]
    reward_perm: list[float]
    reward_hemo: list[float]
    reward_total: list[float]


class CompositeRewardComputer:
    def __init__(self, args: argparse.Namespace, tokenizer) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.condition_token = prepare_condition_token(tokenizer, args.condition_type)[0]
        self.predictor: PeptiVerseBatchPredictor | None = None

    def _get_predictor(self) -> PeptiVerseBatchPredictor:
        if self.predictor is None:
            self.predictor = PeptiVerseBatchPredictor.load(
                manifest_path=Path(self.args.peptiverse_manifest_path),
                weight_root=Path(self.args.peptiverse_weight_root),
                device=self.args.peptiverse_device,
            )
        return self.predictor

    def _predict_perm(self, predictor: PeptiVerseBatchPredictor, smiles: str) -> float:
        try:
            record = predictor.predict_property("permeability_pampa", smiles)
            return float(record["score"])
        except Exception:
            return -10.0

    def _predict_hemo_prob(self, predictor: PeptiVerseBatchPredictor, smiles: str) -> float:
        try:
            record = predictor.predict_property("hemolysis", smiles)
            return float(record["score"])
        except Exception:
            return 1.0

    def score_sequences(self, query_responses: torch.Tensor) -> SequenceMetrics:
        decoded = self.tokenizer.batch_decode(query_responses, skip_special_tokens=False)
        smiles = [
            clean_generated_sequence(seq, condition_token=self.condition_token, tokenizer=self.tokenizer)
            for seq in decoded
        ]
        valid = [is_valid_smiles(item) for item in smiles]
        perm_raw = [float("nan")] * len(smiles)
        hemo_prob = [float("nan")] * len(smiles)
        hemo_safe = [float("nan")] * len(smiles)
        reward_perm = [0.0] * len(smiles)
        reward_hemo = [0.0] * len(smiles)
        reward_total = [self.args.reward_invalid_penalty] * len(smiles)

        predictor = self._get_predictor() if any(valid) and (self.args.reward_use_perm or self.args.reward_use_hemo) else None
        for idx, (smiles_item, is_valid_flag) in enumerate(zip(smiles, valid)):
            if not is_valid_flag:
                continue

            cur_perm = float("nan")
            cur_hemo = float("nan")
            r_perm = 1.0  # neutral default when property is disabled
            r_hemo = 1.0

            if predictor is not None and self.args.reward_use_perm:
                cur_perm = self._predict_perm(predictor, smiles_item)
                perm_raw[idx] = cur_perm
                r_perm = sigmoid_permeability_reward(
                    cur_perm, center=self.args.reward_perm_sigmoid_center,
                )
                reward_perm[idx] = r_perm
            if predictor is not None and self.args.reward_use_hemo:
                cur_hemo = self._predict_hemo_prob(predictor, smiles_item)
                hemo_prob[idx] = cur_hemo
                hemo_safe[idx] = 1.0 - cur_hemo
                r_hemo = exp_hemo_reward(
                    cur_hemo,
                    decay_rate=self.args.reward_hemo_decay_rate,
                    floor=self.args.reward_hemo_floor,
                )
                reward_hemo[idx] = r_hemo

            if not self.args.reward_use_perm:
                perm_raw[idx] = float("nan")
            if not self.args.reward_use_hemo:
                hemo_prob[idx] = float("nan")
                hemo_safe[idx] = float("nan")

            # --- Geometric mean: "veto" semantics ---
            # Both r_perm and r_hemo ∈ [0, 1].  If either is near 0,
            # the total reward collapses, preventing one-sided optimisation.
            reward_total[idx] = calculate_total_reward(r_perm, r_hemo)

        if predictor is not None:
            predictor.clear_runtime_cache()

        return SequenceMetrics(
            smiles=smiles,
            valid=valid,
            perm_raw=perm_raw,
            hemo_prob=hemo_prob,
            hemo_safe=hemo_safe,
            reward_perm=reward_perm,
            reward_hemo=reward_hemo,
            reward_total=reward_total,
        )


def build_reward_tag(args: argparse.Namespace) -> str:
    parts = []
    if args.reward_use_perm:
        parts.append("perm")
    if args.reward_use_hemo:
        parts.append("hemo")
    return "_".join(parts)


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ppo_{args.condition_type}_{build_reward_tag(args)}_{timestamp}"


def build_ppo_config(args: argparse.Namespace, output_dir: str) -> PPOConfig:
    micro_batch_size = args.mini_batch_size // args.gradient_accumulation_steps
    return PPOConfig(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_mini_batches=args.batch_size // args.mini_batch_size,
        total_episodes=args.total_steps * args.batch_size,
        response_length=args.max_length - 1,
        temperature=args.temperature,
        kl_coef=args.kl_coef,
        cliprange=args.cliprange,
        cliprange_value=args.cliprange_value,
        vf_coef=args.vf_coef,
        gamma=args.gamma,
        lam=args.lam,
        num_ppo_epochs=args.ppo_epochs,
        logging_strategy="steps",
        logging_steps=args.log_interval,
        save_strategy="steps",
        save_steps=args.save_interval,
        eval_strategy="no",
        disable_tqdm=True,
        report_to=[],
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        num_sample_generations=0,
    )


def load_policy_model(base_model_path: str, adapter_path: str, tokenizer, device: torch.device):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    base_model.resize_token_embeddings(len(tokenizer))
    peft_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    merged_model = peft_model.merge_and_unload()
    for parameter in merged_model.parameters():
        parameter.requires_grad = True
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_model)
    policy_model.to(device)
    policy_model.train()
    return policy_model


def load_reference_model(base_model_path: str, adapter_path: str, tokenizer, device: torch.device):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    base_model.resize_token_embeddings(len(tokenizer))
    ref_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False).merge_and_unload()
    ref_model.to(device)
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad = False
    return ref_model


def prepare_forward_inputs(query_responses: torch.Tensor, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return input_ids, attention_mask, position_ids


def forward_policy(model, query_responses: torch.Tensor, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask, position_ids = prepare_forward_inputs(query_responses, pad_token_id)
    logits, _, values = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_past_key_values=False,
    )
    return logits, values


def forward_reference(model, query_responses: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    input_ids, attention_mask, position_ids = prepare_forward_inputs(query_responses, pad_token_id)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        use_cache=False,
    )
    return outputs.logits


def build_generation_config(args: argparse.Namespace, tokenizer, *, context_length: int) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=max(1, args.max_length - context_length),
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def generate_query_responses(
    model,
    *,
    start_token_id: int,
    batch_size: int,
    generation_config: GenerationConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            query_responses = model.generate(input_ids=queries, generation_config=generation_config)
    finally:
        if was_training:
            model.train()
    return queries, query_responses


def compute_rollout_batch(
    policy_model,
    ref_model,
    query_tensors: torch.Tensor,
    query_responses: torch.Tensor,
    *,
    pad_token_id: int,
    temperature: float,
) -> dict[str, Any]:
    context_length = query_tensors.shape[1]
    responses = query_responses[:, context_length:]
    was_training = policy_model.training
    policy_model.eval()
    ref_model.eval()
    try:
        with torch.no_grad():
            policy_logits_full, values_full = forward_policy(policy_model, query_responses, pad_token_id)
            ref_logits_full = forward_reference(ref_model, query_responses, pad_token_id)
    finally:
        if was_training:
            policy_model.train()

    policy_logits = policy_logits_full[:, context_length - 1 : -1] / (temperature + 1e-7)
    ref_logits = ref_logits_full[:, context_length - 1 : -1] / (temperature + 1e-7)
    values = values_full[:, context_length - 1 : -1]
    logprobs = selective_log_softmax(policy_logits, responses)
    ref_logprobs = selective_log_softmax(ref_logits, responses)

    sequence_lengths = first_true_indices(responses == pad_token_id) - 1
    response_idxs = torch.arange(responses.shape[1], device=responses.device).unsqueeze(0).expand_as(responses)
    padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
    logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
    ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
    sequence_lengths_p1 = sequence_lengths + 1
    padding_mask_p1 = response_idxs > sequence_lengths_p1.unsqueeze(1)
    values = torch.masked_fill(values, padding_mask_p1, 0)

    return {
        "context_length": context_length,
        "pad_token_id": pad_token_id,
        "query_tensors": query_tensors,
        "query_responses": query_responses,
        "responses": responses,
        "logprobs": logprobs,
        "ref_logprobs": ref_logprobs,
        "values": values,
        "sequence_lengths": sequence_lengths,
        "sequence_lengths_p1": sequence_lengths_p1,
        "padding_mask": padding_mask,
        "padding_mask_p1": padding_mask_p1,
    }


def compute_rewards_and_advantages(
    rollout: dict[str, Any],
    sequence_metrics: SequenceMetrics,
    args: argparse.Namespace,
    *,
    effective_kl_coef: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kl_coef = effective_kl_coef if effective_kl_coef is not None else args.kl_coef
    scores = torch.tensor(sequence_metrics.reward_total, dtype=torch.float32, device=rollout["responses"].device)
    logr = rollout["ref_logprobs"] - rollout["logprobs"]
    kl = -logr
    non_score_reward = -kl_coef * kl
    rewards = non_score_reward.clone()

    actual_start = torch.arange(rewards.size(0), device=rewards.device)
    # actual_end = torch.where(
    #     rollout["sequence_lengths_p1"] < rewards.size(1),
    #     rollout["sequence_lengths_p1"],
    #     rollout["sequence_lengths"],
    # )
    actual_end = rollout["sequence_lengths"]
    rewards[actual_start, actual_end] += scores

    lastgaelam = 0
    advantages_reversed = []
    gen_length = rollout["responses"].shape[1]
    for time_index in reversed(range(gen_length)):
        next_values = rollout["values"][:, time_index + 1] if time_index < gen_length - 1 else 0.0
        delta = rewards[:, time_index] + args.gamma * next_values - rollout["values"][:, time_index]
        lastgaelam = delta + args.gamma * args.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + rollout["values"]
    advantages = masked_whiten(advantages, ~rollout["padding_mask"])
    advantages = torch.masked_fill(advantages, rollout["padding_mask"], 0)
    return rewards, scores, advantages, returns


def ppo_update(
    policy_model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rollout: dict[str, Any],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    batch_size = rollout["responses"].shape[0]
    mini_batch_size = args.mini_batch_size
    micro_batch_size = mini_batch_size // args.gradient_accumulation_steps
    context_length = rollout["context_length"]
    pad_token_id = rollout["pad_token_id"]

    approxkl_values: list[float] = []
    policy_loss_values: list[float] = []
    value_loss_values: list[float] = []
    entropy_values: list[float] = []
    stop_early = False

    policy_model.train()
    for _ppo_epoch in range(args.ppo_epochs):
        permutation = np.random.permutation(batch_size)
        for mini_start in range(0, batch_size, mini_batch_size):
            mini_end = mini_start + mini_batch_size
            mini_indices = permutation[mini_start:mini_end]
            optimizer.zero_grad(set_to_none=True)
            mini_batch_approxkls = []

            for micro_start in range(0, mini_batch_size, micro_batch_size):
                micro_end = micro_start + micro_batch_size
                micro_indices = mini_indices[micro_start:micro_end]

                mb_advantage = advantages[micro_indices]
                mb_responses = rollout["responses"][micro_indices]
                mb_query_responses = rollout["query_responses"][micro_indices]
                mb_logprobs = rollout["logprobs"][micro_indices]
                mb_returns = returns[micro_indices]
                mb_values = rollout["values"][micro_indices]
                padding_mask = rollout["padding_mask"][micro_indices]
                padding_mask_p1 = rollout["padding_mask_p1"][micro_indices]

                logits_full, vpred_full = forward_policy(policy_model, mb_query_responses, pad_token_id)
                logits = logits_full[:, context_length - 1 : -1] / (args.temperature + 1e-7)
                new_logprobs = selective_log_softmax(logits, mb_responses)
                new_logprobs = torch.masked_fill(new_logprobs, padding_mask, INVALID_LOGPROB)

                vpred = vpred_full[:, context_length - 1 : -1]
                vpred = torch.masked_fill(vpred, padding_mask_p1, 0)
                vpredclipped = torch.clamp(
                    vpred,
                    mb_values - args.cliprange_value,
                    mb_values + args.cliprange_value,
                )
                vf_losses1 = torch.square(vpred - mb_returns)
                vf_losses2 = torch.square(vpredclipped - mb_returns)
                vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), ~padding_mask_p1)

                logprobs_diff = new_logprobs - mb_logprobs
                ratio = torch.exp(logprobs_diff)
                pg_losses = -mb_advantage * ratio
                pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), ~padding_mask)

                # Entropy bonus: prevent mode collapse by encouraging exploration
                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                mean_entropy = masked_mean(entropy, ~padding_mask)
                loss = pg_loss + args.vf_coef * vf_loss - args.entropy_coef * mean_entropy
                (loss / args.gradient_accumulation_steps).backward()

                with torch.no_grad():
                    approxkl = 0.5 * masked_mean(logprobs_diff**2, ~padding_mask)
                    approxkl_value = float(approxkl.item())
                    mini_batch_approxkls.append(approxkl_value)
                    approxkl_values.append(approxkl_value)
                    policy_loss_values.append(float(pg_loss.item()))
                    value_loss_values.append(float(vf_loss.item()))
                    entropy_values.append(float(mean_entropy.item()))

            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if mini_batch_approxkls and float(np.mean(mini_batch_approxkls)) > args.target_kl:
                stop_early = True
                break

        if stop_early:
            break

    scheduler.step()
    return {
        "policy_loss": float(np.mean(policy_loss_values)) if policy_loss_values else 0.0,
        "value_loss": float(np.mean(value_loss_values)) if value_loss_values else 0.0,
        "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "approx_kl": float(np.mean(approxkl_values)) if approxkl_values else 0.0,
    }


def summarize_sequence_metrics(
    sequence_metrics: SequenceMetrics,
    *,
    sequence_lengths: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    valid_smiles = [smiles for smiles, is_valid_flag in zip(sequence_metrics.smiles, sequence_metrics.valid) if is_valid_flag]
    canonical_valid = [canonicalize_smiles(smiles) for smiles in valid_smiles]
    canonical_valid = [smiles for smiles in canonical_valid if smiles is not None]

    perm_values = [value for value, is_valid_flag in zip(sequence_metrics.perm_raw, sequence_metrics.valid) if is_valid_flag and math.isfinite(value)]
    hemo_prob_values = [value for value, is_valid_flag in zip(sequence_metrics.hemo_prob, sequence_metrics.valid) if is_valid_flag and math.isfinite(value)]
    hemo_safe_values = [value for value, is_valid_flag in zip(sequence_metrics.hemo_safe, sequence_metrics.valid) if is_valid_flag and math.isfinite(value)]

    return {
        "valid_rate": float(np.mean(sequence_metrics.valid)) if sequence_metrics.valid else 0.0,
        "unique_rate_valid": (len(set(canonical_valid)) / len(canonical_valid)) if canonical_valid else 0.0,
        "mean_perm_raw": float(np.mean(perm_values)) if perm_values else float("nan"),
        "perm_pass_rate": float(np.mean([value >= args.reward_perm_threshold for value in perm_values])) if perm_values else float("nan"),
        "mean_hemo_prob": float(np.mean(hemo_prob_values)) if hemo_prob_values else float("nan"),
        "mean_hemo_safe": float(np.mean(hemo_safe_values)) if hemo_safe_values else float("nan"),
        "hemo_safe_rate": (
            float(np.mean([value >= (1.0 - args.reward_hemo_safe_threshold) for value in hemo_safe_values]))
            if hemo_safe_values
            else float("nan")
        ),
        "mean_response_length": float(sequence_lengths.float().mean().item() + 1.0) if sequence_lengths.numel() else 0.0,
    }


def build_sample_dataframe(sequence_metrics: SequenceMetrics, *, condition_type: str, step: int) -> pd.DataFrame:
    rows = []
    for idx, smiles in enumerate(sequence_metrics.smiles):
        rows.append(
            {
                "SMILES": smiles,
                "Valid": sequence_metrics.valid[idx],
                "Cyclization": condition_type,
                "perm_raw": sequence_metrics.perm_raw[idx],
                "hemo_prob": sequence_metrics.hemo_prob[idx],
                "hemo_safe": sequence_metrics.hemo_safe[idx],
                "reward_perm": sequence_metrics.reward_perm[idx],
                "reward_hemo": sequence_metrics.reward_hemo[idx],
                "reward_total": sequence_metrics.reward_total[idx],
                "step": step,
            }
        )
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def append_trace_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in TRACE_COLUMNS})


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def save_policy_checkpoint(
    policy_model,
    tokenizer,
    save_dir: Path,
    args: argparse.Namespace,
    *,
    step: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    policy_model.pretrained_model.save_pretrained(str(save_dir))
    torch.save(policy_model.pretrained_model.state_dict(), save_dir / FULL_FINETUNE_STATE_FILENAME)
    tokenizer.save_pretrained(str(save_dir))
    metadata = {
        "checkpoint_mode": "full_finetune",
        "rl_algorithm": "ppo",
        "condition_type": args.condition_type,
        "base_model_path": args.base_model_path,
        "adapter_init_path": args.adapter_init_path,
        "step": int(step),
        "reward_mode": build_reward_tag(args),
        "max_length": int(args.max_length),
        "reward_use_perm": bool(args.reward_use_perm),
        "reward_use_hemo": bool(args.reward_use_hemo),
        "train_mode": "full_finetune",
    }
    save_json(save_dir / CHECKPOINT_METADATA_FILENAME, metadata)


def save_trainer_state(
    policy_model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    save_dir: Path,
    *,
    step: int,
    ppo_config: PPOConfig,
    trace_row: dict[str, Any],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy_model.v_head.state_dict(), save_dir / "value_head.pt")
    torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), save_dir / "scheduler.pt")
    save_json(
        save_dir / "trainer_state.json",
        {
            "step": int(step),
            "ppo_config": ppo_config.to_dict(),
            "trace_row": trace_row,
        },
    )


def sample_policy(
    policy_model,
    reward_computer: CompositeRewardComputer,
    *,
    tokenizer,
    start_token_id: int,
    generation_config: GenerationConfig,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    condition_type: str,
    step: int,
) -> pd.DataFrame:
    dataframes = []
    generated = 0
    while generated < num_samples:
        current_batch_size = min(batch_size, num_samples - generated)
        _, query_responses = generate_query_responses(
            policy_model,
            start_token_id=start_token_id,
            batch_size=current_batch_size,
            generation_config=generation_config,
            device=device,
        )
        sequence_metrics = reward_computer.score_sequences(query_responses)
        dataframes.append(build_sample_dataframe(sequence_metrics, condition_type=condition_type, step=step))
        generated += current_batch_size
    return pd.concat(dataframes, ignore_index=True)


def build_trace_row(
    *,
    step: int,
    sequence_metrics: SequenceMetrics,
    sequence_lengths: torch.Tensor,
    scores: torch.Tensor,
    approx_kl: float,
    policy_loss: float,
    value_loss: float,
    entropy: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = summarize_sequence_metrics(sequence_metrics, sequence_lengths=sequence_lengths, args=args)
    return {
        "step": int(step),
        "mean_reward_total": float(scores.mean().item()),
        "std_reward_total": float(scores.std(unbiased=False).item()),
        "valid_rate": summary["valid_rate"],
        "unique_rate_valid": summary["unique_rate_valid"],
        "mean_perm_raw": summary["mean_perm_raw"],
        "perm_pass_rate": summary["perm_pass_rate"],
        "mean_hemo_prob": summary["mean_hemo_prob"],
        "mean_hemo_safe": summary["mean_hemo_safe"],
        "hemo_safe_rate": summary["hemo_safe_rate"],
        "mean_response_length": summary["mean_response_length"],
        "kl": float(approx_kl),
        "policy_loss": float(policy_loss),
        "value_loss": float(value_loss),
        "entropy": float(entropy),
    }


def maybe_empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer_path, model_max_length=args.max_length)
    _, start_token_id = prepare_condition_token(tokenizer, args.condition_type)
    run_dir = Path(args.output_dir) / build_run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    ppo_config = build_ppo_config(args, str(run_dir))
    save_json(
        run_dir / "config.json",
        {
            **vars(args),
            "run_dir": str(run_dir),
            "reward_tag": build_reward_tag(args),
            "device_resolved": str(device),
            "trl_api": _ExperimentalPPOTrainer.__name__,
            "ppo_config": ppo_config.to_dict(),
        },
    )

    policy_model = load_policy_model(args.base_model_path, args.adapter_init_path, tokenizer, device)
    ref_model = load_reference_model(args.base_model_path, args.adapter_init_path, tokenizer, device)
    optimizer = AdamW(
        (parameter for parameter in policy_model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    reward_computer = CompositeRewardComputer(args, tokenizer)
    generation_config = build_generation_config(args, tokenizer, context_length=1)
    reward_trace_path = run_dir / "reward_trace.csv"

    probe_df = sample_policy(
        policy_model,
        reward_computer,
        tokenizer=tokenizer,
        start_token_id=start_token_id,
        generation_config=generation_config,
        device=device,
        num_samples=args.probe_samples,
        batch_size=min(args.batch_size, args.probe_samples),
        condition_type=args.condition_type,
        step=0,
    )
    probe_df.to_csv(run_dir / "probe_samples_step0000.csv", index=False)

    latest_trace_row: dict[str, Any] | None = None
    consecutive_zero_valid = 0  # collapse watchdog counter
    progress = tqdm(range(1, args.total_steps + 1), desc="PPO")
    for step in progress:
        query_tensors, query_responses = generate_query_responses(
            policy_model,
            start_token_id=start_token_id,
            batch_size=args.batch_size,
            generation_config=generation_config,
            device=device,
        )
        rollout = compute_rollout_batch(
            policy_model,
            ref_model,
            query_tensors,
            query_responses,
            pad_token_id=tokenizer.pad_token_id,
            temperature=args.temperature,
        )
        sequence_metrics = reward_computer.score_sequences(query_responses)

        # --- Valid-rate watchdog: dynamically increase KL penalty ---
        current_valid_rate = float(np.mean(sequence_metrics.valid))
        if current_valid_rate < 0.5:
            effective_kl_coef = args.kl_coef * 2.5
        elif current_valid_rate < 0.8:
            effective_kl_coef = args.kl_coef * 1.3
        else:
            effective_kl_coef = args.kl_coef

        _, scores, advantages, returns = compute_rewards_and_advantages(
            rollout, sequence_metrics, args, effective_kl_coef=effective_kl_coef,
        )
        update_stats = ppo_update(policy_model, optimizer, scheduler, rollout, advantages, returns, args)
        latest_trace_row = build_trace_row(
            step=step,
            sequence_metrics=sequence_metrics,
            sequence_lengths=rollout["sequence_lengths"],
            scores=scores,
            approx_kl=update_stats["approx_kl"],
            policy_loss=update_stats["policy_loss"],
            value_loss=update_stats["value_loss"],
            entropy=update_stats["entropy"],
            args=args,
        )
        progress.set_postfix(
            reward=f"{latest_trace_row['mean_reward_total']:.3f}",
            valid=f"{latest_trace_row['valid_rate']:.2f}",
            kl=f"{latest_trace_row['kl']:.3f}",
        )

        if step == 1 or step % args.log_interval == 0 or step == args.total_steps:
            append_trace_row(reward_trace_path, latest_trace_row)

        if step % args.sample_interval == 0 or step == args.total_steps:
            samples_df = sample_policy(
                policy_model,
                reward_computer,
                tokenizer=tokenizer,
                start_token_id=start_token_id,
                generation_config=generation_config,
                device=device,
                num_samples=args.sample_num_samples,
                batch_size=min(args.batch_size, args.sample_num_samples),
                condition_type=args.condition_type,
                step=step,
            )
            samples_df.to_csv(run_dir / f"samples_step{step}.csv", index=False)

        if step % args.save_interval == 0 or step == args.total_steps:
            save_policy_checkpoint(
                policy_model,
                tokenizer,
                run_dir / f"agent_step{step}",
                args,
                step=step,
            )
            if args.save_trainer_state:
                save_trainer_state(
                    policy_model,
                    optimizer,
                    scheduler,
                    run_dir / "trainer_state" / f"step{step}",
                    step=step,
                    ppo_config=ppo_config,
                    trace_row=latest_trace_row or {"step": step},
                )

        maybe_empty_cuda_cache(device)

        # --- Collapse early-stopping ---
        if current_valid_rate == 0.0:
            consecutive_zero_valid += 1
            if consecutive_zero_valid >= 10:
                tqdm.write(
                    f"ALERT: {consecutive_zero_valid} consecutive steps with 0% valid rate "
                    f"at step {step}. Stopping training to prevent further collapse."
                )
                break
        else:
            consecutive_zero_valid = 0

    progress.close()


if __name__ == "__main__":
    main()
