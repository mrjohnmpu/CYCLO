#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:-help}"
if [[ "$#" -gt 0 ]]; then shift; fi
case "$STAGE" in
  pretrain) ENTRY=model/training/pretrain/train_cyc_gpt_with_mask_trainer.py; CONFIG=configs/pretrain_gpt2.json ;;
  finetune) ENTRY=model/training/lora/train.py; CONFIG=configs/finetune_lora.json ;;
  generate) ENTRY=model/training/lora/generate.py; CONFIG=configs/generate_lora.json ;;
  generate_base) ENTRY=model/inference/generate_base_gpt2.py; CONFIG=configs/generate_base_gpt2_final_model_maxlen1024.json ;;
  reinforce) ENTRY=model/reinforce/train_ppo_v3_full_finetune.py; CONFIG=configs/paper/ppo_head_to_tail.json ;;
  reinforce_kras) ENTRY=model/reinforce/train_ppo_kras_v2_full_finetune.py; CONFIG=configs/paper/ppo_kras.json ;;
  reinforce_generate) ENTRY=model/inference/Cyc_GPT_reinforce_generater_full_finetune.py; CONFIG=configs/paper/generate_ppo_head_to_tail.json ;;
  *) printf 'Usage: bash scripts/run_stage.sh {pretrain|finetune|generate|generate_base|reinforce|reinforce_kras|reinforce_generate} [--print-command] [script arguments]\n'; exit 0 ;;
esac
PRINT_ONLY=0
if [[ "${1:-}" == --print-command ]]; then PRINT_ONLY=1; shift; fi
cd "$ROOT"
COMMAND=("$PYTHON_BIN" "$ROOT/$ENTRY" --config "$ROOT/$CONFIG" "$@")
if [[ "$PRINT_ONLY" == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
export PYTHONUNBUFFERED=1
exec "${COMMAND[@]}"
