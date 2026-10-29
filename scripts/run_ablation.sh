#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VARIANT="${1:-help}"
ACTION="${2:-train}"
if [[ "$#" -ge 2 ]]; then shift 2; fi
case "$VARIANT" in
  token_only) SUFFIX=token_only_frozen_prior; CONFIG=token_only_frozen_prior_seed42.json ;;
  lora_only) SUFFIX=lora_only_no_token; CONFIG=lora_only_no_token_seed42.json ;;
  strict_token_lora) SUFFIX=strict_token_lora; CONFIG=strict_token_lora_seed42.json ;;
  shared_token_lora) SUFFIX=shared_strict_token_lora; CONFIG=shared_strict_token_lora_r64_seed42.json ;;
  full_finetune) SUFFIX=full_finetune_strict_token; CONFIG=full_finetune_strict_token_seed42.json ;;
  *) printf 'Usage: bash scripts/run_ablation.sh {token_only|lora_only|strict_token_lora|shared_token_lora|full_finetune} {train|generate} [--print-command] [script arguments]\n'; exit 0 ;;
esac
case "$ACTION" in train|generate) ;; *) echo 'Action must be train or generate' >&2; exit 2 ;; esac
PRINT_ONLY=0
if [[ "${1:-}" == --print-command ]]; then PRINT_ONLY=1; shift; fi
cd "$ROOT"
COMMAND=("$PYTHON_BIN" "$ROOT/model/training/ablation/${ACTION}_${SUFFIX}.py" --config "$ROOT/configs/ablation/$CONFIG" "$@")
if [[ "$PRINT_ONLY" == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
export PYTHONUNBUFFERED=1
exec "${COMMAND[@]}"
