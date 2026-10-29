# TopoCycGPT

Curated research-code release for topology-conditioned cyclic peptide generation.
This directory contains a copy of the original paper workflow, not a replacement
for the development project. No training or GitHub upload is performed by setup.

## Contents

```text
model/          pretraining, independent topology LoRA, PPO, generation
baselines/      adapted AAE, VAE and CharRNN implementations
evaluation/     generation metrics, annotation, topology/residue analysis, figures
configs/        portable settings and paper-run PPO/generation configurations
scripts/        launchers, baseline commands, dataset and release validation
utils/          plotting helpers and PepINVENT fragment postprocessing
tests/          regression tests and small fixtures
data/           four complete datasets and integrity manifest
docs/           external dependencies, validation records and source provenance
```

Shared token+LoRA training, generation and evaluation are retained as an ablation.
PPO initialized from shared adapters, predictor-reliability audits, Free-State 3D
audits, unrelated targets, old tuning configurations, logs, caches, model weights
and generated datasets are excluded. Structure prediction, MD and MM-PBSA
execution pipelines are outside this release. `model/shared/` contains reusable
utilities; shared token+LoRA entrypoints are under `model/training/ablation/`.

## Installation

Use Python 3.12 in an isolated environment on your compute server. Install a
PyTorch build appropriate for the GPU/CUDA runtime, then install the remaining
requirements. The requirements reflect the validation environment, not an
invented historical environment lock.

```bash
conda create -n topocycgpt python=3.12
conda activate topocycgpt
python -m pip install -r requirements.txt
python scripts/prepare_data.py --unpack
```

Training and generation require external checkpoints. Baseline training/metrics
require MOSES; property annotation/PPO require PeptiVerse and its model assets;
residue analysis can require SMILES2PEPTIDE or an authorized private parser.
Install these in their appropriate environments as described in
[THIRD_PARTY.md](docs/THIRD_PARTY.md). They are not automatically downloaded.

## Running The Paper Workflow

The launchers locate this repository from their own location and use the active
environment's `python`, or `PYTHON_BIN` if set. Config/CLI paths in model entrypoints
are repository-relative unless absolute. CLI flags override JSON defaults. Use
absolute external paths when the data, models or outputs reside on another disk.

```bash
bash scripts/run_stage.sh pretrain --help
bash scripts/run_stage.sh pretrain --print-command
# Original pretraining used two processes; this command starts real training.
torchrun --standalone --nproc_per_node=2 \
  model/training/pretrain/train_cyc_gpt_with_mask_trainer.py \
  --config configs/pretrain_gpt2.json

# Three independent adapters, not one shared adapter.
bash scripts/run_stage.sh finetune
bash scripts/run_stage.sh generate
bash scripts/run_stage.sh generate --condition_type sc_to_tail \
  --adapter_path runs/lora/adapter_sc_to_tail --output_filename sc_to_tail_n30000.csv
bash scripts/run_stage.sh generate --condition_type disulfide_bridge \
  --adapter_path runs/lora/adapter_disulfide_bridge --output_filename disulfide_bridge_n30000.csv

# Actual paper-run PPO parameters, with a new output/run name when desired.
bash scripts/run_stage.sh reinforce --config configs/paper/ppo_head_to_tail.json
bash scripts/run_stage.sh reinforce --config configs/paper/ppo_sc_to_tail.json
bash scripts/run_stage.sh reinforce --config configs/paper/ppo_disulfide_bridge.json
bash scripts/run_stage.sh reinforce_kras
bash scripts/run_stage.sh reinforce_generate --config configs/paper/generate_ppo_head_to_tail.json
```

These examples are opt-in commands: running them starts compute-intensive work.
Specify `CUDA_VISIBLE_DEVICES` yourself; scripts never select an occupied physical
GPU on your behalf. Use a new `--run_name`/`--output_dir` for new runs and manage
long jobs with your scheduler, `tmux`, or `nohup` and explicit logs.

## Validation And Publication Notes

```bash
python scripts/prepare_data.py --report results/data_verification.json
python scripts/validate_release.py
# Optional test runner for function-style tests:
python -m pip install -r requirements-dev.txt
python -m pytest tests
```

Release validation is CPU-only: syntax/config checks, complete dataset integrity,
small regression fixtures, reward/split checks, synthetic plotting smoke tests,
and `--help` launched outside the repository. It does not validate full training
convergence, all external checkpoints, or predictive-model scientific accuracy.
External parser tests require access to those dependencies. The validation report
states skipped or unavailable checks explicitly.

One historical length-estimation branch depends on a missing `type_oracle.py`
source file. It is retained for traceability with an explicit error; use the
documented SMILES2PEPTIDE/private-parser workflows instead. No bytecode cache is
distributed as source and no replacement topology algorithm is silently inserted.

`docs/source_manifest.json` records original file hashes, publication hashes and
edits. `docs/provenance/` preserves historical run/source records; server paths
there are archival identifiers, not executable defaults. Input manifests for
generated/annotated datasets are templates: provide those external CSVs before
running corresponding plots. Dataset origins/redistribution permissions and the
license for original project code require author confirmation before publishing;
no blanket license is assigned by this preparation step. Upstream-derived code
keeps its own attribution/license obligations.

The complete training CSV is shipped as gzip. Its uncompressed file is ignored
by Git. The archive is larger than GitHub's browser-upload limit, so use normal
Git transfer rather than browser upload. See the
[GitHub file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

