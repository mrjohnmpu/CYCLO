# Third-Party Dependencies And Attribution

External repositories and weights are not vendored or automatically fetched.
`third_party/` is ignored. Explicit environment variables override checkout
defaults; CLI arguments separately control weight/manifest/model locations.

| Component | Source | Integration |
| --- | --- | --- |
| MOSES | https://github.com/molecularsets/moses | `MOSES_ROOT` accepts its checkout root or Python package directory; alternatively install `molsets` |
| PepINVENT | https://github.com/MolecularAI/PepINVENT | Separate upstream environment; `input_to_sampling.py` and fragment reassembly in this project |
| PeptiVerse | https://huggingface.co/spaces/ChatterjeeLab/PeptiVerse | `PEPTIVERSE_ROOT`, plus predictor manifest/weights; default `third_party/PeptiVerse` |
| MolGPT tokenizer | https://huggingface.co/jonghyunlee/MolGPT_pretrained-by-ZINC15 | Original pretraining tokenization; local override via `--tokenizer_path` |
| SMILES2PEPTIDE | https://huggingface.co/spaces/ChatterjeeLab/SMILES2PEPTIDE | `SMILES2PEPTIDE_ROOT`; default `third_party/SMILES2PEPTIDE` |

Examples below configure external dependencies:

```bash
export MOSES_ROOT=/absolute/path/to/moses
export PEPTIVERSE_ROOT=/absolute/path/to/PeptiVerse
export SMILES2PEPTIDE_ROOT=/absolute/path/to/SMILES2PEPTIDE
```

Baseline training needs the full upstream MOSES checkout (including
`moses/interfaces.py`), not a metrics-only copy. If `MOSES_ROOT` is set, it takes
precedence over installed packages; use the same complete checkout for training
and metric validation.

MOSES AAE/VAE/CharRNN implementations under `baselines/moses_baselines/` are
adapted from MOSES, not original model architectures invented for this project.
Local modifications include peptide-type data filtering, output/provenance
handling and CPU packed-sequence lengths. Retain upstream copyright/license
notices and cite the MOSES paper. The inspected server checkout has no resolvable
Git HEAD; its exact upstream commit cannot honestly be supplied. Local source
hashes in the publication manifest identify the copied adaptation.

MOSES metrics also require its upstream metric resources and optional packages
such as FCD; installing the baseline wrappers alone is insufficient. Follow the
upstream installation instructions. Optional chemical-space acceleration may use
FPSim2; the retained implementation also provides its documented fallback.

Shared token+LoRA similarity comparison requires parquet support (for example,
`pyarrow`) and the configured FPSim2 backend. It also requires an authorized
local `peptide_parser` checkout and its own dependencies (including `loguru`).
Set `peptide_parser_root` in the similarity JSON; `third_party/peptide_parser` is
only a placeholder, not a bundled or automatically downloaded private repository.

PeptiVerse uses its own tokenizer, inference code, classifier/regressor resources,
and dependencies. Keep the external snapshot and matching weights together; do
not substitute an updated predictor and call the outputs identical to the paper.
The inspected source snapshot declares Apache-2.0 in its README but is not a
versioned checkout. `external_assets.json` stores hashes of the observed helper,
tokenizer/manifest and KRAS predictor when available; those assets are not bundled.

The SMILES2PEPTIDE checkout used during validation is commit
`f652da392789b9bb3c06e40bdeae9745d0d763b1`. Its source README declares
CC-BY-NC-ND-4.0. The integration shim avoids loading a Gradio web UI; it is not
permission to redistribute or relicense the upstream source. Obtain and use
third-party materials only within their applicable terms.

The KRAS predictor `kras_xgboost_reg.pkl` is an external project/HELM-GPT-derived
asset, not a model trained or validated by this packaging step. Provide it via
`--reward_kras_model_path` or the scorer's `--model_path`. Source version, model
license, training provenance, calibration and feature definition must be checked
before making scientific or redistribution claims. The integration uses radius-3,
2048-bit Morgan fingerprints with the original reverse-sigmoid transformations.
