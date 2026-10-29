"""CPU-only release validation; no model training or bulk generation."""
import argparse
import ast
import contextlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/release_validation")
    parser.add_argument("--source-root", type=Path, help="Optional original project for reward/source invariance checks")
    parser.add_argument("--skip-help", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checks = []

    def run(name, function):
        try:
            details = function()
            checks.append({"name": name, "passed": True, "details": details})
            print(f"PASS {name}", flush=True)
        except Exception as exc:
            checks.append({"name": name, "passed": False, "error": repr(exc)})
            print(f"FAIL {name}: {exc}", flush=True)

    files = []
    for directory in ["model", "baselines", "scripts", "evaluation", "utils", "configs", "tests"]:
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("._"))

    def syntax():
        counts = {"python": 0, "json": 0, "shell": 0}
        for path in files:
            if path.suffix == ".py":
                compile(path.read_text(), str(path), "exec")
                counts["python"] += 1
            elif path.suffix == ".json":
                json.loads(path.read_text())
                counts["json"] += 1
            elif path.suffix == ".sh":
                subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
                counts["shell"] += 1
        return counts
    run("syntax_and_config_parse", syntax)

    def data():
        from scripts.prepare_data import verify
        return verify(ROOT / "data", unpack=True)
    run("complete_dataset_hashes_counts_and_unpack", data)

    def tests():
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
        buffer = io.StringIO()
        result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
        (args.out_dir / "unit_tests.log").write_text(buffer.getvalue())
        if not result.wasSuccessful():
            raise AssertionError(buffer.getvalue())
        return {"tests_run": result.testsRun, "skipped": [(str(test), reason) for test, reason in result.skipped]}
    run("retained_unittest_suite", tests)

    def smiles_parser():
        spec = importlib.util.spec_from_file_location("release_smiles2peptide_tests", ROOT / "tests/test_smiles2peptide_batch.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.test_adapter_imports_app()
        module.test_batch_conversion_smoke()
        return {"functions_tested": 2}
    run("smiles2peptide_fixture", smiles_parser)

    def functional():
        import numpy as np
        import pandas as pd
        from rdkit import Chem
        from evaluation.metrics.core import build_valid_mask, compute_prefix_uniqueness
        from evaluation.property_pipeline.data_inventory import DatasetSpec
        from evaluation.property_pipeline.mol_cache import build_dataset_cache
        from evaluation.property_pipeline.rdkit_properties import compute_basic_rdkit_property_table
        from model.reinforce.train_ppo_kras_v2_full_finetune import kd_sharp_kras_reward, calculate_total_reward, exp_hemo_reward
        from model.training.lora import dataset
        from utils.process_pepinvent_generated_data import process_generated_data
        frame = pd.DataFrame({"SMILES": ["CCO", "OCC", "invalid", "CC(=O)N"]})
        assert build_valid_mask(frame, "SMILES", "Valid").tolist() == [True, True, False, True]
        assert abs(compute_prefix_uniqueness(frame, "SMILES", "Valid", [4])["unique@4"] - 2 / 3) < 1e-12
        path = args.out_dir / "metric_fixture.csv"
        frame.to_csv(path, index=False)
        cache = build_dataset_cache(DatasetSpec("fixture", "Fixture", str(path), cyclization_col=None))
        properties = compute_basic_rdkit_property_table(cache)
        assert len(properties) == 3 and properties.iloc[0].HBA == 1 and properties.iloc[0].HBD == 1
        assert abs(properties.iloc[0].MW - 46.041864812) < 1e-6
        assert kd_sharp_kras_reward(1) == 0.525
        assert kd_sharp_kras_reward(0) > kd_sharp_kras_reward(1) > kd_sharp_kras_reward(2)
        assert exp_hemo_reward(0) == 1 and exp_hemo_reward(1) == 0.05
        assert abs(calculate_total_reward(.5, .5, .5) - .5) < 1e-12
        output = process_generated_data(pd.DataFrame({"Input": ["C|?|N"], "Generated_smi_1": ["C(=O)"], "Generated_smi_2": [np.nan]}))
        assert output.SMILES.tolist() == ["CC(=O)N"]
        expected = json.loads((ROOT / "docs/provenance/lora.json").read_text())
        class ListDataset:
            def __init__(self, values, *unused): self.values = list(values)
        with unittest.mock.patch.object(dataset, "TypeSpecificDataset", ListDataset):
            splits, summaries = dataset.PeptideDataProcessor(None).load_and_split(str(ROOT / "data/restored_validation.csv"), list(expected))
        for topology, summary in summaries.items():
            assert summary == expected[topology]["split_summary"]
        from model.training.ablation import train_token_only_frozen_prior as ablation
        options = argparse.Namespace(data_path=str(ROOT / "data/restored_validation.csv"), peptide_types=list(expected), split_seed=42, seed=42, train_ratio=.8, val_ratio=.1, max_samples_per_type=None, max_length=8)
        with unittest.mock.patch.object(ablation, "TokenConditionDataset", ListDataset):
            ablation_splits, ablation_summary, split_manifest = ablation.prepare_splits(options, None)
        assert summaries == ablation_summary
        for topology in expected:
            for split in ["train", "val", "test"]:
                assert splits[topology][split].values == ablation_splits[topology][split].values
        split_manifest.to_csv(args.out_dir / "ablation_split_manifest.csv", index=False)
        from scripts.prepare_data import sha256
        manifest_hash = sha256(args.out_dir / "ablation_split_manifest.csv")
        target_hash = json.loads((ROOT / "configs/ablation/strict_token_lora_seed42.json").read_text())["expected_manifest_sha256"]
        assert manifest_hash == target_hash, (manifest_hash, target_hash)
        return {"rdkit_valid": 3, "rows": 4, "split_summary": summaries, "ablation_manifest_sha256": manifest_hash}
    run("rdkit_rewards_pepinvent_and_ablation_splits", functional)

    def reward_invariance():
        for name in ["train_ppo_v3_full_finetune.py", "train_ppo_kras_v2_full_finetune.py"]:
            relative = Path("model/reinforce") / name
            original = ast.parse((args.source_root / relative).read_text())
            release = ast.parse((ROOT / relative).read_text())
            selected = {node.name: ast.dump(node, include_attributes=False) for node in original.body if isinstance(node, ast.FunctionDef) and ("reward" in node.name or node.name in {"compute_rollout_batch", "compute_rewards_and_advantages", "ppo_update"})}
            published = {node.name: ast.dump(node, include_attributes=False) for node in release.body if isinstance(node, ast.FunctionDef) and node.name in selected}
            assert selected == published, name
        return "Reward mappings, rollout/GAE and PPO update ASTs unchanged"
    if args.source_root:
        run("original_reward_and_ppo_invariance", reward_invariance)

    def plotting():
        import numpy as np
        import pandas as pd
        from evaluation.plots import head_to_tail_property_plot as plot
        rng = np.random.default_rng(42)
        entries = []
        for index, method in enumerate(plot.DEFAULT_METHOD_ORDER):
            frame = pd.DataFrame({property_name: rng.normal(3 + index / 10, .5, 40) for property_name in plot.DEFAULT_PROPERTIES})
            frame["Cyclization"] = "head_to_tail"
            path = args.out_dir / f"plot_{method}.csv"
            frame.to_csv(path, index=False)
            entries.append({"id": method, "label": "TopoCycGPT" if method == "ours" else method, "path": str(path)})
        manifest_path = args.out_dir / "plot_sources.json"
        manifest_path.write_text(json.dumps({"datasets": entries}))
        command = [sys.executable, str(ROOT / "evaluation/plots/head_to_tail_property_plot.py"), "--manifest", str(manifest_path), "--out-dir", str(args.out_dir / "plot_smoke"), "--formats", "png", "--max-rows-per-dataset", "40"]
        proc = subprocess.run(command, cwd="/tmp", capture_output=True, text=True, timeout=120)
        (args.out_dir / "plot_smoke.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode: raise RuntimeError(proc.stderr[-2500:])
        summary = pd.DataFrame([{"Method": method, "Cyclization": topology, "Validity": .8, "Novelty": .9, "IntDiv": .7} for method in ["Token-only", "LoRA-only", "Strict token+LoRA", "Full fine-tune", "TopoCycGPT"] for topology in ["head_to_tail", "sc_to_tail", "disulfide_bridge"]])
        summary_path = args.out_dir / "ablation_metrics.csv"
        summary.to_csv(summary_path, index=False)
        subprocess.run([sys.executable, str(ROOT / "evaluation/plots/ablation_metrics_plot.py"), "--csv", str(summary_path), "--out-dir", str(args.out_dir / "ablation_plot_smoke"), "--formats", "png", "pdf", "svg"], cwd="/tmp", capture_output=True, check=True, timeout=60)
        paths = list((args.out_dir / "plot_smoke").rglob("*.png"))
        assert len(paths) >= 12
        return {"synthetic_only": True, "property_plot_pngs": len(paths), "ablation_formats": ["png", "pdf", "svg"]}
    run("plotting_smoke_synthetic_not_paper_data", plotting)

    if not args.skip_help:
        env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "", "TRL_EXPERIMENTAL_SILENCE": "1"}
        with tempfile.TemporaryDirectory(prefix="cyclo_outside_") as outside:
            for path in sorted(files):
                if path.suffix != ".py" or path == Path(__file__) or path.parts[-2] == "tests": continue
                text = path.read_text()
                if "ArgumentParser" not in text or "__main__" not in text: continue
                def help_check(path=path):
                    result = subprocess.run([sys.executable, str(path), "--help"], cwd=outside, env=env, capture_output=True, text=True, timeout=60)
                    log = args.out_dir / "help" / (str(path.relative_to(ROOT)).replace("/", "__") + ".txt")
                    log.parent.mkdir(exist_ok=True)
                    log.write_text(result.stdout + result.stderr)
                    if result.returncode != 0: raise RuntimeError((result.stdout + result.stderr)[-2200:])
                    return {"cwd": "outside_repository", "original_project_pythonpath": False}
                run("help:" + str(path.relative_to(ROOT)), help_check)

    def import_isolation():
        original = str(args.source_root.resolve()) if args.source_root else None
        imported = []
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if path and Path(path).is_relative_to(ROOT): imported.append({"module": name, "path": path})
            if path and original and str(path).startswith(original) and "/third_party/" not in str(path):
                raise AssertionError(f"Original project module imported: {name}: {path}")
        return imported
    run("import_isolation", import_isolation)
    report = {"checks": checks, "passed": all(check["passed"] for check in checks), "no_full_training_or_bulk_generation": True}
    (args.out_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    from unittest import mock
    main()
