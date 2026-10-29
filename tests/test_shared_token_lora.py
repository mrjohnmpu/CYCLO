import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from evaluation.analysis import compare_shared_vs_paper_similarity as similarity
from evaluation.analysis import shared_base_property_eval as property_eval
from evaluation.metrics import finalize_shared_token_lora_metrics as metrics
from model.training.ablation.train_shared_strict_token_lora import StrictTokenDataset


ROOT = Path(__file__).resolve().parents[1]
TOPOLOGIES = ("head_to_tail", "sc_to_tail", "disulfide_bridge")


class SharedTokenLoraTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.config = {
            "expected_num_samples": 4,
            "combined_output_dir": str(root / "output"),
            "existing_fifteen_method_summary": str(root / "existing.csv"),
            "shared_entries": [],
        }
        for topology in TOPOLOGIES:
            generated = root / f"{topology}.csv"
            pd.DataFrame({"SMILES": ["C", "O", "", "CC"], "Valid": [True, True, False, True]}).to_csv(generated, index=False)
            metric_path = root / f"{topology}.json"
            metric_path.write_text(json.dumps({
                "IntDiv": .4, "IntDiv2": .3, "Novelty": .8, "SNN/Test": .6,
                "unique@10000": 1., "unique@20000": 1., "unique@30000": 1.,
                "validity": .75, "num_valid": 3, "num_total": 4,
            }))
            self.config["shared_entries"].append({
                "cyclization": topology, "generated_csv": str(generated),
                "metrics_json": str(metric_path),
            })
        methods = ("Token-only", "LoRA-only", "Paper-exact token+LoRA", "Strict token+LoRA", "Full fine-tune")
        existing = []
        for method in methods:
            for topology in TOPOLOGIES:
                row = {column: .5 for column in metrics.SUMMARY_COLUMNS}
                row.update(Method=method, Cyclization=topology, ValidCount=3, TotalCount=4, GeneratedCSV="fixture.csv")
                existing.append(row)
        pd.DataFrame(existing, columns=metrics.SUMMARY_COLUMNS).to_csv(self.config["existing_fifteen_method_summary"], index=False)

    def test_validity_is_recomputed_over_all_rows(self):
        metrics.validate(self.config)
        frame = pd.read_csv(Path(self.config["combined_output_dir"]) / "shared_generation_validation.csv")
        self.assertEqual(frame["TotalCount"].tolist(), [4, 4, 4])
        self.assertEqual(frame["ValidCount"].tolist(), [3, 3, 3])
        self.assertEqual(frame["Validity"].tolist(), [.75, .75, .75])

    def test_incorrect_valid_flag_is_rejected(self):
        path = Path(self.config["shared_entries"][0]["generated_csv"])
        frame = pd.read_csv(path)
        frame.loc[0, "Valid"] = False
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(RuntimeError, "RDKit validity mismatch"):
            metrics.validate(self.config)

    def test_shared_summary_contains_eighteen_rows(self):
        metrics.summarize(self.config)
        frame = pd.read_csv(Path(self.config["combined_output_dir"]) / "metrics_summary.csv")
        self.assertEqual(list(frame.columns), metrics.SUMMARY_COLUMNS)
        self.assertEqual(len(frame), 18)
        self.assertEqual(frame.loc[frame.Method == "Shared token+LoRA", "Cyclization"].tolist(), list(TOPOLOGIES))

    def test_portable_settings_preserve_parameter_budget(self):
        config = metrics.load_config(ROOT / "configs/ablation/shared_strict_token_lora_r64_seed42.json")
        self.assertEqual((config["lora_r"], config["lora_alpha"]), (64, 128))
        self.assertEqual(config["expected_trainable_parameters"], 9439488)
        self.assertEqual(config["expected_condition_token_ids"], dict(zip(TOPOLOGIES, (2140, 2141, 2142))))
        for entry in config["shared_entries"]:
            self.assertTrue(Path(entry["generated_csv"]).is_relative_to(ROOT))
            self.assertTrue(Path(entry["metrics_json"]).is_relative_to(ROOT))

    def test_strict_condition_token_and_padding(self):
        class Tokenizer:
            eos_token_id = 2
            pad_token_id = 0

            def encode(self, value, **kwargs):
                return [4]

        item = StrictTokenDataset(["C"], 2140, Tokenizer(), 4)[0]
        self.assertEqual(item["input_ids"].tolist(), [2140, 4, 2, 0])
        self.assertEqual(item["attention_mask"].tolist(), [1, 1, 1, 0])
        self.assertEqual(item["labels"].tolist(), [2140, 4, 2, -100])

    def test_similarity_contract_and_relative_paths(self):
        similarity.run_internal_tests()
        for name in ("shared_vs_paper_similarity_20260921.json", "shared_vs_paper_head_to_tail_valid_similarity_20260921.json"):
            config = json.loads((ROOT / "evaluation/configs" / name).read_text())
            original = copy.deepcopy(config)
            resolved = similarity.resolve_config(config)
            similarity.validate_config(resolved)
            self.assertTrue(Path(resolved["chemical_space_module"]).is_relative_to(ROOT))
            self.assertEqual(resolved["topology_validation"], original["topology_validation"])
            self.assertEqual(resolved.get("analysis_filter"), original.get("analysis_filter"))

    def test_shared_ppo_is_still_excluded(self):
        self.assertFalse((ROOT / "model/reinforce/train_ppo_v3_shared_multitopology_full_finetune.py").exists())
        self.assertFalse(any((ROOT / "configs").glob("reinforce_ppo_shared*.json")))
        self.assertFalse((ROOT / "evaluation/analysis/evaluate_shared_vs_paper_ppo.py").exists())
        self.assertFalse((ROOT / "evaluation/analysis/evaluate_shared_multitopology_ppo.py").exists())

    def test_property_preparation_preserves_shipped_manifests(self):
        root = Path(self.directory.name)
        jobs = []
        rows = []
        for topology in TOPOLOGIES:
            path = root / f"{topology}_fixture_n30000.csv"
            pd.DataFrame({"SMILES": ["C"] * 30000, "Valid": [True] * 30000}).to_csv(path, index=False)
            jobs.append({"name": topology + "_n30000", "csv_path": str(path)})
            rows.append({"Method": "Shared token+LoRA", "Cyclization": topology, "TotalCount": 30000, "ValidCount": 30000, "Validity": 1., "GeneratedCSV": str(path)})
        config_path = root / "annotation.json"
        config_path.write_text(json.dumps({"inputs": jobs, "device": "cpu"}))
        metric_path = root / "property_metrics.csv"
        pd.DataFrame(rows).to_csv(metric_path, index=False)
        templates = ROOT / "evaluation/plots/configs"
        before = {topology: (templates / f"{topology}_property_sources.json").read_bytes() for topology in TOPOLOGIES}
        output = root / "property_output"
        figures = root / "property_figures"
        with mock.patch.multiple(property_eval, CONFIG=config_path, METRICS=metric_path, OUTPUT=output, FIGURES=figures, PLOT_CONFIGS=templates):
            property_eval.prepare()
        resolved = json.loads((output / "resolved_config.json").read_text())
        self.assertEqual(resolved["output_dir"], str(output / "outputs"))
        for topology in TOPOLOGIES:
            self.assertEqual((templates / f"{topology}_property_sources.json").read_bytes(), before[topology])
            self.assertEqual(len(pd.read_csv(output / f"smoke/inputs/{topology}_n32.csv")), 32)
            manifest = json.loads((figures / f"manifests/{topology}_property_sources_shared_seed42_20260923.json").read_text())
            highlight = next(item for item in manifest["datasets"] if item["id"] == "ours")
            self.assertEqual(highlight["label"], "Shared token+LoRA")
            self.assertTrue(Path(highlight["path"]).is_relative_to(output))


if __name__ == "__main__":
    unittest.main()
