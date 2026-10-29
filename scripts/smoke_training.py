"""Two tiny CPU optimizer updates using publication pretraining/LoRA interfaces."""
import argparse
import ast
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/training_api_smoke")
    args = parser.parse_args()
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from transformers import Trainer
    from model.training.pretrain import train_cyc_gpt_with_mask_trainer as pretrain
    from model.training.lora.dataset import TypeSpecificDataset
    from model.training.lora.trainer import LoRAPeptideTrainer
    from model.shared.pipeline_utils import seed_everything

    torch.set_num_threads(1)
    seed_everything(42)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "CCO": 4, "CCN": 5}, unk_token="<unk>")), pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=6, n_positions=8, n_ctx=8, n_embd=16, n_layer=1, n_head=2, bos_token_id=1, eos_token_id=2, pad_token_id=0))
    dataset = pretrain.MyDataset([[1, 4, 2], [1, 5, 2]])
    options = argparse.Namespace(best_model_dir=str(args.out_dir / "pretrain"), epochs=1, batch_size=2, accumulation_steps=1, lr=5e-5, max_grad_norm=1., log_step=1, patience=1)
    # One-step fixtures need no warmup, otherwise their only update has LR zero.
    namespace = {**vars(pretrain), "model": model, "tokenizer": tokenizer, "args": options, "warmup_steps": 0, "warmup_ratio": 0., "use_fp16": False, "train_dataset": dataset, "eval_dataset": dataset, "data_collator": pretrain.DataCollatorForLanguageModeling(tokenizer)}
    tree = ast.parse(Path(pretrain.__file__).read_text())
    # Instantiate the actual source call sites without executing its full main.
    for name in ["training_args", "trainer"]:
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), pretrain.__file__, "exec"), namespace)
    pretrain_before = {name: value.detach().clone() for name, value in model.named_parameters()}
    train_output = namespace["trainer"].train()
    assert train_output.global_step == 1 and math.isfinite(train_output.training_loss)
    pretrain_changed = [name for name, value in model.named_parameters() if not torch.equal(pretrain_before[name], value.detach())]
    assert pretrain_changed, "Pretraining completed without updating weights"
    base = args.out_dir / "tiny_base"
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    trainer = LoRAPeptideTrainer(base_model_path=str(base), tokenizer=tokenizer, output_dir=str(args.out_dir / "lora"), seed=42, base_tokenizer_path=str(base), command_args=["synthetic_cpu_smoke"])
    trainer.prepare_tokenizer(["head_to_tail"])
    samples = TypeSpecificDataset(["CCO", "CCN"], "head_to_tail", tokenizer, max_length=8)
    lora_audit = {}

    class AuditedTrainer(Trainer):
        def train(self, *train_args, **train_kwargs):
            before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
            result = super().train(*train_args, **train_kwargs)
            changed = [name for name, value in self.model.named_parameters() if not torch.equal(before[name], value.detach())]
            assert any("lora_" in name for name in changed), "LoRA weights did not update"
            assert all(value.requires_grad for name, value in self.model.named_parameters() if name in changed), "Frozen weights changed"
            lora_audit.update(global_step=result.global_step, changed_parameters=changed, frozen_parameters_unchanged=True)
            return result

    with patch("model.training.lora.trainer.Trainer", AuditedTrainer):
        summary = trainer.train_single_type(peptide_type="head_to_tail", train_dataset=samples, val_dataset=samples, test_dataset=samples, split_summary={"total": 2, "train": 2, "val": 2, "test": 2}, num_epochs=1, batch_size=2, learning_rate=1e-4, warmup_ratio=0., grad_accum_steps=1, early_stop_patience=1, save_total_limit=1, log_steps=1, weight_decay=.01, lora_r=2, lora_alpha=4, lora_dropout=.1, target_modules=["c_attn", "c_proj", "c_fc"])
    assert math.isfinite(summary["train_loss"]) and math.isfinite(summary["best_eval_loss"])
    assert lora_audit["global_step"] == 1
    report = {"passed": True, "synthetic_only": True, "cpu_only": model.device.type == "cpu", "pretrain_updates": train_output.global_step, "pretrain_changed_parameters": pretrain_changed, "lora_updates": lora_audit["global_step"], "lora_weight_audit": lora_audit, "lora_summary": summary, "not_a_paper_experiment": True}
    (args.out_dir / "training_smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "lora_summary"}, indent=2))


if __name__ == "__main__":
    main()
