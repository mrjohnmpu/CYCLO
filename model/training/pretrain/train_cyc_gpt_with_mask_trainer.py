import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from model.shared.pipeline_utils import load_tokenizer, parse_args_with_config, seed_everything


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_model_dir', default="output/best_model_with_mask_trainer", type=str,
                        help='Trainer will save checkpoint to this directory')
    parser.add_argument('--final_model_path', default="output/final_model_with_mask_trainer", type=str, help='Final model save path')
    parser.add_argument('--train_raw_path', default='data/restored_train.csv', type=str, help='Training data path (CSV with a "SMILES" column)')
    parser.add_argument('--tokenizer_path', default=None, type=str,
                        help='Optional local tokenizer directory; otherwise use the original MolGPT tokenizer.')
    parser.add_argument('--batch_size', default=32, type=int, required=False,
                        help='per_device batch size')
    parser.add_argument('--accumulation_steps', default=4, type=int, required=False)
    parser.add_argument('--epochs', default=1001, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=10000, type=int, required=False, help='warm up steps')
    parser.add_argument('--warmup_ratio', default=None, type=float, required=False,
                        help='Optional warmup ratio in [0, 1). If set, it overrides warmup_steps.')
    parser.add_argument('--lr', default=5e-5, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='logging steps')
    parser.add_argument('--patience', default=10, type=int, required=False, help='early stopping patience')
    parser.add_argument('--max_len', default=576, type=int, required=False, help='The max length for each sequence')
    args = parse_args_with_config(parser)
    args.best_model_dir = resolve_repo_path(args.best_model_dir)
    args.final_model_path = resolve_repo_path(args.final_model_path)
    args.train_raw_path = resolve_repo_path(args.train_raw_path)
    if args.warmup_ratio is not None and not 0 <= args.warmup_ratio < 1:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {args.warmup_ratio}")
    return args


def resolve_repo_path(path_str):
    if os.path.isabs(path_str):
        return path_str

    # Support both project-root-relative paths ("output/...") and the legacy
    # config style that used "../..." when this script lived under model/.
    base_dir = os.path.join(PROJECT_ROOT, "model") if path_str.startswith("..") else PROJECT_ROOT
    return os.path.abspath(os.path.join(base_dir, path_str))



def compute_metrics(eval_pred):
    """
    Calculate token-level accuracy
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # --- 1. Shift ---
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    # --- 2. Predictions ---
    preds = np.argmax(shift_logits, axis=-1)

    # --- 3. Accuracy ---
    not_ignore = shift_labels != -100  # Ignore padding

    num_targets = not_ignore.sum()
    if num_targets == 0:
        return {"accuracy": 0.0}

    correct = (preds == shift_labels) & not_ignore
    correct = correct.sum()

    accuracy = correct / num_targets

    return {"accuracy": accuracy}



class DataCollatorForLanguageModeling:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id.")

    def __call__(self, batch):
        max_len = max(len(seq) for seq in batch)
        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []
        for seq in batch:
            pad_len = max_len - len(seq)
            padded_inputs = seq + [self.pad_token_id] * pad_len
            input_ids_batch.append(padded_inputs)
            padded_labels = seq + [-100] * pad_len
            labels_batch.append(padded_labels)
            mask = [1] * len(seq) + [0] * pad_len
            attention_mask_batch.append(mask)
        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long)
        }


def prepare_datasets(args, tokenizer):
    data_list = []
    eval_data_list = []

    df = pd.read_csv(args.train_raw_path)
    print("Total rows:{}".format(len(df)))

    df.dropna(subset=['SMILES'], inplace=True)
    df = df[df['SMILES'].str.len() > 0]
    print("Total rows after cleaning: {}".format(len(df)))

    train_df, eval_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)
    print(f"Training set size: {len(train_df)}, Validation set size: {len(eval_df)}")

    max_seq_length = tokenizer.model_max_length
    train_smiles_list = train_df['SMILES'].tolist()
    eval_smiles_list = eval_df['SMILES'].tolist()

    # Only the main process should show the progress bar to avoid cluttering the output in distributed training
    is_main_process = os.environ.get("RANK", "0") == "0"

    for smiles_string in tqdm(train_smiles_list, desc="Tokenizing training data", disable=not is_main_process):
        tokenized_ids = tokenizer.encode(
            smiles_string,
            truncation=True,
            max_length=max_seq_length
        )
        data_list.append(tokenized_ids)

    train_dataset = MyDataset(data_list)

    for smiles_string in tqdm(eval_smiles_list, desc="Tokenizing eval data", disable=not is_main_process):
        tokenized_ids = tokenizer.encode(
            smiles_string,
            truncation=True,
            max_length=max_seq_length
        )
        eval_data_list.append(tokenized_ids)

    eval_dataset = MyDataset(eval_data_list)

    return train_dataset, eval_dataset



def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    seed_everything(42)
    args = setup_args()
    use_fp16 = torch.cuda.is_available()
    warmup_steps = 0 if args.warmup_ratio is not None else args.warmup_steps
    warmup_ratio = 0.0 if args.warmup_ratio is None else args.warmup_ratio
    warmup_desc = (
        f"Warmup ratio: {warmup_ratio}"
        if args.warmup_ratio is not None
        else f"Warmup steps: {warmup_steps}"
    )

    print(f"Resolved train_raw_path: {args.train_raw_path}")
    print(f"Resolved best_model_dir: {args.best_model_dir}")
    print(f"Resolved final_model_path: {args.final_model_path}")
    print(f"Gradient accumulation steps: {args.accumulation_steps}")
    print(f"Logging steps: {args.log_step}")
    print(warmup_desc)
    print(f"fp16 enabled: {use_fp16}")

    # --- 1. Load tokenizer and model ---
    tokenizer = load_tokenizer(args.tokenizer_path or "jonghyunlee/MolGPT_pretrained-by-ZINC15", model_max_length=args.max_len)

    model_config = GPT2Config(
        architectures=["GPT2LMHeadModel"],
        model_type="GPT2LMHeadModel",
        vocab_size=tokenizer.vocab_size,
        n_positions=args.max_len,
        n_ctx=args.max_len,
        n_embd=768,
        n_layer=12,
        n_head=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        task_specific_params={
            "text-generation": {
                "do_sample": True,
                "max_length": args.max_len
            }
        }
    )
    model = GPT2LMHeadModel(config=model_config)

    # --- 2. Prepare Datasets ---
    train_dataset, eval_dataset = prepare_datasets(args, tokenizer)

    # --- 3. Prepare Collator ---
    data_collator = DataCollatorForLanguageModeling(tokenizer)

    # --- 4. Define Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.best_model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        max_grad_norm=args.max_grad_norm,

        # Evaluation & Logging
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.log_step,
        report_to="tensorboard",

        # Saving
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
        greater_is_better=False,

        fp16=use_fp16,
    )

    # --- 5. Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    print("Starting training...")
    trainer.train()

    # (Due to `load_best_model_at_end=True`, the trainer will automatically load the best model after training, so we can directly save it.)
    print("Saving the best model...")
    trainer.save_model(args.final_model_path)
    print(f"The model has been saved to: {args.final_model_path}")
