import os
import sys
import random
from pathlib import Path
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
PROJECT_ROOT = Path(__file__).resolve().parents[3]


try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    # 在交互式环境（如 notebook）中，__file__ 未定义
    print("Warning: Running in interactive mode. Assuming relative paths are correct.")
    # 如果在 notebook 中运行，请手动调整路径或确保 sys.path 正确

def seed_everything(seed: int):
    """
    固定所有随机种子以确保结果可复现。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)

def preprocess_logits_for_metrics(logits, labels):
    """
    在累积 logits 之前对其进行预处理。
    我们只需要 argmax，所以返回 token ID。
    """
    if isinstance(logits, tuple):
        # logits 可能是元组 (如 (prediction_scores, ...))
        logits = logits[0]
    
    # 返回 logits 的 argmax，这将成为 compute_metrics 中的 "predictions"
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    """ 
    计算 Token 级别的准确率 (已修改)
    现在 eval_pred.predictions 是 token ID (来自 preprocess_logits_for_metrics)，
    而不是原始 logits。
    """
    # eval_pred.predictions 已经是 (num_samples, seq_len) 的 token ID
    preds_ids, labels = eval_pred.predictions, eval_pred.label_ids

    # 2. 直接对 token ID 进行移位
    # 预测的 IDs 需要向右移一位来对齐 (preds 预测 label[i+1])
    shift_preds = preds_ids[..., :-1]
    shift_labels = labels[..., 1:]

    not_ignore = shift_labels != -100  # 忽略 padding
    num_targets = not_ignore.sum()
    
    if num_targets == 0:
        return {"accuracy": 0.0}

    # 3. 比较移位后的 ID
    correct = (shift_preds == shift_labels) & not_ignore
    correct = correct.sum()

    accuracy = correct / num_targets

    return {"accuracy": accuracy}

# def compute_metrics(eval_pred):
    """ 
    计算 Token 级别的准确率 (从训练脚本复制)
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    preds = np.argmax(shift_logits, axis=-1)

    not_ignore = shift_labels != -100  # 忽略 padding

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

    eval_data_list = []

    try:
        df = pd.read_csv(args.train_raw_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.train_raw_path}")
        print("Please ensure the path is correct.")
        sys.exit(1)
        
    print(f"数据总行数: {len(df)}")

    df.dropna(subset=['SMILES'], inplace=True)
    df = df[df['SMILES'].str.len() > 0]
    print(f"清洗后数据总行数: {len(df)}")

    train_df, eval_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)
    print(f"训练集大小: {len(train_df)}, 验证集大小: {len(eval_df)}")

    max_seq_length = tokenizer.model_max_length
    eval_smiles_list = eval_df['SMILES'].tolist()

    is_main_process = os.environ.get("RANK", "0") == "0"

    for smiles_string in tqdm(eval_smiles_list, desc="Tokenizing eval data", disable=not is_main_process):
        tokenized_ids = tokenizer.encode(
            smiles_string,
            truncation=True,
            max_length=max_seq_length
        )
        eval_data_list.append(tokenized_ids)

    eval_dataset = MyDataset(eval_data_list)

    return eval_dataset

def setup_eval_args():
    """
    为评估脚本设置参数。
    """
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--model_path', default="../output/best_model_with_mask_trainer/checkpoint-177198", type=str, 
    #                     help='指向已保存的 final (best) 模型目录的路径。')
    parser.add_argument('--model_path', default=str(PROJECT_ROOT / "runs/pretrain/final_model"), type=str,
                    help='指向已保存的 final (best) 模型目录的路径。')
    
    # parser.add_argument('--train_raw_path', default='../data/restored_train.csv', type=str, 
    #                     help='指向原始训练数据 CSV 的路径，以获取验证集拆分。')
    parser.add_argument('--train_raw_path', default=str(PROJECT_ROOT / "data/restored_train.csv"), type=str,
                        help='指向原始训练数据 CSV 的路径，以获取验证集拆分。')
    
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='评估时每个设备的 batch size')
    parser.add_argument('--max_len', default=576, type=int, 
                        help='每个序列的最大长度')
    
    parser.add_argument('--temp_output_dir', default="./eval_output_temp", type=str,
                        help='Trainer 评估时使用的临时输出目录。')
    return parser.parse_args()


if __name__ == '__main__':
    seed_everything(42)
    args = setup_eval_args()
    
    # --- 1. 加载 Tokenizer ---

    tokenizer = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")

    tokenizer.model_max_length = args.max_len
    
    if tokenizer.pad_token_id is None:
        print("Warning: pad_token_id not set. Setting to eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. 加载模型 ---
    print(f"正在从 {args.model_path} 加载训练好的模型...")
    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    # --- 3. 准备验证集 ---
    # 传入 args (其中包含 train_raw_path)，以获取 *相同* 的验证集拆分
    eval_dataset = prepare_datasets(args, tokenizer)

    # --- 4. 准备 Collator ---
    data_collator = DataCollatorForLanguageModeling(tokenizer)

    # --- 5. 定义评估参数 ---
    # 我们只需要设置几个用于 .evaluate() 的参数
    training_args = TrainingArguments(
        output_dir=args.temp_output_dir,       # 必需项，但仅用于临时文件
        per_device_eval_batch_size=args.batch_size,
        fp16=True if torch.cuda.is_available() else False, # 如果有 GPU，使用 fp16 加速评估
        report_to="none"                       # 评估时不需要报告
    )

    # --- 6. 实例化 Trainer ---
    # 这一次，我们只传入评估所需的部分
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # --- 7. 运行评估 ---
    print("--- 开始在验证集上进行评估 ---")
    results = trainer.evaluate()

    print("--- 评估完成 ---")
    print(f"模型: {args.model_path}")
    print(f"验证集大小: {len(eval_dataset)}")
    
    # 清晰地打印结果
    print("\n--- 评估指标 ---")
    for key, value in results.items():
        if "accuracy" in key:
            # 将准确率格式化为百分比
            print(f"  {key}: {value*100:.4f}%")
        else:
            # 其他指标（如 eval_loss）保留4位小数
            print(f"  {key}: {value:.4f}")
