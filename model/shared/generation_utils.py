import re

import torch

from model.shared.pipeline_utils import build_condition_token, configure_condition_token, is_valid_smiles


def clean_generated_sequence(seq, *, condition_token, tokenizer):
    cleaned = seq
    if tokenizer.pad_token:
        cleaned = cleaned.replace(tokenizer.pad_token, "")
    if tokenizer.eos_token:
        cleaned = cleaned.replace(tokenizer.eos_token, "")
    cleaned = cleaned.replace(condition_token, "")
    cleaned = cleaned.replace(" ", "").strip()
    return re.sub(r"^[a-z]+(?=[A-Z\[])", "", cleaned)


def prepare_condition_token(tokenizer, condition_type):
    condition_token = build_condition_token(condition_type)
    start_token_id = configure_condition_token(tokenizer, condition_type)
    return condition_token, start_token_id


def generate_conditioned_outputs(
    model,
    tokenizer,
    condition_type,
    *,
    batch_size,
    max_length,
    temperature,
    top_k,
    top_p,
    device,
    min_length=None,
    num_return_sequences=1,
    skip_special_tokens=False,
):
    condition_token = build_condition_token(condition_type)
    start_token_id = configure_condition_token(tokenizer, condition_type)

    input_ids = torch.full(
        (batch_size, 1),
        start_token_id,
        dtype=torch.long,
        device=device,
    )

    generation_kwargs = {
        "max_length": max_length,
        "do_sample": True,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": num_return_sequences,
    }
    if min_length is not None:
        generation_kwargs["min_length"] = min_length

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output_ids = model.generate(input_ids, **generation_kwargs)
    finally:
        if was_training:
            model.train()

    decoded_sequences = tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)
    return condition_token, start_token_id, decoded_sequences


def build_generation_records(decoded_sequences, *, condition_type, tokenizer, include_raw_output=False):
    condition_token = build_condition_token(condition_type)
    records = []

    for seq in decoded_sequences:
        cleaned = clean_generated_sequence(seq, condition_token=condition_token, tokenizer=tokenizer)
        record = {
            "SMILES": cleaned,
            "Valid": is_valid_smiles(cleaned),
            "Cyclization": condition_type,
        }
        if include_raw_output:
            record["raw_output"] = seq
        records.append(record)

    valid_count = sum(1 for record in records if record["Valid"])
    return records, valid_count
