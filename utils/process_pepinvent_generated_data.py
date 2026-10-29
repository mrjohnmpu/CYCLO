"""Reassemble PepINVENT fragment outputs using the original replacement rule."""
import argparse
from pathlib import Path

import pandas as pd


def process_generated_data(df: pd.DataFrame) -> pd.DataFrame:
    generated_cols = [col for col in df.columns if col.startswith("Generated_smi_")]
    if "Input" not in df or not generated_cols:
        raise ValueError("Expected Input and at least one Generated_smi_* column")
    sequences = []
    for _, row in df.iterrows():
        input_parts = row["Input"].split("|")
        for col in generated_cols:
            if pd.isna(row[col]):
                continue
            gen_parts = row[col].split("|")
            final_parts = []
            gen_index = 0
            for part in input_parts:
                if part == "?":
                    if gen_index < len(gen_parts):
                        final_parts.append(gen_parts[gen_index])
                        gen_index += 1
                else:
                    final_parts.append(part)
            sequences.append("".join(final_parts))
    return pd.DataFrame(sequences, columns=["SMILES"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    result = process_generated_data(pd.read_csv(args.input_csv))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Saved {len(result)} sequences to {args.output_csv}")


if __name__ == "__main__":
    main()
