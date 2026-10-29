import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.shared.pipeline_utils import parse_args_with_config, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Sample and draw generated molecules from any CSV.")
    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--smiles_col", type=str, default="SMILES")
    parser.add_argument("--valid_col", type=str, default="Valid")
    parser.add_argument("--group_col", type=str, default="condition_type")
    parser.add_argument("--group_value", type=str, default=None)
    parser.add_argument("--samples_per_group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_unique", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_valid", action=argparse.BooleanOptionalAction, default=True)
    args = parse_args_with_config(parser)
    missing = [name for name in ("input_csv", "output_dir", "group_value") if not getattr(args, name)]
    if missing:
        parser.error(
            "Missing required arguments after loading config: " + ", ".join(f"--{name}" for name in missing)
        )
    return args


def build_subset(df: pd.DataFrame, args) -> pd.DataFrame:
    subset = df[df[args.group_col] == args.group_value].copy()
    if args.require_valid:
        subset = subset[subset[args.valid_col] == True].copy()
    subset = subset.dropna(subset=[args.smiles_col])
    subset[args.smiles_col] = subset[args.smiles_col].astype(str).str.strip()
    subset = subset[subset[args.smiles_col].str.len() > 0]
    if args.require_unique:
        subset = subset.drop_duplicates(subset=[args.smiles_col]).copy()
    if subset.empty:
        raise ValueError(f"No rows available after filtering for {args.group_value}")
    sample_count = min(args.samples_per_group, len(subset))
    return subset.sample(n=sample_count, random_state=args.seed).reset_index(drop=True)


def main():
    args = parse_args()
    seed_everything(args.seed)

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    sampled = build_subset(df, args)

    mols = []
    legends = []
    exported_rows = []
    for idx, row in sampled.iterrows():
        smiles = row[args.smiles_col]
        mols.append(Chem.MolFromSmiles(smiles))
        legends.append(f"{args.group_value}\n#{idx + 1}")
        exported_rows.append(
            {
                "group_value": args.group_value,
                "sample_rank": idx + 1,
                "SMILES": smiles,
            }
        )

    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=args.samples_per_group,
        subImgSize=(320, 280),
        legends=legends,
        useSVG=False,
    )
    image.save(str(output_dir / "sampled_grid.png"))
    pd.DataFrame(exported_rows).to_csv(output_dir / "sampled_sequences.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "group_col": args.group_col,
                "group_value": args.group_value,
                "smiles_col": args.smiles_col,
                "valid_col": args.valid_col,
                "samples_per_group": args.samples_per_group,
                "seed": args.seed,
                "require_unique": args.require_unique,
                "require_valid": args.require_valid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
