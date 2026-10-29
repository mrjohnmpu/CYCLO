import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from utils.release_paths import peptiverse_root


def parse_args():
    parser = argparse.ArgumentParser(description="Plot distribution comparisons for generated peptide sets.")
    parser.add_argument("--config", type=str, required=True, help="CSV file with name and path columns.")
    parser.add_argument("--img_folder", type=str, required=True, help="Directory to store plots.")
    parser.add_argument("--test_csv", type=str, required=True, help="Reference CSV file.")
    parser.add_argument("--cyclization_type", type=str, default="head_to_tail", help="Cyclization type used for filtering.")
    parser.add_argument("--metrics", nargs="+", default=["permeability_pampa", "hemolysis_safe", "weight", "logP", "SA", "QED"])
    parser.add_argument("--plot_type", type=str, default="all", choices=["kde", "box", "halfviolin", "both", "all"])
    parser.add_argument("--file_formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--max_rows_per_dataset", type=int, default=3000, help="Subsample each dataset before plotting expensive metrics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--smiles_col", type=str, default="SMILES")
    parser.add_argument("--type_col", type=str, default="Cyclization")
    parser.add_argument("--valid_col", type=str, default="Valid")
    parser.add_argument("--peptiverse_manifest_path", type=str, default=str(peptiverse_root() / "best_models.txt"))
    parser.add_argument("--peptiverse_weight_root", type=str, default=str(peptiverse_root()))
    parser.add_argument("--show_fliers", action="store_true")
    parser.add_argument("--scatter_size", type=float, default=1.0)
    parser.add_argument("--scatter_alpha", type=float, default=0.8)
    parser.add_argument("--jitter_mean_shift", type=float, default=-0.10)
    parser.add_argument("--jitter_scale", type=float, default=0.05)
    parser.add_argument("--violin_alpha", type=float, default=0.7)
    parser.add_argument("--box_width", type=float, default=0.12)
    return parser.parse_args()


def save_figure(target, output_dir: Path, stem: str, file_formats):
    for file_format in file_formats:
        save_kwargs = {"dpi": 300} if file_format == "png" else {}
        target.savefig(output_dir / f"{stem}.{file_format}", **save_kwargs)


def load_dataset(path: Path, *, smiles_col: str, cyclization_type: str, type_col: str, valid_col: str, max_rows: int, rng):
    df = pd.read_csv(path)
    if type_col in df.columns:
        df = df[df[type_col] == cyclization_type]
    if valid_col in df.columns:
        df = df[df[valid_col].astype(bool)]
    df = df[df[smiles_col].notna()].copy()
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df[df[smiles_col] != ""]

    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=int(rng.integers(0, 2**31 - 1)))

    df["ROMol"] = df[smiles_col].map(Chem.MolFromSmiles)
    df = df[df["ROMol"].notna()].copy()
    return df


def build_predictor(args):
    sys.path.insert(0, str(peptiverse_root().parent))
    from PeptiVerse.inference import PeptiVersePredictor

    return PeptiVersePredictor(
        manifest_path=args.peptiverse_manifest_path,
        classifier_weight_root=args.peptiverse_weight_root,
        device=args.device,
    )


def predict_permeability_pampa(smiles_list, predictor):
    values = []
    for smiles in smiles_list:
        try:
            output = predictor.predict_property("permeability_pampa", mode="smiles", input_str=smiles)
            values.append(float(output["score"]))
        except Exception:
            values.append(None)
    return values


def predict_hemolysis_safe(smiles_list, predictor):
    values = []
    for smiles in smiles_list:
        try:
            output = predictor.predict_property("hemolysis", mode="smiles", input_str=smiles)
            values.append(1.0 - float(output["score"]))
        except Exception:
            values.append(None)
    return values


def build_distributions(datasets, metrics, predictor):
    from moses.metrics import QED, SA, logP, weight

    distributions = OrderedDict()
    rdkit_metrics = {
        "weight": lambda mols, _: [weight(mol) for mol in mols],
        "logP": lambda mols, _: [logP(mol) for mol in mols],
        "SA": lambda mols, _: [SA(mol) for mol in mols],
        "QED": lambda mols, _: [QED(mol) for mol in mols],
    }

    for metric_name in metrics:
        distributions[metric_name] = OrderedDict()
        for dataset_name, df in datasets.items():
            smiles = df["SMILES"].tolist()
            mols = df["ROMol"].tolist()
            if metric_name == "permeability_pampa":
                values = predict_permeability_pampa(smiles, predictor)
            elif metric_name == "hemolysis_safe":
                values = predict_hemolysis_safe(smiles, predictor)
            else:
                values = rdkit_metrics[metric_name](mols, predictor)
            distributions[metric_name][dataset_name] = [value for value in values if value is not None]
    return distributions


def build_long_df(distributions):
    rows = []
    for metric_name, per_dataset in distributions.items():
        for dataset_name, values in per_dataset.items():
            for value in values:
                rows.append((metric_name, dataset_name, value))
    return pd.DataFrame(rows, columns=["metric", "model", "value"])


def plot_kde_for_metric(metric_name, distributions_for_metric, save_path, file_formats):
    plt.figure(figsize=(7.5, 5.0))
    for model_name, values in distributions_for_metric.items():
        if not values:
            continue
        if "Test" in distributions_for_metric and model_name != "Test":
            distance = wasserstein_distance(distributions_for_metric["Test"], values)
        elif model_name == "Test":
            distance = 0.0
        else:
            distance = float("nan")
        sns.kdeplot(x=values, fill=True, linewidth=2.0, label=f"{model_name} ({distance:0.2g})")
    plt.title(metric_name)
    plt.legend()
    plt.tight_layout()
    save_figure(plt, save_path, f"{metric_name}_kde", file_formats)
    plt.close()


def plot_box_for_metric(metric_name, long_df, model_order, save_path, show_fliers=False, file_formats=None):
    df_metric = long_df[long_df["metric"] == metric_name].copy()
    plt.figure(figsize=(max(8.0, 0.6 * len(model_order)), 5.0))
    ax = sns.boxplot(data=df_metric, x="model", y="value", order=model_order, showfliers=show_fliers)
    ax.set_xlabel("")
    ax.set_ylabel(metric_name)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_figure(plt, save_path, f"{metric_name}_box", file_formats or ["png", "pdf"])
    plt.close()


def plot_halfviolin_box_scatter_for_metric(
    metric_name,
    distributions_for_metric,
    model_order,
    save_path,
    *,
    show_fliers=False,
    scatter_size=1.0,
    scatter_alpha=0.8,
    jitter_mean_shift=-0.10,
    jitter_scale=0.05,
    violin_alpha=0.7,
    box_width=0.12,
    file_formats=None,
):
    data = []
    for model_name in model_order:
        values = distributions_for_metric.get(model_name, [])
        data.append(values)

    colors = sns.color_palette("tab10", n_colors=len(model_order))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * len(model_order)), 5.0))

    for idx, values in enumerate(data):
        if not values:
            continue
        violin = ax.violinplot(values, positions=[idx], showmeans=False, showmedians=False, showextrema=False)
        for body in violin["bodies"]:
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.clip(vertices[:, 0], idx, np.inf)
            body.set_facecolor(colors[idx])
            body.set_edgecolor("black")
            body.set_alpha(violin_alpha)

    ax.boxplot(
        data,
        positions=np.arange(len(data)),
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor="white", color="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", color="black", markersize=1),
        showfliers=show_fliers,
    )

    rng = np.random.default_rng(42)
    for idx, values in enumerate(data):
        if not values:
            continue
        jittered_x = rng.uniform(
            low=(idx + jitter_mean_shift - jitter_scale),
            high=(idx + jitter_mean_shift + jitter_scale),
            size=len(values),
        )
        ax.scatter(jittered_x, values, color=colors[idx], alpha=scatter_alpha, s=scatter_size)

    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels(model_order, rotation=30, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} (scatter + box + half violin)")
    plt.tight_layout()
    save_figure(plt, save_path, f"{metric_name}_halfviolin", file_formats or ["png", "pdf"])
    plt.close()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.img_folder).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_df = pd.read_csv(args.config)
    datasets = OrderedDict()
    datasets["Test"] = load_dataset(
        Path(args.test_csv),
        smiles_col=args.smiles_col,
        cyclization_type=args.cyclization_type,
        type_col=args.type_col,
        valid_col=args.valid_col,
        max_rows=args.max_rows_per_dataset,
        rng=rng,
    )

    for _, row in config_df.iterrows():
        datasets[str(row["name"])] = load_dataset(
            Path(row["path"]),
            smiles_col=args.smiles_col,
            cyclization_type=args.cyclization_type,
            type_col=args.type_col,
            valid_col=args.valid_col,
            max_rows=args.max_rows_per_dataset,
            rng=rng,
        )

    predictor = build_predictor(args)
    distributions = build_distributions(datasets, args.metrics, predictor)
    long_df = build_long_df(distributions)
    model_order = list(datasets.keys())

    sns.set_style("whitegrid")
    for metric_name in args.metrics:
        if args.plot_type in ("kde", "both", "all"):
            plot_kde_for_metric(metric_name, distributions[metric_name], output_dir, args.file_formats)
        if args.plot_type in ("box", "both", "all"):
            plot_box_for_metric(
                metric_name,
                long_df,
                model_order,
                output_dir,
                show_fliers=args.show_fliers,
                file_formats=args.file_formats,
            )
        if args.plot_type in ("halfviolin", "all"):
            plot_halfviolin_box_scatter_for_metric(
                metric_name,
                distributions[metric_name],
                model_order,
                output_dir,
                show_fliers=args.show_fliers,
                scatter_size=args.scatter_size,
                scatter_alpha=args.scatter_alpha,
                jitter_mean_shift=args.jitter_mean_shift,
                jitter_scale=args.jitter_scale,
                violin_alpha=args.violin_alpha,
                box_width=args.box_width,
                file_formats=args.file_formats,
            )

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
