#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdFingerprintGenerator


MM_PER_INCH = 25.4
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "rl_after_joint_property_distribution.json"
DEFAULT_OUTPUT_NAME = "rl_after_joint_property_distribution"
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot after-RL joint permeability and hemolysis distribution.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-size-per-topology", type=int, default=None)
    parser.add_argument("--formats", nargs="+", choices=["pdf", "svg", "png"], default=None)
    return parser.parse_args()


def mm_to_inches(value: float) -> float:
    return value / MM_PER_INCH


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config.get("datasets"):
        raise ValueError("Config must contain non-empty datasets.")
    for key in ("sequence", "smiles", "permeability", "hemolysis"):
        if key not in config.get("columns", {}):
            raise ValueError(f"Config columns is missing {key!r}.")
    return config


def prepare_layout(out_dir: Path) -> dict[str, Path]:
    layout = {
        "root": out_dir,
        "figures": out_dir / "figures",
        "tables": out_dir / "tables",
        "assets": out_dir / "assets",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def apply_publication_style(config: dict[str, Any]) -> str:
    font_name = str(config.get("font_family", "Arial"))
    font_path = config.get("font_path")
    if font_path:
        font_file = Path(str(font_path))
        if not font_file.is_file():
            raise FileNotFoundError(f"Font file not found: {font_file}")
        font_manager.fontManager.addfont(str(font_file))
        font_name = font_manager.FontProperties(fname=str(font_file)).get_name()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_name, "Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return font_name


def required_columns(config: dict[str, Any]) -> list[str]:
    columns = config["columns"]
    return list(dict.fromkeys([
        columns["sequence"],
        columns["smiles"],
        columns["permeability"],
        columns["hemolysis"],
    ]))


def load_and_sample(
    config: dict[str, Any],
    *,
    sample_size_per_topology: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = config["columns"]
    seed = int(config.get("random_seed", 20260707))
    p_col = columns["permeability"]
    h_col = columns["hemolysis"]
    required = required_columns(config)
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for index, dataset in enumerate(config["datasets"]):
        path = Path(str(dataset["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        raw = pd.read_csv(path)
        missing = [column for column in required if column not in raw.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        work = raw.copy()
        work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
        work[h_col] = pd.to_numeric(work[h_col], errors="coerce")
        nonempty_sequence = work[columns["sequence"]].notna() & (work[columns["sequence"]].astype(str).str.len() > 0)
        nonempty_smiles = work[columns["smiles"]].notna() & (work[columns["smiles"]].astype(str).str.len() > 0)
        valid = work[p_col].notna() & work[h_col].notna() & nonempty_sequence & nonempty_smiles
        valid_frame = work.loc[valid].copy()
        sample_n = min(sample_size_per_topology, len(valid_frame))
        if sample_n == 0:
            raise ValueError(f"No valid rows available for {dataset['label']} from {path}")
        sample = valid_frame.sample(n=sample_n, random_state=seed + index).copy()
        sample.insert(0, "topology", dataset["topology"])
        sample.insert(1, "topology_label", dataset["label"])
        sample.insert(2, "source_path", str(path))
        sample["permeability"] = sample[p_col]
        sample["hemolysis"] = sample[h_col]
        sample["is_hit"] = (
            (sample["permeability"] > float(config["permeability_threshold"]))
            & (sample["hemolysis"] < float(config["hemolysis_threshold"]))
        )
        sample["sample_index"] = np.arange(1, len(sample) + 1)
        frames.append(sample)
        summary_rows.append({
            "topology": dataset["topology"],
            "topology_label": dataset["label"],
            "path": str(path),
            "raw_rows": len(raw),
            "valid_rows": len(valid_frame),
            "sampled_rows": len(sample),
            "hit_count": int(sample["is_hit"].sum()),
            "hit_denominator": len(sample),
        })

    return pd.concat(frames, ignore_index=True), pd.DataFrame(summary_rows)


def fingerprint_from_smiles(smiles: str) -> Any | None:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return MORGAN_GENERATOR.GetFingerprint(mol)
    except Exception:
        return None


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    fp_a = fingerprint_from_smiles(smiles_a)
    fp_b = fingerprint_from_smiles(smiles_b)
    if fp_a is None or fp_b is None:
        return float("nan")
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def representative_candidate_table(
    subset: pd.DataFrame,
    config: dict[str, Any],
    *,
    pool_size: int,
) -> pd.DataFrame:
    p_threshold = float(config["permeability_threshold"])
    h_threshold = float(config["hemolysis_threshold"])
    smiles_col = config["columns"]["smiles"]
    valid_smiles = subset[subset[smiles_col].astype(str).str.len() > 0].copy()
    hits = valid_smiles[valid_smiles["is_hit"]].copy()
    if not hits.empty:
        candidates = hits
        center = np.array([hits["permeability"].median(), hits["hemolysis"].median()], dtype=float)
        status = "diverse_hit"
    else:
        candidates = valid_smiles if not valid_smiles.empty else subset
        center = np.array([p_threshold, h_threshold], dtype=float)
        status = "no_hit_fallback"

    scale = np.array([
        max(float(subset["permeability"].std(ddof=0)), 1e-6),
        max(float(subset["hemolysis"].std(ddof=0)), 1e-6),
    ])
    coords = candidates[["permeability", "hemolysis"]].to_numpy(dtype=float)
    candidates = candidates.copy()
    candidates["selection_status"] = status
    candidates["selection_distance"] = np.sum(((coords - center) / scale) ** 2, axis=1)
    candidates = candidates.sort_values(["selection_distance", "sample_index"], ascending=[True, True])
    return candidates.head(min(pool_size, len(candidates))).reset_index(drop=True)


def select_representatives(sampled: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    strategy = str(config.get("representative_strategy", "hit_median"))
    pool_size = int(config.get("representative_candidate_pool", 200))
    smiles_col = config["columns"]["smiles"]
    candidates_by_topology: dict[str, pd.DataFrame] = {}
    for dataset in config["datasets"]:
        subset = sampled[sampled["topology"] == dataset["topology"]].copy()
        candidates_by_topology[dataset["topology"]] = representative_candidate_table(
            subset,
            config,
            pool_size=pool_size,
        )

    if strategy != "diverse_hit":
        representatives = [table.iloc[0].copy() for table in candidates_by_topology.values()]
        return pd.DataFrame(representatives).reset_index(drop=True)

    head_key = "head_to_tail"
    sc_key = "sc_to_tail"
    head_candidates = candidates_by_topology.get(head_key)
    sc_candidates = candidates_by_topology.get(sc_key)
    selected_indices: dict[str, int] = {}
    if head_candidates is not None and sc_candidates is not None and not head_candidates.empty and not sc_candidates.empty:
        head_weight = float(config.get("head_sc_dissimilarity_weight", 2.0))
        head_dist = max(float(head_candidates["selection_distance"].max()), 1e-6)
        sc_dist = max(float(sc_candidates["selection_distance"].max()), 1e-6)
        best_score = float("inf")
        best_pair = (0, 0)
        for head_idx, head_row in head_candidates.iterrows():
            for sc_idx, sc_row in sc_candidates.iterrows():
                similarity = tanimoto_similarity(str(head_row[smiles_col]), str(sc_row[smiles_col]))
                if not np.isfinite(similarity):
                    similarity = 1.0
                score = (
                    float(head_row["selection_distance"]) / head_dist
                    + float(sc_row["selection_distance"]) / sc_dist
                    + head_weight * similarity
                )
                if score < best_score:
                    best_score = score
                    best_pair = (int(head_idx), int(sc_idx))
        selected_indices[head_key] = best_pair[0]
        selected_indices[sc_key] = best_pair[1]

    representatives: list[pd.Series] = []
    for dataset in config["datasets"]:
        table = candidates_by_topology[dataset["topology"]]
        index = selected_indices.get(dataset["topology"], 0)
        representatives.append(table.iloc[index].copy())
    return pd.DataFrame(representatives).reset_index(drop=True)


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    rdDepictor.Compute2DCoords(mol)
    return mol


def write_molecule_svg(mol: Chem.Mol, path: Path, *, width: int = 420, height: int = 260) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    if hasattr(options, "clearBackground"):
        options.clearBackground = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")
    path.write_text(svg, encoding="utf-8")
    return svg


def svg_to_image(svg_path: Path, mol: Chem.Mol, *, size: tuple[int, int] = (420, 260)) -> tuple[Image.Image, str]:
    try:
        import cairosvg  # type: ignore
        png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=size[0], output_height=size[1])
        return Image.open(BytesIO(png_bytes)).convert("RGBA"), "cairosvg"
    except Exception:
        return Draw.MolToImage(mol, size=size).convert("RGBA"), "rdkit_pil_fallback"


def generate_molecule_assets(
    representatives: pd.DataFrame,
    config: dict[str, Any],
    assets_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Image.Image]]:
    rows: list[dict[str, Any]] = []
    images: dict[str, Image.Image] = {}
    smiles_col = config["columns"]["smiles"]
    for _, row in representatives.iterrows():
        topology = str(row["topology"])
        svg_path = assets_dir / f"representative_{topology}.svg"
        embed_status = ""
        error = ""
        try:
            mol = mol_from_smiles(str(row[smiles_col]))
            write_molecule_svg(mol, svg_path)
            image, embed_status = svg_to_image(svg_path, mol)
            images[topology] = image
            molecule_status = "success"
        except Exception as exc:
            molecule_status = "failed"
            error = str(exc)
        rows.append({
            "topology": topology,
            "topology_label": row["topology_label"],
            "molecule_svg_path": str(svg_path) if svg_path.exists() else "",
            "molecule_status": molecule_status,
            "embedding_status": embed_status,
            "molecule_error": error,
        })
    assets = pd.DataFrame(rows)
    representatives_for_merge = representatives.copy()
    if "molecule_svg_path" in representatives_for_merge.columns:
        representatives_for_merge = representatives_for_merge.rename(
            columns={"molecule_svg_path": "source_molecule_svg_path"}
        )
    merged = representatives_for_merge.merge(assets, on=["topology", "topology_label"], how="left")
    return merged, images


def compute_pairwise_similarity(representatives: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    smiles_col = config["columns"]["smiles"]
    rows: list[dict[str, Any]] = []
    for (_, row_a), (_, row_b) in combinations(representatives.iterrows(), 2):
        rows.append({
            "topology_a": row_a["topology"],
            "topology_label_a": row_a["topology_label"],
            "topology_b": row_b["topology"],
            "topology_label_b": row_b["topology_label"],
            "tanimoto_similarity": tanimoto_similarity(str(row_a[smiles_col]), str(row_b[smiles_col])),
        })
    return pd.DataFrame(rows)


def axis_limits(values: pd.Series, threshold: float, *, pad_fraction: float = 0.08) -> tuple[float, float]:
    v_min = float(values.min())
    v_max = float(values.max())
    span = max(v_max - v_min, 1e-6)
    lower = min(v_min - span * pad_fraction, threshold - span * 0.15)
    upper = max(v_max + span * pad_fraction, threshold + span * 0.15)
    return lower, upper


def plot_joint_distribution(
    sampled: pd.DataFrame,
    representatives: pd.DataFrame,
    images: dict[str, Image.Image],
    source_summary: pd.DataFrame,
    config: dict[str, Any],
    layout: dict[str, Path],
) -> list[str]:
    width = mm_to_inches(float(config.get("figure_width_mm", 180)))
    height = mm_to_inches(float(config.get("figure_height_mm", 95)))
    p_threshold = float(config["permeability_threshold"])
    h_threshold = float(config["hemolysis_threshold"])
    point_size = float(config.get("point_size", 10))
    point_alpha = float(config.get("point_alpha", 0.3))
    dpi = int(config.get("dpi", 600))
    formats = list(config.get("formats", ["pdf"]))
    embed_molecules = bool(config.get("embed_molecules", True))

    fig = plt.figure(figsize=(width, height))
    main_ax_rect = [0.08, 0.17, 0.53, 0.74] if embed_molecules else [0.08, 0.17, 0.86, 0.74]
    main_ax = fig.add_axes(main_ax_rect)
    x_min, x_max = axis_limits(sampled["permeability"], p_threshold)
    _, y_max = axis_limits(sampled["hemolysis"], h_threshold)
    y_min = float(config.get("hemolysis_axis_min", 0.0))
    main_ax.set_xlim(x_min, x_max)
    main_ax.set_ylim(y_min, y_max)

    ideal_width = max(x_max - p_threshold, 0)
    ideal_height = max(h_threshold - y_min, 0)
    if ideal_width > 0 and ideal_height > 0:
        main_ax.add_patch(Rectangle(
            (p_threshold, y_min),
            ideal_width,
            ideal_height,
            facecolor="#E8F4EA",
            edgecolor="#5D8A66",
            linewidth=0.8,
            alpha=0.35,
            zorder=0,
        ))
    main_ax.axvline(p_threshold, color="0.45", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    main_ax.axhline(h_threshold, color="0.45", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)

    for dataset in config["datasets"]:
        subset = sampled[sampled["topology"] == dataset["topology"]]
        main_ax.scatter(
            subset["permeability"],
            subset["hemolysis"],
            s=point_size,
            alpha=point_alpha,
            c=dataset["color"],
            marker=dataset["marker"],
            edgecolors="none",
            label=dataset["label"],
            zorder=2,
        )

    for _, row in representatives.iterrows():
        dataset = next(item for item in config["datasets"] if item["topology"] == row["topology"])
        main_ax.scatter(
            [row["permeability"]],
            [row["hemolysis"]],
            s=58,
            c=dataset["color"],
            marker=dataset["marker"],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

    main_ax.set_xlabel("Predicted permeability")
    main_ax.set_ylabel("Predicted hemolysis")
    main_ax.tick_params(labelsize=6)
    main_ax.legend(loc="lower left", frameon=False, handletextpad=0.3, borderpad=0.2, markerscale=1.4)
    main_ax.text(
        0.02,
        0.98,
        "\n".join(
            f"{row.topology_label}: {int(row.hit_count)}/{int(row.hit_denominator)}"
            for row in source_summary.itertuples(index=False)
        ),
        transform=main_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "round,pad=0.25", "alpha": 0.85},
    )
    if config.get("panel_label"):
        fig.text(0.012, 0.965, str(config["panel_label"]), fontsize=8, fontweight="bold", ha="left", va="top")

    if embed_molecules:
        molecule_axes = [
            fig.add_axes([0.68, 0.66, 0.23, 0.24]),
            fig.add_axes([0.68, 0.38, 0.23, 0.24]),
            fig.add_axes([0.68, 0.10, 0.23, 0.24]),
        ]
        y_centers = [0.78, 0.50, 0.22]
        rep_by_topology = {str(row["topology"]): row for _, row in representatives.iterrows()}
        for idx, dataset in enumerate(config["datasets"]):
            ax_img = molecule_axes[idx]
            row = rep_by_topology[dataset["topology"]]
            image = images.get(dataset["topology"])
            ax_img.set_axis_off()
            if image is not None:
                ax_img.imshow(image)
            else:
                ax_img.text(0.5, 0.5, "Molecule\nrendering failed", ha="center", va="center", fontsize=6)
            ax_img.set_title(
                f"{dataset['label']}\nP={row['permeability']:.2f}, H={row['hemolysis']:.2f}",
                fontsize=6,
                pad=2,
            )
            connection = ConnectionPatch(
                xyA=(row["permeability"], row["hemolysis"]),
                coordsA=main_ax.transData,
                xyB=(0.675, y_centers[idx]),
                coordsB=fig.transFigure,
                color=dataset["color"],
                linewidth=0.7,
                alpha=0.85,
                zorder=4,
            )
            fig.add_artist(connection)

    outputs: list[str] = []
    figure_prefix = layout["figures"] / DEFAULT_OUTPUT_NAME
    for fmt in formats:
        output = figure_prefix.with_suffix(f".{fmt}")
        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": float(config.get("pad_inches", 0.02)),
        }
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(output, **kwargs)
        outputs.append(str(output))
    plt.close(fig)
    return outputs


def write_run_record(
    path: Path,
    *,
    config_path: Path,
    command: str,
    config: dict[str, Any],
    outputs: Iterable[str],
    source_summary: pd.DataFrame,
) -> None:
    lines = [
        "# RL After Joint Property Distribution Record",
        "",
        f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Command: `{command}`",
        f"- Config: `{config_path}`",
        f"- Output directory: `{config['output_dir']}`",
        f"- Random seed: `{config.get('random_seed')}`",
        f"- Sample size per topology: `{config.get('sample_size_per_topology')}`",
        f"- DPI: `{config.get('dpi')}`",
        f"- Formats: `{', '.join(config.get('formats', []))}`",
        f"- Font family: `{config.get('font_family')}`",
        f"- Font path: `{config.get('font_path')}`",
        f"- Figure width mm: `{config.get('figure_width_mm')}`",
        f"- Figure height mm: `{config.get('figure_height_mm')}`",
        f"- Pad inches: `{config.get('pad_inches')}`",
        f"- Permeability threshold: `{config.get('permeability_threshold')}`",
        f"- Hemolysis threshold: `{config.get('hemolysis_threshold')}`",
        f"- Hemolysis axis minimum: `{config.get('hemolysis_axis_min', 0.0)}`",
        f"- Molecule image format: `{config.get('molecule_image_format')}`",
        f"- Embed molecules in main figure: `{config.get('embed_molecules', True)}`",
        f"- Representative strategy: `{config.get('representative_strategy')}`",
        f"- Representative candidate pool: `{config.get('representative_candidate_pool')}`",
        f"- Head/SC dissimilarity weight: `{config.get('head_sc_dissimilarity_weight')}`",
        "",
        "## Dataset Colors",
        "",
        "| topology | color | marker |",
        "|---|---|---|",
    ]
    for dataset in config["datasets"]:
        lines.append(f"| {dataset['label']} | `{dataset.get('color')}` | `{dataset.get('marker')}` |")
    lines.extend([
        "",
        "## Inputs",
        "",
        "| topology | raw_rows | valid_rows | sampled_rows | hit_count | path |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in source_summary.itertuples(index=False):
        lines.append(
            f"| {row.topology_label} | {row.raw_rows} | {row.valid_rows} | "
            f"{row.sampled_rows} | {row.hit_count} | `{row.path}` |"
        )
    lines.extend(["", "## Outputs", ""])
    for output in outputs:
        lines.append(f"- `{output}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_command() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.out_dir is not None:
        config["output_dir"] = str(args.out_dir)
    if args.sample_size_per_topology is not None:
        config["sample_size_per_topology"] = int(args.sample_size_per_topology)
    if args.formats is not None:
        config["formats"] = list(args.formats)
    out_dir = Path(str(config["output_dir"]))
    layout = prepare_layout(out_dir)
    apply_publication_style(config)

    sampled, source_summary = load_and_sample(
        config,
        sample_size_per_topology=int(config.get("sample_size_per_topology", 1000)),
    )
    representatives = select_representatives(sampled, config)
    representatives, molecule_images = generate_molecule_assets(representatives, config, layout["assets"])
    pairwise_similarity = compute_pairwise_similarity(representatives, config)

    sampled_path = layout["tables"] / "sampled_points.csv"
    reps_path = layout["tables"] / "representative_sequences.csv"
    summary_path = layout["tables"] / "source_summary.csv"
    similarity_path = layout["tables"] / "representative_pairwise_similarity.csv"
    sampled.to_csv(sampled_path, index=False)
    representatives.to_csv(reps_path, index=False)
    source_summary.to_csv(summary_path, index=False)
    pairwise_similarity.to_csv(similarity_path, index=False)

    figure_outputs = plot_joint_distribution(sampled, representatives, molecule_images, source_summary, config, layout)
    outputs = [*figure_outputs, str(sampled_path), str(reps_path), str(summary_path), str(similarity_path)]
    outputs.extend(str(path) for path in sorted(layout["assets"].glob("*.svg")))
    run_record = layout["root"] / "run_record.md"
    write_run_record(
        run_record,
        config_path=args.config,
        command=build_command(),
        config=config,
        outputs=outputs,
        source_summary=source_summary,
    )
    outputs.append(str(run_record))
    print(json.dumps({"output_dir": str(out_dir), "outputs": outputs, "run_record": str(run_record)}, indent=2))


if __name__ == "__main__":
    main()
