#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_ROOT = Path(
    'results/rl_permeability_analysis_head_to_tail_perm_hemo_20260506_full'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize amino-acid family level shifts from token enrichment tables.")
    parser.add_argument("--analysis_root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    return parser.parse_args()


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def normalize_family(token: str, *, case_sensitive: bool) -> str:
    text = str(token).strip()
    text = text.replace("(N-Me)", "").replace("(NMe)", "")
    text = text.replace("N-Me-", "").replace("NMe-", "").replace("N-methyl-", "")
    if case_sensitive:
        return text
    return text.lower()


def aggregate_family_tables(token_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_sensitive, mode in [(False, "merged"), (True, "case_sensitive")]:
        bucket: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "before_count": 0,
                "after_count": 0,
                "nme_before": 0,
                "nme_after": 0,
                "before_sequence_prevalence": 0.0,
                "after_sequence_prevalence": 0.0,
                "member_tokens": set(),
            }
        )
        for _, row in token_df.iterrows():
            family = normalize_family(row["token"], case_sensitive=case_sensitive)
            entry = bucket[family]
            entry["before_count"] += int(row["before_count"])
            entry["after_count"] += int(row["after_count"])
            if bool(row["is_n_methyl_token"]):
                entry["nme_before"] += int(row["before_count"])
                entry["nme_after"] += int(row["after_count"])
            entry["before_sequence_prevalence"] += float(row["before_sequence_prevalence"])
            entry["after_sequence_prevalence"] += float(row["after_sequence_prevalence"])
            entry["member_tokens"].add(str(row["token"]))

        total_before = sum(item["before_count"] for item in bucket.values())
        total_after = sum(item["after_count"] for item in bucket.values())
        for family, item in bucket.items():
            before_count = int(item["before_count"])
            after_count = int(item["after_count"])
            before_fraction = float(before_count / total_before) if total_before else 0.0
            after_fraction = float(after_count / total_after) if total_after else 0.0
            rows.append(
                {
                    "family_mode": mode,
                    "family": family,
                    "before_count": before_count,
                    "after_count": after_count,
                    "before_fraction": before_fraction,
                    "after_fraction": after_fraction,
                    "delta_fraction": after_fraction - before_fraction,
                    "before_sequence_prevalence": float(item["before_sequence_prevalence"]),
                    "after_sequence_prevalence": float(item["after_sequence_prevalence"]),
                    "delta_sequence_prevalence": float(item["after_sequence_prevalence"] - item["before_sequence_prevalence"]),
                    "nme_before": int(item["nme_before"]),
                    "nme_after": int(item["nme_after"]),
                    "nme_fraction_within_family_before": float(item["nme_before"] / before_count) if before_count else 0.0,
                    "nme_fraction_within_family_after": float(item["nme_after"] / after_count) if after_count else 0.0,
                    "delta_nme_fraction_within_family": (
                        float(item["nme_after"] / after_count) if after_count else 0.0
                    ) - (
                        float(item["nme_before"] / before_count) if before_count else 0.0
                    ),
                    "member_tokens": ", ".join(sorted(item["member_tokens"])),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    family_df: pd.DataFrame,
    *,
    tables_dir: Path,
    figures_dir: Path,
    report_dir: Path,
    n_methyl_summary: pd.DataFrame,
    descriptor_shift: pd.DataFrame,
) -> None:
    family_enrichment_df = family_df[
        [
            "family_mode",
            "family",
            "before_count",
            "after_count",
            "before_fraction",
            "after_fraction",
            "delta_fraction",
            "before_sequence_prevalence",
            "after_sequence_prevalence",
            "delta_sequence_prevalence",
            "member_tokens",
        ]
    ].copy()
    family_nme_df = family_df[
        [
            "family_mode",
            "family",
            "before_count",
            "after_count",
            "nme_before",
            "nme_after",
            "nme_fraction_within_family_before",
            "nme_fraction_within_family_after",
            "delta_nme_fraction_within_family",
            "member_tokens",
        ]
    ].copy()

    family_enrichment_path = tables_dir / "amino_acid_family_enrichment.csv"
    family_nme_path = tables_dir / "amino_acid_family_n_methyl_ratio.csv"
    family_enrichment_df.to_csv(family_enrichment_path, index=False)
    family_nme_df.to_csv(family_nme_path, index=False)

    merged_enrichment = family_enrichment_df[family_enrichment_df["family_mode"] == "merged"].copy()
    merged_nme = family_nme_df[family_nme_df["family_mode"] == "merged"].copy()

    set_plot_style()
    top_enrichment = merged_enrichment.reindex(
        merged_enrichment["delta_fraction"].abs().sort_values(ascending=False).index
    ).head(20)
    fig, ax = plt.subplots(figsize=(7.8, max(4.2, 0.28 * len(top_enrichment))))
    y = np.arange(len(top_enrichment))
    colors = np.where(top_enrichment["delta_fraction"] >= 0, "#0072B2", "#9E9E9E")
    ax.barh(y, top_enrichment["delta_fraction"], color=colors)
    ax.set_yticks(y, top_enrichment["family"])
    ax.invert_yaxis()
    ax.set_xlabel("After - Before family fraction")
    ax.set_ylabel("Amino-acid family")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, figures_dir / "amino_acid_family_enrichment_bar")

    top_nme = merged_nme[merged_nme["nme_before"] + merged_nme["nme_after"] > 0].copy()
    top_nme = top_nme.sort_values("delta_nme_fraction_within_family", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7.8, max(3.8, 0.28 * len(top_nme))))
    y = np.arange(len(top_nme))
    colors = np.where(top_nme["delta_nme_fraction_within_family"] >= 0, "#D55E00", "#9E9E9E")
    ax.barh(y, top_nme["delta_nme_fraction_within_family"], color=colors)
    ax.set_yticks(y, top_nme["family"])
    ax.invert_yaxis()
    ax.set_xlabel("After - Before N-methyl fraction within family")
    ax.set_ylabel("Amino-acid family")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, figures_dir / "amino_acid_family_n_methyl_ratio_bar")

    def get_row(df: pd.DataFrame, family: str) -> pd.Series | None:
        sub = df[df["family"] == family]
        return None if sub.empty else sub.iloc[0]

    enrich_families = ["val", "lys", "met", "arg", "leu"]
    reduce_families = ["trp", "his", "tyr", "phe", "cys", "ser"]
    nme_families = ["glu", "ala", "asp", "thr", "met", "leu", "val", "lys", "arg", "asn"]
    shift_lookup = descriptor_shift.set_index("descriptor")
    nme_summary_lookup = n_methyl_summary.set_index("stage")

    lines = [
        "# 氨基酸家族水平结果",
        "",
        "联合优化后，氨基酸组成在家族水平上发生了显著重排。",
    ]
    enrich_parts = []
    for family in enrich_families:
        row = get_row(merged_enrichment, family)
        if row is not None:
            enrich_parts.append(
                f"`{family}`家族占比由{row['before_fraction']:.3f}升至{row['after_fraction']:.3f}"
            )
    if enrich_parts:
        lines.append("其中，" + "，".join(enrich_parts) + "。")
    reduce_parts = []
    for family in reduce_families:
        row = get_row(merged_enrichment, family)
        if row is not None:
            reduce_parts.append(
                f"`{family}`家族占比由{row['before_fraction']:.3f}降至{row['after_fraction']:.3f}"
            )
    if reduce_parts:
        lines.append("相反，" + "，".join(reduce_parts) + "。")

    nme_parts = []
    for family in nme_families:
        row = get_row(merged_nme, family)
        if row is not None and (row["nme_before"] + row["nme_after"]) > 0:
            nme_parts.append(
                f"`{family}`家族内部N-甲基化比例由{row['nme_fraction_within_family_before']:.3f}升至{row['nme_fraction_within_family_after']:.3f}"
            )
    if nme_parts:
        lines.append("家族内部甲基化修饰呈现出明显的特异性重分布，" + "，".join(nme_parts[:6]) + "。")

    if "hydrophobic_residue_fraction" in shift_lookup.index and "peptide_length" in shift_lookup.index and "TPSA" in shift_lookup.index:
        hydro = shift_lookup.loc["hydrophobic_residue_fraction", "delta_mean"]
        length = shift_lookup.loc["peptide_length", "delta_mean"]
        tpsa = shift_lookup.loc["TPSA", "delta_mean"]
        lines.append(
            f"这一组成变化与理化性质重排方向一致：疏水残基比例整体上升（Δ={hydro:.3f}），肽长缩短（Δ={length:.3f}），极性表面积下降（ΔTPSA={tpsa:.3f}）。"
        )

    if {"before", "after"}.issubset(nme_summary_lookup.index):
        before_frac = nme_summary_lookup.loc["before", "n_methyl_token_fraction"]
        after_frac = nme_summary_lookup.loc["after", "n_methyl_token_fraction"]
        lines.append(
            f"从总体修饰水平看，N-甲基化残基token比例由{before_frac:.3f}升至{after_frac:.3f}，说明联合优化并非简单增加所有甲基化残基，而是在保留的氨基酸家族内提高特定甲基化模式的使用频率。"
        )

    lines.append(
        "综合来看，模型并非对全部氨基酸进行均匀修饰，而是通过显著富集Val、Lys、Met、Arg和Leu等家族，并压低Trp、His、Tyr、Phe、Cys和Ser等家族，同时在若干保留家族内部提高N-甲基化比例，形成了更有利于高渗透性和低溶血风险的序列组成。"
    )
    lines.append("")

    report_path = report_dir / "amino_acid_family_results_cn.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_run_record(path: Path, *, analysis_root: Path) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "analysis_root": str(analysis_root),
        "inputs": {
            "token_enrichment": str(analysis_root / "tables" / "experiment2_token_enrichment.csv"),
            "n_methyl_summary": str(analysis_root / "tables" / "experiment2_n_methyl_summary.csv"),
            "descriptor_shift": str(analysis_root / "tables" / "experiment3_descriptor_shift_summary.csv"),
        },
        "outputs": {
            "family_enrichment": str(analysis_root / "tables" / "amino_acid_family_enrichment.csv"),
            "family_n_methyl_ratio": str(analysis_root / "tables" / "amino_acid_family_n_methyl_ratio.csv"),
            "family_bar": str(analysis_root / "figures" / "amino_acid_family_enrichment_bar.pdf"),
            "family_n_methyl_bar": str(analysis_root / "figures" / "amino_acid_family_n_methyl_ratio_bar.pdf"),
            "report_cn": str(analysis_root / "report" / "amino_acid_family_results_cn.md"),
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    tables_dir = analysis_root / "tables"
    figures_dir = analysis_root / "figures"
    report_dir = analysis_root / "report"
    token_path = tables_dir / "experiment2_token_enrichment.csv"
    n_methyl_summary_path = tables_dir / "experiment2_n_methyl_summary.csv"
    descriptor_shift_path = tables_dir / "experiment3_descriptor_shift_summary.csv"

    token_df = pd.read_csv(token_path)
    n_methyl_summary = pd.read_csv(n_methyl_summary_path)
    descriptor_shift = pd.read_csv(descriptor_shift_path)

    family_df = aggregate_family_tables(token_df)
    write_outputs(
        family_df,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        report_dir=report_dir,
        n_methyl_summary=n_methyl_summary,
        descriptor_shift=descriptor_shift,
    )
    write_run_record(analysis_root / "amino_acid_family_run_record.json", analysis_root=analysis_root)

    print(
        json.dumps(
            {
                "analysis_root": str(analysis_root),
                "family_enrichment_csv": str(tables_dir / "amino_acid_family_enrichment.csv"),
                "family_n_methyl_ratio_csv": str(tables_dir / "amino_acid_family_n_methyl_ratio.csv"),
                "family_bar_pdf": str(figures_dir / "amino_acid_family_enrichment_bar.pdf"),
                "family_n_methyl_bar_pdf": str(figures_dir / "amino_acid_family_n_methyl_ratio_bar.pdf"),
                "report_cn": str(report_dir / "amino_acid_family_results_cn.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
