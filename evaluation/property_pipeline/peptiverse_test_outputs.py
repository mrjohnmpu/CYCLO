#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_peptiverse_root() -> Path:
    import os
    return Path(os.environ.get("PEPTIVERSE_ROOT", PROJECT_ROOT / "third_party" / "PeptiVerse")).expanduser().resolve()


PEPTIVERSE_ROOT = resolve_peptiverse_root()
THIRD_PARTY_ROOT = PEPTIVERSE_ROOT.parent
for entry in (PROJECT_ROOT, THIRD_PARTY_ROOT):
    entry_str = str(entry)
    if entry_str not in sys.path:
        sys.path.insert(0, entry_str)


DEFAULT_SEQUENCE = "GIVEQCCTSICSLYQLENYCN"
DEFAULT_SMILES = (
    "CC(C)C[C@@H]1NC(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H](C)N(C)C(=O)"
    "[C@H](Cc2ccccc2)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@H]2CCCN2C1=O"
)
DEFAULT_TARGET_SEQUENCE = (
    "TKLADVYQAELRELRLRLDQLTANSARLEVERDNLAQDLATVRQKLQDETNLRLEAENNLAAYRQEADEAT"
    "LARLDLERKIESLEEEIRFLRKIHEEEVRELQE"
)

PROPERTY_CASES = [
    {"case": "hemolysis_wt", "kind": "property", "property": "hemolysis", "mode": "wt", "input_key": "sequence"},
    {"case": "hemolysis_smiles", "kind": "property", "property": "hemolysis", "mode": "smiles", "input_key": "smiles"},
    {"case": "nf_wt", "kind": "property", "property": "nf", "mode": "wt", "input_key": "sequence"},
    {"case": "nf_smiles", "kind": "property", "property": "nf", "mode": "smiles", "input_key": "smiles"},
    {"case": "solubility_wt", "kind": "property", "property": "solubility", "mode": "wt", "input_key": "sequence"},
    {"case": "permeability_penetrance_wt", "kind": "property", "property": "permeability_penetrance", "mode": "wt", "input_key": "sequence"},
    {"case": "toxicity_smiles", "kind": "property", "property": "toxicity", "mode": "smiles", "input_key": "smiles"},
    {"case": "permeability_pampa_smiles", "kind": "property", "property": "permeability_pampa", "mode": "smiles", "input_key": "smiles"},
    {"case": "permeability_caco2_smiles", "kind": "property", "property": "permeability_caco2", "mode": "smiles", "input_key": "smiles"},
    {"case": "halflife_wt", "kind": "property", "property": "halflife", "mode": "wt", "input_key": "sequence"},
    {"case": "halflife_smiles", "kind": "property", "property": "halflife", "mode": "smiles", "input_key": "smiles"},
    {"case": "binding_affinity_wt", "kind": "binding", "mode": "wt", "binder_key": "sequence"},
    {"case": "binding_affinity_smiles", "kind": "binding", "mode": "smiles", "binder_key": "smiles"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test PeptiVerse and print the raw output for each supported property."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Inference device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--sequence", type=str, default=DEFAULT_SEQUENCE, help="Peptide sequence for wt-mode properties")
    parser.add_argument("--smiles", type=str, default=DEFAULT_SMILES, help="Peptide SMILES for smiles-mode properties")
    parser.add_argument(
        "--target-seq",
        type=str,
        default=DEFAULT_TARGET_SEQUENCE,
        help="Protein target sequence for binding affinity tests",
    )
    parser.add_argument("--manifest-path", type=Path, default=PEPTIVERSE_ROOT / "best_models.txt")
    parser.add_argument("--weight-root", type=Path, default=PEPTIVERSE_ROOT)
    parser.add_argument(
        "--cases",
        type=str,
        default=",".join(case["case"] for case in PROPERTY_CASES),
        help="Comma-separated subset of case names to run",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save all outputs as JSON")
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def select_cases(case_arg: str) -> list[dict[str, str]]:
    wanted = {name.strip() for name in case_arg.split(",") if name.strip()}
    selected = [case for case in PROPERTY_CASES if case["case"] in wanted]
    missing = sorted(wanted - {case["case"] for case in selected})
    if missing:
        raise ValueError(f"Unknown cases: {', '.join(missing)}")
    return selected


def run_case(predictor: Any, case: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    if case["kind"] == "property":
        input_str = args.sequence if case["input_key"] == "sequence" else args.smiles
        output = predictor.predict_property(case["property"], case["mode"], input_str)
        return {
            "case": case["case"],
            "kind": case["kind"],
            "property": case["property"],
            "mode": case["mode"],
            "input": input_str,
            "output": to_builtin(output),
        }

    binder_str = args.sequence if case["binder_key"] == "sequence" else args.smiles
    output = predictor.predict_binding_affinity(
        mode=case["mode"],
        target_seq=args.target_seq,
        binder_str=binder_str,
    )
    return {
        "case": case["case"],
        "kind": case["kind"],
        "property": "binding_affinity",
        "mode": case["mode"],
        "target_seq": args.target_seq,
        "binder": binder_str,
        "output": to_builtin(output),
    }


def main() -> None:
    args = parse_args()
    selected_cases = select_cases(args.cases)
    from PeptiVerse.inference import PeptiVersePredictor

    predictor = PeptiVersePredictor(
        manifest_path=args.manifest_path,
        classifier_weight_root=args.weight_root,
        device=args.device,
    )

    results: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "peptiverse_root": str(PEPTIVERSE_ROOT),
        "manifest_path": str(args.manifest_path),
        "weight_root": str(args.weight_root),
        "device": args.device,
        "results": [],
    }

    for case in selected_cases:
        print(f"===== {case['case']} =====", flush=True)
        try:
            record = run_case(predictor, case, args)
            record["status"] = "ok"
            print(json.dumps(record["output"], ensure_ascii=False, indent=2), flush=True)
        except Exception as exc:
            record = {
                "case": case["case"],
                "kind": case["kind"],
                "property": case.get("property", "binding_affinity"),
                "mode": case["mode"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(
                json.dumps(
                    {"error_type": type(exc).__name__, "error": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
        results["results"].append(record)
        print(flush=True)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON results to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
