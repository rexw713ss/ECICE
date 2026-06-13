#!/usr/bin/env python3
"""Generate and evaluate OCR ablation-study settings."""

import argparse
import csv
import json
from pathlib import Path

from cer_evaluation import evaluate_pair, format_percent, normalize_text
from paddleocr_baseline import create_baseline_engine, run_baseline
from pipeline_paths import (
    ABLATION_OCR_DIR,
    BASELINE_OCR_DIR,
    ENSEMBLE_OCR_DIR,
    EVALUATION_DIR,
    GROUND_TRUTH_DIR,
    LLM_CORRECTION_DIR,
    PREPROCESS_DIR,
)


SINGLE_VARIANT_SETTINGS = (
    {
        "id": "document_rectification",
        "parent_id": "raw_paddleocr",
        "variant": "document",
        "label": "+ document rectification",
        "description": "只加文件矯正，使用單一 document variant",
    },
    {
        "id": "blue_ink_extraction",
        "parent_id": "raw_paddleocr",
        "variant": "blue_ink",
        "label": "+ blue ink extraction",
        "description": "單一 blue-ink variant（含上游矯正、光照正規化與銳化）",
    },
    {
        "id": "line_removal",
        "parent_id": "raw_paddleocr",
        "variant": "line_removed",
        "label": "+ line removal",
        "description": "單一 line-removed variant（含上游矯正、光照正規化與銳化）",
    },
)

ABLATION_SETTINGS = (
    {
        "id": "raw_paddleocr",
        "parent_id": None,
        "label": "Raw image + PaddleOCR",
        "description": "baseline",
    },
    *SINGLE_VARIANT_SETTINGS,
    {
        "id": "multi_variant_ensemble",
        "parent_id": "raw_paddleocr",
        "label": "+ multi-variant ensemble",
        "description": "看 ensemble 是否有效",
    },
    {
        "id": "rule_based_normalization",
        "parent_id": "multi_variant_ensemble",
        "label": "+ rule-based normalization",
        "description": "只套用繁簡轉換、Unicode NFC、標點空白與 OCR 噪聲規則",
    },
    {
        "id": "llm_correction",
        "parent_id": "rule_based_normalization",
        "label": "+ LLM correction",
        "description": "在 rule-based normalization 後加入 grounded LLM",
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate or evaluate OCR ablation settings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Run single-pass PaddleOCR on selected preprocessing variants.",
    )
    generate.add_argument("--image", required=True, help="Original input image.")
    generate.add_argument(
        "--manifest",
        default=None,
        help="Preprocessing manifest. Inferred from --preprocess-dir by default.",
    )
    generate.add_argument("--preprocess-dir", default=str(PREPROCESS_DIR))
    generate.add_argument("--output-dir", default=str(ABLATION_OCR_DIR))
    generate.add_argument("--lang", default="ch")
    generate.add_argument("--device", default="cpu")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Calculate CER for every ablation setting.",
    )
    evaluate.add_argument("--ground-truth-dir", default=str(GROUND_TRUTH_DIR))
    evaluate.add_argument("--baseline-dir", default=str(BASELINE_OCR_DIR))
    evaluate.add_argument("--ablation-dir", default=str(ABLATION_OCR_DIR))
    evaluate.add_argument("--ensemble-dir", default=str(ENSEMBLE_OCR_DIR))
    evaluate.add_argument("--llm-dir", default=str(LLM_CORRECTION_DIR))
    evaluate.add_argument("--output-dir", default=str(EVALUATION_DIR))
    evaluate.add_argument("--stem", action="append", dest="stems")
    evaluate.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def load_manifest(image_path, manifest_path=None, preprocess_dir=PREPROCESS_DIR):
    image_path = Path(image_path)
    manifest_path = (
        Path(manifest_path)
        if manifest_path
        else Path(preprocess_dir) / f"{image_path.stem}_preprocess_manifest.json"
    )
    data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    return manifest_path, data


def resolve_variant_path(manifest_path, raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = (
        Path.cwd() / path,
        manifest_path.parent / path,
        manifest_path.parent / path.name,
    )
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), path)


def generate_single_variant_predictions(
    image_path,
    manifest_path=None,
    preprocess_dir=PREPROCESS_DIR,
    output_dir=ABLATION_OCR_DIR,
    *,
    lang="ch",
    device="cpu",
):
    image_path = Path(image_path)
    manifest_path, manifest = load_manifest(image_path, manifest_path, preprocess_dir)
    variants = manifest.get("variants", {})
    missing = [
        setting["variant"]
        for setting in SINGLE_VARIANT_SETTINGS
        if setting["variant"] not in variants
    ]
    if missing:
        raise ValueError(f"Preprocessing manifest is missing variants: {', '.join(missing)}")

    engine = create_baseline_engine(lang, device)
    outputs = []
    for setting in SINGLE_VARIANT_SETTINGS:
        variant_path = resolve_variant_path(manifest_path, variants[setting["variant"]])
        if not variant_path.is_file():
            raise ValueError(f"Ablation variant image not found: {variant_path}")
        text_path, json_path = run_baseline(
            engine,
            variant_path,
            output_dir,
            lang=lang,
            device=device,
            output_stem=image_path.stem,
            output_suffix=f"_{setting['id']}",
            method="single_pass_paddleocr_preprocessing_ablation",
            setting={
                "id": setting["id"],
                "label": setting["label"],
                "variant": setting["variant"],
                "description": setting["description"],
            },
        )
        outputs.extend((text_path, json_path))
    return outputs


def prediction_paths(stem, baseline_dir, ablation_dir, ensemble_dir, llm_dir):
    return {
        "raw_paddleocr": Path(baseline_dir) / f"{stem}_paddleocr_baseline.txt",
        "document_rectification": Path(ablation_dir) / f"{stem}_document_rectification.txt",
        "blue_ink_extraction": Path(ablation_dir) / f"{stem}_blue_ink_extraction.txt",
        "line_removal": Path(ablation_dir) / f"{stem}_line_removal.txt",
        "multi_variant_ensemble": Path(ensemble_dir) / f"{stem}_merged_ocr.txt",
        "rule_based_normalization": Path(llm_dir) / f"{stem}_rule_based.txt",
        "llm_correction": Path(llm_dir) / f"{stem}_corrected.txt",
    }


def aggregate_results(documents, profile, setting_id):
    results = [
        document["profiles"][profile][setting_id]
        for document in documents
        if setting_id in document["profiles"][profile]
    ]
    reference_characters = sum(result["reference_characters"] for result in results)
    edits = sum(result["edits"] for result in results)
    cer = edits / reference_characters if reference_characters else None
    return {
        "document_count": len(results),
        "reference_characters": reference_characters,
        "edits": edits,
        "cer": cer,
        "character_accuracy": max(0.0, 1.0 - cer) if cer is not None else None,
    }


def evaluate_ablation(
    ground_truth_dir,
    baseline_dir,
    ablation_dir,
    ensemble_dir,
    llm_dir,
    *,
    stems=None,
    allow_missing=False,
):
    ground_truth_paths = sorted(Path(ground_truth_dir).glob("*.txt"))
    if stems:
        requested = set(stems)
        ground_truth_paths = [path for path in ground_truth_paths if path.stem in requested]
    if not ground_truth_paths:
        raise ValueError(f"No ground-truth .txt files found in {ground_truth_dir}.")

    documents = []
    missing = []
    for ground_truth_path in ground_truth_paths:
        stem = ground_truth_path.stem
        reference = ground_truth_path.read_text(encoding="utf-8-sig")
        if not normalize_text(reference, remove_whitespace=True):
            raise ValueError(f"Ground truth is empty after normalization: {ground_truth_path}")
        paths = prediction_paths(stem, baseline_dir, ablation_dir, ensemble_dir, llm_dir)
        document = {
            "stem": stem,
            "ground_truth_path": str(ground_truth_path),
            "prediction_paths": {},
            "profiles": {"strict": {}, "no_whitespace": {}},
        }
        for setting in ABLATION_SETTINGS:
            setting_id = setting["id"]
            path = paths[setting_id]
            if not path.is_file():
                missing.append({"stem": stem, "setting": setting_id, "path": str(path)})
                continue
            hypothesis = path.read_text(encoding="utf-8-sig")
            document["prediction_paths"][setting_id] = str(path)
            for profile, remove_whitespace in (("strict", False), ("no_whitespace", True)):
                document["profiles"][profile][setting_id] = evaluate_pair(
                    reference,
                    hypothesis,
                    remove_whitespace=remove_whitespace,
                )
        documents.append(document)

    if missing and not allow_missing:
        paths = "\n".join(f"- {item['path']}" for item in missing)
        raise ValueError(f"Ablation predictions are missing:\n{paths}")

    aggregate = {}
    for profile in ("strict", "no_whitespace"):
        aggregate[profile] = {}
        baseline_cer = None
        for setting in ABLATION_SETTINGS:
            setting_id = setting["id"]
            result = aggregate_results(documents, profile, setting_id)
            if setting_id == "raw_paddleocr":
                baseline_cer = result["cer"]
            result["delta_cer_vs_baseline"] = (
                result["cer"] - baseline_cer
                if result["cer"] is not None and baseline_cer is not None
                else None
            )
            parent_id = setting.get("parent_id")
            parent_cer = (
                aggregate[profile].get(parent_id, {}).get("cer")
                if parent_id
                else None
            )
            result["delta_cer_vs_parent"] = (
                result["cer"] - parent_cer
                if result["cer"] is not None and parent_cer is not None
                else None
            )
            aggregate[profile][setting_id] = result

    return {
        "metric": "CER = edits / reference characters; lower is better",
        "design": (
            "Module-level ablation: each preprocessing row independently replaces the "
            "raw image with one pipeline-produced variant and uses the same single-pass "
            "PaddleOCR baseline configuration. Single-variant rows are not cumulative; "
            "ensemble and correction-stage rows use their pipeline outputs."
        ),
        "settings": [dict(setting) for setting in ABLATION_SETTINGS],
        "missing_predictions": missing,
        "aggregate": aggregate,
        "documents": documents,
    }


def write_markdown(report, path):
    lines = [
        "# Ablation Study",
        "",
        report["design"],
        "",
    ]
    for profile in ("strict", "no_whitespace"):
        lines.extend(
            [
                f"## {profile.replace('_', ' ').title()}",
                "",
                "| Setting | CER ↓ | Accuracy ↑ | Δ CER vs baseline | Δ CER vs parent | 說明 |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for setting in report["settings"]:
            result = report["aggregate"][profile][setting["id"]]
            lines.append(
                f"| {setting['label']} | {format_percent(result['cer'])} | "
                f"{format_percent(result['character_accuracy'])} | "
                f"{format_percent(result['delta_cer_vs_baseline'])} | "
                f"{format_percent(result['delta_cer_vs_parent'])} | "
                f"{setting['description']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(report, path):
    fieldnames = [
        "profile",
        "setting",
        "setting_id",
        "cer",
        "cer_percent",
        "character_accuracy",
        "character_accuracy_percent",
        "delta_cer_vs_baseline",
        "delta_cer_vs_parent",
        "description",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for profile in ("strict", "no_whitespace"):
            for setting in report["settings"]:
                result = report["aggregate"][profile][setting["id"]]
                writer.writerow(
                    {
                        "profile": profile,
                        "setting": setting["label"],
                        "setting_id": setting["id"],
                        "cer": result["cer"],
                        "cer_percent": (
                            result["cer"] * 100 if result["cer"] is not None else None
                        ),
                        "character_accuracy": result["character_accuracy"],
                        "character_accuracy_percent": (
                            result["character_accuracy"] * 100
                            if result["character_accuracy"] is not None
                            else None
                        ),
                        "delta_cer_vs_baseline": result["delta_cer_vs_baseline"],
                        "delta_cer_vs_parent": result["delta_cer_vs_parent"],
                        "description": setting["description"],
                    }
                )


def save_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ablation_report.json"
    csv_path = output_dir / "ablation_report.csv"
    markdown_path = output_dir / "ablation_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(report, csv_path)
    write_markdown(report, markdown_path)
    return json_path, csv_path, markdown_path


def main():
    args = parse_args()
    try:
        if args.command == "generate":
            paths = generate_single_variant_predictions(
                args.image,
                args.manifest,
                args.preprocess_dir,
                args.output_dir,
                lang=args.lang,
                device=args.device,
            )
        else:
            report = evaluate_ablation(
                args.ground_truth_dir,
                args.baseline_dir,
                args.ablation_dir,
                args.ensemble_dir,
                args.llm_dir,
                stems=args.stems,
                allow_missing=args.allow_missing,
            )
            paths = save_report(report, args.output_dir)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Ablation study failed: {exc}") from exc

    print("Ablation study complete.")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
