#!/usr/bin/env python3
"""Compare OCR methods with character error rate (CER)."""

import argparse
import csv
import json
import math
import re
import unicodedata
from pathlib import Path

from pipeline_paths import (
    BASELINE_OCR_DIR,
    ENSEMBLE_OCR_DIR,
    EVALUATION_DIR,
    GROUND_TRUTH_DIR,
    LLM_CORRECTION_DIR,
)


DEFAULT_METHOD_SUFFIXES = {
    "paddleocr_baseline": "_paddleocr_baseline.txt",
    "ensemble_only": "_merged_ocr.txt",
    "ensemble_llm": "_corrected.txt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate strict and whitespace-insensitive CER for PaddleOCR baseline, "
            "ensemble-only OCR, and ensemble + LLM correction."
        )
    )
    parser.add_argument(
        "--ground-truth-dir",
        default=str(GROUND_TRUTH_DIR),
        help="Directory containing <image_stem>.txt human transcriptions.",
    )
    parser.add_argument(
        "--baseline-dir",
        default=str(BASELINE_OCR_DIR),
        help="Directory containing *_paddleocr_baseline.txt.",
    )
    parser.add_argument(
        "--ensemble-dir",
        default=str(ENSEMBLE_OCR_DIR),
        help="Directory containing *_merged_ocr.txt.",
    )
    parser.add_argument(
        "--llm-dir",
        default=str(LLM_CORRECTION_DIR),
        help="Directory containing *_corrected.txt.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EVALUATION_DIR),
        help="Directory for JSON, CSV, and Markdown reports.",
    )
    parser.add_argument(
        "--stem",
        action="append",
        dest="stems",
        help="Evaluate only this image stem. May be provided multiple times.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Evaluate available files and report missing predictions instead of failing.",
    )
    return parser.parse_args()


def normalize_text(text, *, remove_whitespace):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    if remove_whitespace:
        text = re.sub(r"\s+", "", text)
    return text


def edit_statistics(reference, hypothesis):
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distances = [[0] * columns for _ in range(rows)]

    for index in range(rows):
        distances[index][0] = index
    for index in range(columns):
        distances[0][index] = index

    for ref_index in range(1, rows):
        for hyp_index in range(1, columns):
            substitution_cost = 0 if reference[ref_index - 1] == hypothesis[hyp_index - 1] else 1
            distances[ref_index][hyp_index] = min(
                distances[ref_index - 1][hyp_index] + 1,
                distances[ref_index][hyp_index - 1] + 1,
                distances[ref_index - 1][hyp_index - 1] + substitution_cost,
            )

    substitutions = 0
    deletions = 0
    insertions = 0
    ref_index = len(reference)
    hyp_index = len(hypothesis)
    while ref_index > 0 or hyp_index > 0:
        if (
            ref_index > 0
            and hyp_index > 0
            and reference[ref_index - 1] == hypothesis[hyp_index - 1]
            and distances[ref_index][hyp_index] == distances[ref_index - 1][hyp_index - 1]
        ):
            ref_index -= 1
            hyp_index -= 1
        elif (
            ref_index > 0
            and hyp_index > 0
            and distances[ref_index][hyp_index] == distances[ref_index - 1][hyp_index - 1] + 1
        ):
            substitutions += 1
            ref_index -= 1
            hyp_index -= 1
        elif ref_index > 0 and distances[ref_index][hyp_index] == distances[ref_index - 1][hyp_index] + 1:
            deletions += 1
            ref_index -= 1
        else:
            insertions += 1
            hyp_index -= 1

    edits = substitutions + deletions + insertions
    cer = edits / len(reference) if reference else (0.0 if not hypothesis else math.inf)
    return {
        "reference_characters": len(reference),
        "hypothesis_characters": len(hypothesis),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "edits": edits,
        "cer": cer,
        "character_accuracy": max(0.0, 1.0 - cer),
    }


def evaluate_pair(reference_text, hypothesis_text, *, remove_whitespace):
    reference = normalize_text(reference_text, remove_whitespace=remove_whitespace)
    hypothesis = normalize_text(hypothesis_text, remove_whitespace=remove_whitespace)
    return edit_statistics(reference, hypothesis)


def prediction_paths(stem, baseline_dir, ensemble_dir, llm_dir):
    return {
        "paddleocr_baseline": baseline_dir / f"{stem}{DEFAULT_METHOD_SUFFIXES['paddleocr_baseline']}",
        "ensemble_only": ensemble_dir / f"{stem}{DEFAULT_METHOD_SUFFIXES['ensemble_only']}",
        "ensemble_llm": llm_dir / f"{stem}{DEFAULT_METHOD_SUFFIXES['ensemble_llm']}",
    }


def aggregate_results(per_document, profile, method):
    results = [
        document["profiles"][profile][method]
        for document in per_document
        if method in document["profiles"][profile]
    ]
    reference_characters = sum(item["reference_characters"] for item in results)
    edits = sum(item["edits"] for item in results)
    micro_cer = edits / reference_characters if reference_characters else None
    macro_cer = sum(item["cer"] for item in results) / len(results) if results else None
    return {
        "document_count": len(results),
        "reference_characters": reference_characters,
        "hypothesis_characters": sum(item["hypothesis_characters"] for item in results),
        "substitutions": sum(item["substitutions"] for item in results),
        "deletions": sum(item["deletions"] for item in results),
        "insertions": sum(item["insertions"] for item in results),
        "edits": edits,
        "micro_cer": micro_cer,
        "micro_character_accuracy": max(0.0, 1.0 - micro_cer) if micro_cer is not None else None,
        "macro_cer": macro_cer,
        "macro_character_accuracy": (
            sum(item["character_accuracy"] for item in results) / len(results)
            if results
            else None
        ),
    }


def improvement_statistics(baseline, proposed):
    if not baseline or not proposed:
        return None
    baseline_cer = baseline["micro_cer"]
    proposed_cer = proposed["micro_cer"]
    if baseline_cer is None or proposed_cer is None:
        return None
    return {
        "absolute_cer_reduction": baseline_cer - proposed_cer,
        "relative_error_reduction": (
            (baseline_cer - proposed_cer) / baseline_cer if baseline_cer else None
        ),
    }


def evaluate_directories(
    ground_truth_dir,
    baseline_dir,
    ensemble_dir,
    llm_dir=None,
    *,
    stems=None,
    allow_missing=False,
):
    ground_truth_dir = Path(ground_truth_dir)
    baseline_dir = Path(baseline_dir)
    ensemble_dir = Path(ensemble_dir)
    llm_dir = Path(llm_dir) if llm_dir is not None else ensemble_dir
    ground_truth_paths = sorted(ground_truth_dir.glob("*.txt"))
    if stems:
        requested_stems = set(stems)
        ground_truth_paths = [
            path for path in ground_truth_paths if path.stem in requested_stems
        ]
    if not ground_truth_paths:
        raise ValueError(
            f"No ground-truth .txt files found in {ground_truth_dir}. "
            "Use one human transcription per image, named <image_stem>.txt."
        )

    per_document = []
    missing = []
    for ground_truth_path in ground_truth_paths:
        stem = ground_truth_path.stem
        reference_text = ground_truth_path.read_text(encoding="utf-8-sig")
        if not normalize_text(reference_text, remove_whitespace=True):
            raise ValueError(f"Ground truth is empty after normalization: {ground_truth_path}")
        methods = prediction_paths(stem, baseline_dir, ensemble_dir, llm_dir)
        document = {
            "stem": stem,
            "ground_truth_path": str(ground_truth_path),
            "prediction_paths": {},
            "profiles": {"strict": {}, "no_whitespace": {}},
        }
        for method, prediction_path in methods.items():
            if not prediction_path.is_file():
                missing.append({"stem": stem, "method": method, "path": str(prediction_path)})
                continue
            hypothesis_text = prediction_path.read_text(encoding="utf-8-sig")
            document["prediction_paths"][method] = str(prediction_path)
            document["profiles"]["strict"][method] = evaluate_pair(
                reference_text,
                hypothesis_text,
                remove_whitespace=False,
            )
            document["profiles"]["no_whitespace"][method] = evaluate_pair(
                reference_text,
                hypothesis_text,
                remove_whitespace=True,
            )
        per_document.append(document)

    required_missing = [
        item for item in missing if item["method"] in {"paddleocr_baseline", "ensemble_llm"}
    ]
    if required_missing and not allow_missing:
        formatted = "\n".join(f"- {item['path']}" for item in required_missing)
        raise ValueError(
            "Required baseline or ensemble + LLM predictions are missing:\n"
            f"{formatted}\nRun with --allow-missing only for an intentionally partial evaluation."
        )

    aggregate = {}
    for profile in ("strict", "no_whitespace"):
        aggregate[profile] = {}
        for method in DEFAULT_METHOD_SUFFIXES:
            aggregate[profile][method] = aggregate_results(per_document, profile, method)
        aggregate[profile]["baseline_to_ensemble_llm_improvement"] = improvement_statistics(
            aggregate[profile]["paddleocr_baseline"],
            aggregate[profile]["ensemble_llm"],
        )

    return {
        "metric": "CER = (substitutions + deletions + insertions) / reference characters",
        "accuracy_metric": "Character accuracy = max(0, 1 - CER)",
        "normalization": {
            "all_profiles": "Unicode NFC; CRLF/CR converted to LF; outer whitespace stripped",
            "strict": "Internal whitespace and line breaks retained",
            "no_whitespace": "All Unicode whitespace removed",
        },
        "missing_predictions": missing,
        "aggregate": aggregate,
        "documents": per_document,
    }


def format_percent(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def write_csv(report, path):
    fieldnames = [
        "stem",
        "profile",
        "method",
        "reference_characters",
        "hypothesis_characters",
        "substitutions",
        "deletions",
        "insertions",
        "edits",
        "cer",
        "cer_percent",
        "character_accuracy",
        "character_accuracy_percent",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for document in report["documents"]:
            for profile, methods in document["profiles"].items():
                for method, result in methods.items():
                    writer.writerow(
                        {
                            "stem": document["stem"],
                            "profile": profile,
                            "method": method,
                            **result,
                            "cer_percent": result["cer"] * 100,
                            "character_accuracy_percent": result["character_accuracy"] * 100,
                        }
                    )


def write_markdown(report, path):
    lines = [
        "# CER Evaluation",
        "",
        "CER = (Substitutions + Deletions + Insertions) / Ground-truth Characters",
        "Character Accuracy = max(0, 1 - CER)",
        "",
    ]
    for profile in ("strict", "no_whitespace"):
        aggregate = report["aggregate"][profile]
        lines.extend(
            [
                f"## {profile.replace('_', ' ').title()}",
                "",
                "| Method | Documents | Ref chars | S | D | I | Edits | Error Rate (Micro CER) | Accuracy | Macro CER | Macro Accuracy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in DEFAULT_METHOD_SUFFIXES:
            result = aggregate[method]
            lines.append(
                f"| {method} | {result['document_count']} | {result['reference_characters']} | "
                f"{result['substitutions']} | {result['deletions']} | {result['insertions']} | "
                f"{result['edits']} | {format_percent(result['micro_cer'])} | "
                f"{format_percent(result['micro_character_accuracy'])} | "
                f"{format_percent(result['macro_cer'])} | "
                f"{format_percent(result['macro_character_accuracy'])} |"
            )
        improvement = aggregate["baseline_to_ensemble_llm_improvement"]
        if improvement:
            lines.extend(
                [
                    "",
                    "PaddleOCR baseline to ensemble + LLM:",
                    f"- Absolute CER reduction: {format_percent(improvement['absolute_cer_reduction'])}",
                    f"- Relative error reduction: {format_percent(improvement['relative_error_reduction'])}",
                ]
            )
        lines.append("")

    if report["missing_predictions"]:
        lines.extend(["## Missing Predictions", ""])
        for item in report["missing_predictions"]:
            lines.append(f"- `{item['stem']}` / `{item['method']}`: `{item['path']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cer_report.json"
    csv_path = output_dir / "cer_per_document.csv"
    markdown_path = output_dir / "cer_report.md"
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
        report = evaluate_directories(
            args.ground_truth_dir,
            args.baseline_dir,
            args.ensemble_dir,
            args.llm_dir,
            stems=args.stems,
            allow_missing=args.allow_missing,
        )
        paths = save_report(report, args.output_dir)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"CER evaluation failed: {exc}") from exc

    print("CER evaluation complete.")
    for profile in ("strict", "no_whitespace"):
        aggregate = report["aggregate"][profile]
        print(f"{profile.replace('_', ' ').title()} aggregate results:")
        for method in DEFAULT_METHOD_SUFFIXES:
            result = aggregate[method]
            print(
                f"- {method}: error rate (CER) {format_percent(result['micro_cer'])}; "
                f"accuracy {format_percent(result['micro_character_accuracy'])}"
            )
    no_whitespace = report["aggregate"]["no_whitespace"]
    improvement = no_whitespace["baseline_to_ensemble_llm_improvement"]
    if improvement:
        print(
            "Relative error reduction: "
            f"{format_percent(improvement['relative_error_reduction'])}"
        )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
