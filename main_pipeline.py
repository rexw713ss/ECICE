#!/usr/bin/env python3
"""Run the organized note OCR experiment from preprocessing through quiz."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from pipeline_paths import GROUND_TRUTH_DIR, OUTPUT_ROOT, build_stage_paths


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lecture-note OCR, correction, summary, and quiz pipeline."
    )
    parser.add_argument("--input", required=True, help="Input note image.")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Root directory for numbered stage outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model",
        default="gemma4:latest",
        help="Ollama model used for correction, summary, and quiz generation.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for each LLM request.",
    )
    parser.add_argument(
        "--question-count",
        type=int,
        default=5,
        help="Number of questions generated for each quiz type.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a step when all of its expected output files already exist.",
    )
    parser.add_argument(
        "--fast-ocr",
        action="store_true",
        help="Use the faster OCR experiment without overlapping vertical tiles.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run correction and summary without Ollama. Quiz generation is skipped.",
    )
    parser.add_argument(
        "--no-quiz",
        action="store_true",
        help="Generate the page summary but skip quiz generation.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Generate raw PaddleOCR baseline and run CER after LLM correction.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        default=str(GROUND_TRUTH_DIR),
        help="Directory containing <image_stem>.txt human transcriptions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and paths without running any module.",
    )
    return parser.parse_args()


def format_command(command):
    return subprocess.list2cmdline([str(part) for part in command])


def outputs_exist(paths):
    return bool(paths) and all(Path(path).is_file() for path in paths)


def run_step(step_number, step_name, command, expected_outputs, *, resume=False, dry_run=False):
    print()
    print("=" * 72)
    print(f"Step {step_number}: {step_name}")
    print(f"Command: {format_command(command)}")
    print("=" * 72)

    if resume and outputs_exist(expected_outputs):
        print("Skipped: expected outputs already exist.")
        return
    if dry_run:
        print("Dry run: command was not executed.")
        return

    started = time.monotonic()
    try:
        subprocess.run(
            [str(part) for part in command],
            check=True,
            cwd=BASE_DIR,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Step {step_number} failed with exit code {exc.returncode}: {step_name}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Step {step_number} could not start. Missing script or Python interpreter."
        ) from exc

    missing = [str(path) for path in expected_outputs if not Path(path).is_file()]
    if missing:
        raise RuntimeError(
            f"Step {step_number} completed but did not create expected outputs: "
            + ", ".join(missing)
        )
    elapsed = time.monotonic() - started
    print(f"Completed in {elapsed:.1f} seconds.")


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read pipeline metadata: {path}") from exc


def check_ollama(model, base_url):
    endpoint = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(endpoint, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not connect to Ollama at {base_url}.") from exc

    models = {
        item.get("name") or item.get("model")
        for item in data.get("models", [])
        if item.get("name") or item.get("model")
    }
    if model not in models:
        available = ", ".join(sorted(models)) or "none"
        raise RuntimeError(f"Ollama model '{model}' is unavailable. Installed models: {available}")


def validate_pipeline_outputs(paths, *, expect_llm, expect_quiz, question_count):
    if not paths["ocr_text"].read_text(encoding="utf-8", errors="replace").strip():
        raise RuntimeError("OCR output is empty.")
    ocr_data = load_json(paths["ocr_json"])
    if int(ocr_data.get("statistics", {}).get("clean_region_count", 0)) <= 0:
        raise RuntimeError("OCR metadata reports zero clean text regions.")

    if not paths["corrected_text"].read_text(encoding="utf-8", errors="replace").strip():
        raise RuntimeError("Corrected text output is empty.")
    corrected_data = load_json(paths["corrected_json"])
    if expect_llm and not corrected_data.get("used_llm"):
        raise RuntimeError("Text correction did not use the requested LLM.")

    summary_data = load_json(paths["summary_json"])
    if summary_data.get("validation_errors"):
        raise RuntimeError(
            "Page summary validation failed: "
            + "; ".join(summary_data["validation_errors"])
        )
    if expect_llm and not summary_data.get("used_llm"):
        raise RuntimeError("Page summary did not use the requested LLM.")

    if expect_quiz:
        quiz_data = load_json(paths["quiz_json"])
        if quiz_data.get("validation_errors"):
            raise RuntimeError(
                "Quiz validation failed: " + "; ".join(quiz_data["validation_errors"])
            )
        counts = quiz_data.get("question_counts", {})
        expected = {
            "multiple_choice": question_count,
            "true_false": question_count,
            "short_answer": question_count,
        }
        if any(int(counts.get(key, 0)) != value for key, value in expected.items()):
            raise RuntimeError(
                f"Quiz question counts are incorrect. Expected {expected}, received {counts}."
            )


def main():
    args = parse_args()
    input_image = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_image.is_file():
        raise SystemExit(f"Input image not found: {input_image}")
    if args.evaluate and args.no_llm:
        raise SystemExit("--evaluate requires LLM correction; remove --no-llm.")
    output_dir.mkdir(parents=True, exist_ok=True)

    python_bin = Path(sys.executable).resolve()
    paths = build_stage_paths(input_image, output_dir)
    use_llm_args = (
        []
        if args.no_llm
        else ["--use-llm", "--model", args.model, "--base-url", args.base_url]
    )
    llm_outputs = [
        paths["corrected_text"],
        paths["corrected_json"],
        paths["summary_md"],
        paths["summary_json"],
    ]
    if not args.no_quiz:
        llm_outputs.extend([paths["quiz_md"], paths["quiz_json"]])
    needs_llm_work = not args.resume or not outputs_exist(llm_outputs)
    if not args.no_llm and not args.dry_run and needs_llm_work:
        check_ollama(args.model, args.base_url)

    print("Starting lecture-note digitization pipeline")
    print(f"Input image: {input_image}")
    print(f"Output directory: {output_dir}")
    print(f"Python: {python_bin}")
    print(f"LLM enabled: {not args.no_llm}")
    print(f"CER evaluation enabled: {args.evaluate}")
    if not args.no_llm:
        print(f"Ollama model: {args.model}")
        print(f"Ollama base URL: {args.base_url}")

    ocr_args = [
        python_bin,
        BASE_DIR / "ocr_recognition.py",
        "--image",
        input_image,
        "--output-dir",
        paths["ensemble_dir"],
        "--preprocess-dir",
        paths["preprocess_dir"],
        "--skip-preprocess",
    ]
    if args.fast_ocr:
        ocr_args.extend(
            [
                "--full-sources",
                "document,blue_ink,blue_gray",
                "--no-tiles",
                "--grid-sources",
                "blue_gray",
                "--grid-configs",
                "900:180:520:180",
                "--det-limit-side-len",
                "1536",
                "--no-textline-orientation",
            ]
        )

    steps = [
        (
            "Image preprocessing",
            [
                python_bin,
                BASE_DIR / "image_preprocessing.py",
                "--image",
                input_image,
                "--output-dir",
                paths["preprocess_dir"],
            ],
            [paths["manifest"], paths["preprocessed"]],
        ),
    ]

    if args.evaluate:
        steps.append(
            (
                "Raw PaddleOCR baseline",
                [
                    python_bin,
                    BASE_DIR / "paddleocr_baseline.py",
                    "--input",
                    input_image,
                    "--output-dir",
                    paths["baseline_dir"],
                ],
                [paths["baseline_text"], paths["baseline_json"]],
            )
        )

    steps.extend(
        [
            (
                "Multi-variant ensemble OCR",
                ocr_args,
                [paths["ocr_text"], paths["ocr_json"]],
            ),
            (
                "LLM text correction" if not args.no_llm else "Rule-based text correction",
                [
                    python_bin,
                    BASE_DIR / "llm_correction.py",
                    "--input",
                    paths["ocr_text"],
                    "--output",
                    paths["corrected_text"],
                    "--ocr-json",
                    paths["ocr_json"],
                    "--timeout",
                    str(args.llm_timeout),
                    *use_llm_args,
                ],
                [paths["corrected_text"], paths["corrected_json"]],
            ),
        ]
    )

    if args.evaluate:
        steps.append(
            (
                "CER evaluation",
                [
                    python_bin,
                    BASE_DIR / "cer_evaluation.py",
                    "--ground-truth-dir",
                    Path(args.ground_truth_dir).expanduser().resolve(),
                    "--baseline-dir",
                    paths["baseline_dir"],
                    "--ensemble-dir",
                    paths["ensemble_dir"],
                    "--llm-dir",
                    paths["llm_dir"],
                    "--output-dir",
                    paths["evaluation_dir"],
                    "--stem",
                    input_image.stem,
                ],
                [paths["cer_json"], paths["cer_csv"], paths["cer_md"]],
            )
        )

    steps.append(
        (
            "Page summary generation",
            [
                python_bin,
                BASE_DIR / "note_summarization.py",
                "--input",
                paths["corrected_text"],
                "--output",
                paths["summary_md"],
                "--timeout",
                str(args.llm_timeout),
                *use_llm_args,
            ],
            [paths["summary_md"], paths["summary_json"]],
        )
    )

    if not args.no_quiz and not args.no_llm:
        steps.append(
            (
                "Quiz generation",
                [
                    python_bin,
                    BASE_DIR / "quiz_generation.py",
                    "--input",
                    paths["summary_md"],
                    "--output",
                    paths["quiz_md"],
                    "--question-count",
                    str(args.question_count),
                    "--timeout",
                    str(args.llm_timeout),
                    *use_llm_args,
                ],
                [paths["quiz_md"], paths["quiz_json"]],
            )
        )

    try:
        for index, (name, command, expected_outputs) in enumerate(steps, start=1):
            run_step(
                index,
                name,
                command,
                expected_outputs,
                resume=args.resume,
                dry_run=args.dry_run,
            )
        if not args.dry_run:
            validate_pipeline_outputs(
                paths,
                expect_llm=not args.no_llm,
                expect_quiz=not args.no_quiz and not args.no_llm,
                question_count=args.question_count,
            )
    except RuntimeError as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print()
    print("=" * 72)
    print("Pipeline dry run completed." if args.dry_run else "Pipeline completed successfully.")
    print(f"Preprocessed image: {paths['preprocessed']}")
    print(f"OCR text: {paths['ocr_text']}")
    print(f"Corrected text: {paths['corrected_text']}")
    if args.evaluate:
        print(f"CER evaluation: {paths['cer_md']}")
    print(f"Page summary: {paths['summary_md']}")
    if not args.no_quiz and not args.no_llm:
        print(f"Quiz: {paths['quiz_md']}")
    elif args.no_llm and not args.no_quiz:
        print("Quiz: skipped because --no-llm was selected.")
    else:
        print("Quiz: skipped.")
    print("=" * 72)


if __name__ == "__main__":
    main()
