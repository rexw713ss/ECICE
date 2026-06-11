"""Shared filesystem layout for the note OCR experiment."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
GROUND_TRUTH_DIR = BASE_DIR / "ground_truth"
OUTPUT_ROOT = BASE_DIR / "output"

PREPROCESS_DIR = OUTPUT_ROOT / "01_preprocessing"
BASELINE_OCR_DIR = OUTPUT_ROOT / "02_ocr" / "baseline"
ENSEMBLE_OCR_DIR = OUTPUT_ROOT / "02_ocr" / "ensemble"
LLM_CORRECTION_DIR = OUTPUT_ROOT / "03_llm_correction"
EVALUATION_DIR = OUTPUT_ROOT / "04_evaluation"
SUMMARY_DIR = OUTPUT_ROOT / "05_summary"
QUIZ_DIR = OUTPUT_ROOT / "06_quiz"


def build_stage_paths(input_image, output_root=OUTPUT_ROOT):
    input_image = Path(input_image)
    output_root = Path(output_root)
    stem = input_image.stem
    directories = {
        "preprocess_dir": output_root / "01_preprocessing",
        "baseline_dir": output_root / "02_ocr" / "baseline",
        "ensemble_dir": output_root / "02_ocr" / "ensemble",
        "llm_dir": output_root / "03_llm_correction",
        "evaluation_dir": output_root / "04_evaluation" / stem,
        "summary_dir": output_root / "05_summary",
        "quiz_dir": output_root / "06_quiz",
    }
    return {
        **directories,
        "manifest": directories["preprocess_dir"] / f"{stem}_preprocess_manifest.json",
        "preprocessed": directories["preprocess_dir"] / f"{stem}_preprocessed_input.jpg",
        "baseline_text": directories["baseline_dir"] / f"{stem}_paddleocr_baseline.txt",
        "baseline_json": directories["baseline_dir"] / f"{stem}_paddleocr_baseline.json",
        "ocr_text": directories["ensemble_dir"] / f"{stem}_merged_ocr.txt",
        "ocr_json": directories["ensemble_dir"] / f"{stem}_merged_ocr.json",
        "corrected_text": directories["llm_dir"] / f"{stem}_corrected.txt",
        "corrected_json": directories["llm_dir"] / f"{stem}_corrected.json",
        "cer_json": directories["evaluation_dir"] / "cer_report.json",
        "cer_csv": directories["evaluation_dir"] / "cer_per_document.csv",
        "cer_md": directories["evaluation_dir"] / "cer_report.md",
        "summary_md": directories["summary_dir"] / f"{stem}_summary.md",
        "summary_json": directories["summary_dir"] / f"{stem}_summary.json",
        "quiz_md": directories["quiz_dir"] / f"{stem}_quiz.md",
        "quiz_json": directories["quiz_dir"] / f"{stem}_quiz.json",
    }
