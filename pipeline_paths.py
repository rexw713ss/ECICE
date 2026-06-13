"""Shared filesystem layout for the note OCR experiment."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
GROUND_TRUTH_DIR = BASE_DIR / "ground_truth"
OUTPUT_ROOT = BASE_DIR / "output"

PREPROCESS_DIR = OUTPUT_ROOT / "01_preprocessing"
BASELINE_OCR_DIR = OUTPUT_ROOT / "02_ocr" / "baseline"
ABLATION_OCR_DIR = OUTPUT_ROOT / "02_ocr" / "ablation"
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
        "ablation_dir": output_root / "02_ocr" / "ablation",
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
        "ablation_document_text": directories["ablation_dir"] / f"{stem}_document_rectification.txt",
        "ablation_document_json": directories["ablation_dir"] / f"{stem}_document_rectification.json",
        "ablation_blue_ink_text": directories["ablation_dir"] / f"{stem}_blue_ink_extraction.txt",
        "ablation_blue_ink_json": directories["ablation_dir"] / f"{stem}_blue_ink_extraction.json",
        "ablation_line_removal_text": directories["ablation_dir"] / f"{stem}_line_removal.txt",
        "ablation_line_removal_json": directories["ablation_dir"] / f"{stem}_line_removal.json",
        "ocr_text": directories["ensemble_dir"] / f"{stem}_merged_ocr.txt",
        "ocr_json": directories["ensemble_dir"] / f"{stem}_merged_ocr.json",
        "corrected_text": directories["llm_dir"] / f"{stem}_corrected.txt",
        "corrected_json": directories["llm_dir"] / f"{stem}_corrected.json",
        "rule_based_text": directories["llm_dir"] / f"{stem}_rule_based.txt",
        "rule_based_json": directories["llm_dir"] / f"{stem}_rule_based.json",
        "cer_json": directories["evaluation_dir"] / "cer_report.json",
        "cer_csv": directories["evaluation_dir"] / "cer_per_document.csv",
        "cer_md": directories["evaluation_dir"] / "cer_report.md",
        "error_analysis_csv": directories["evaluation_dir"] / "error_analysis.csv",
        "ablation_json": directories["evaluation_dir"] / "ablation_report.json",
        "ablation_csv": directories["evaluation_dir"] / "ablation_report.csv",
        "ablation_md": directories["evaluation_dir"] / "ablation_report.md",
        "summary_md": directories["summary_dir"] / f"{stem}_summary.md",
        "summary_json": directories["summary_dir"] / f"{stem}_summary.json",
        "quiz_md": directories["quiz_dir"] / f"{stem}_quiz.md",
        "quiz_json": directories["quiz_dir"] / f"{stem}_quiz.json",
    }
