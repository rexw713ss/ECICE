import unittest
from pathlib import Path

from pipeline_paths import build_stage_paths


class PipelinePathTests(unittest.TestCase):
    def test_builds_numbered_stage_directories(self):
        paths = build_stage_paths(Path("dataset") / "page.jpg", Path("output"))

        self.assertEqual(paths["preprocessed"], Path("output/01_preprocessing/page_preprocessed_input.jpg"))
        self.assertEqual(paths["baseline_text"], Path("output/02_ocr/baseline/page_paddleocr_baseline.txt"))
        self.assertEqual(paths["ablation_document_text"], Path("output/02_ocr/ablation/page_document_rectification.txt"))
        self.assertEqual(paths["ablation_blue_ink_text"], Path("output/02_ocr/ablation/page_blue_ink_extraction.txt"))
        self.assertEqual(paths["ablation_line_removal_text"], Path("output/02_ocr/ablation/page_line_removal.txt"))
        self.assertEqual(paths["ocr_text"], Path("output/02_ocr/ensemble/page_merged_ocr.txt"))
        self.assertEqual(paths["corrected_text"], Path("output/03_llm_correction/page_corrected.txt"))
        self.assertEqual(paths["cer_md"], Path("output/04_evaluation/page/cer_report.md"))
        self.assertEqual(paths["error_analysis_csv"], Path("output/04_evaluation/page/error_analysis.csv"))
        self.assertEqual(paths["ablation_md"], Path("output/04_evaluation/page/ablation_report.md"))
        self.assertEqual(paths["summary_md"], Path("output/05_summary/page_summary.md"))
        self.assertEqual(paths["quiz_md"], Path("output/06_quiz/page_quiz.md"))


if __name__ == "__main__":
    unittest.main()
