import tempfile
import unittest
from pathlib import Path

from ui_app import INDEX_HTML, public_run


class UiAppTests(unittest.TestCase):
    def test_exposes_cer_and_error_analysis_in_run_payload(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary_path = Path(temporary_directory) / "page_summary.md"
            summary_path.write_text("# 摘要\n\n內容", encoding="utf-8")
            rows = [
                {
                    "error_type": "缺字",
                    "example": "page：GT「原則」",
                    "raw_ocr": "∅",
                    "corrected": "原則",
                    "success": "是",
                }
            ]
            run = {
                "stem": "page",
                "display_name": "page",
                "summary_path": summary_path,
                "summary_data": {},
                "evaluation_data": {
                    "aggregate": {
                        "no_whitespace": {
                            "ensemble_llm": {
                                "micro_cer": 0.1,
                                "micro_character_accuracy": 0.9,
                            }
                        }
                    },
                    "error_analysis": {"rows": rows},
                },
                "ablation_data": {
                    "settings": [
                        {
                            "id": "raw_paddleocr",
                            "label": "Raw image + PaddleOCR",
                            "description": "baseline",
                        }
                    ],
                    "aggregate": {
                        "no_whitespace": {
                            "raw_paddleocr": {
                                "cer": 0.2,
                                "character_accuracy": 0.8,
                                "delta_cer_vs_baseline": 0,
                                "delta_cer_vs_parent": None,
                            }
                        }
                    },
                },
                "original": None,
                "processed_variants": [],
                "default_processed_key": "",
            }

            payload = public_run(run, include_summary=True)

            self.assertEqual(payload["error_analysis_rows"], rows)
            self.assertEqual(payload["ablation_rows"][0]["cer"], 0.2)
            self.assertEqual(
                payload["evaluation_summary"]["ensemble_llm"]["micro_character_accuracy"],
                0.9,
            )
            self.assertIn("CER 與錯誤分析", INDEX_HTML)
            self.assertIn("Ablation Study", INDEX_HTML)


if __name__ == "__main__":
    unittest.main()
