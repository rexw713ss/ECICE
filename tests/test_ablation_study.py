import tempfile
import unittest
from pathlib import Path

from ablation_study import ABLATION_SETTINGS, evaluate_ablation, save_report


class AblationStudyTests(unittest.TestCase):
    def test_evaluates_requested_settings_in_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ground_truth = root / "ground_truth"
            baseline = root / "baseline"
            ablation = root / "ablation"
            ensemble = root / "ensemble"
            llm = root / "llm"
            for directory in (ground_truth, baseline, ablation, ensemble, llm):
                directory.mkdir()

            (ground_truth / "page.txt").write_text("甲乙丙", encoding="utf-8")
            predictions = {
                baseline / "page_paddleocr_baseline.txt": "甲丁丙",
                ablation / "page_document_rectification.txt": "甲乙丙",
                ablation / "page_blue_ink_extraction.txt": "甲乙",
                ablation / "page_line_removal.txt": "甲乙丙",
                ensemble / "page_merged_ocr.txt": "甲乙丙",
                llm / "page_rule_based.txt": "甲乙丙",
                llm / "page_corrected.txt": "甲乙丙",
            }
            for path, text in predictions.items():
                path.write_text(text, encoding="utf-8")

            report = evaluate_ablation(
                ground_truth,
                baseline,
                ablation,
                ensemble,
                llm,
            )

            self.assertEqual(
                [setting["id"] for setting in report["settings"]],
                [setting["id"] for setting in ABLATION_SETTINGS],
            )
            aggregate = report["aggregate"]["no_whitespace"]
            self.assertAlmostEqual(aggregate["raw_paddleocr"]["cer"], 1 / 3)
            self.assertEqual(aggregate["document_rectification"]["cer"], 0)
            self.assertAlmostEqual(
                aggregate["document_rectification"]["delta_cer_vs_baseline"],
                -1 / 3,
            )
            self.assertEqual(aggregate["llm_correction"]["cer"], 0)
            self.assertEqual(aggregate["llm_correction"]["delta_cer_vs_parent"], 0)

            paths = save_report(report, root / "reports")
            self.assertTrue(all(path.is_file() for path in paths))
            markdown = paths[2].read_text(encoding="utf-8")
            self.assertIn(
                "| Setting | CER ↓ | Accuracy ↑ | Δ CER vs baseline | Δ CER vs parent | 說明 |",
                markdown,
            )
            self.assertIn("+ multi-variant ensemble", markdown)
            self.assertIn("+ rule-based normalization", markdown)

    def test_rejects_empty_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ground_truth = root / "ground_truth"
            ground_truth.mkdir()
            (ground_truth / "page.txt").write_text(" \n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Ground truth is empty"):
                evaluate_ablation(
                    ground_truth,
                    root / "baseline",
                    root / "ablation",
                    root / "ensemble",
                    root / "llm",
                    allow_missing=True,
                )


if __name__ == "__main__":
    unittest.main()
