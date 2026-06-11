import tempfile
import unittest
from pathlib import Path

from cer_evaluation import edit_statistics, evaluate_directories, evaluate_pair, save_report


class EditStatisticsTests(unittest.TestCase):
    def test_counts_substitution_deletion_and_insertion(self):
        result = edit_statistics("abc", "axcd")

        self.assertEqual(result["substitutions"], 1)
        self.assertEqual(result["deletions"], 0)
        self.assertEqual(result["insertions"], 1)
        self.assertEqual(result["edits"], 2)
        self.assertAlmostEqual(result["cer"], 2 / 3)
        self.assertAlmostEqual(result["character_accuracy"], 1 / 3)

    def test_accuracy_is_never_negative_when_cer_exceeds_one(self):
        result = edit_statistics("a", "abcdef")

        self.assertGreater(result["cer"], 1)
        self.assertEqual(result["character_accuracy"], 0)

    def test_no_whitespace_profile_ignores_layout(self):
        strict = evaluate_pair("甲乙\n丙", "甲 乙丙", remove_whitespace=False)
        no_whitespace = evaluate_pair("甲乙\n丙", "甲 乙丙", remove_whitespace=True)

        self.assertGreater(strict["cer"], 0)
        self.assertEqual(no_whitespace["cer"], 0)

    def test_nfc_preserves_full_width_characters(self):
        result = evaluate_pair("ＡＢＣ", "ABC", remove_whitespace=True)

        self.assertEqual(result["cer"], 1)


class DirectoryEvaluationTests(unittest.TestCase):
    def test_compares_required_methods_and_optional_ensemble(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ground_truth = root / "ground_truth"
            baseline = root / "baseline"
            proposed = root / "proposed"
            ground_truth.mkdir()
            baseline.mkdir()
            proposed.mkdir()
            (ground_truth / "page.txt").write_text("甲乙丙", encoding="utf-8")
            (baseline / "page_paddleocr_baseline.txt").write_text("甲丁丙", encoding="utf-8")
            (proposed / "page_merged_ocr.txt").write_text("甲乙", encoding="utf-8")
            (proposed / "page_corrected.txt").write_text("甲乙丙", encoding="utf-8")

            report = evaluate_directories(ground_truth, baseline, proposed, proposed)

            aggregate = report["aggregate"]["no_whitespace"]
            self.assertAlmostEqual(aggregate["paddleocr_baseline"]["micro_cer"], 1 / 3)
            self.assertAlmostEqual(
                aggregate["paddleocr_baseline"]["micro_character_accuracy"],
                2 / 3,
            )
            self.assertAlmostEqual(aggregate["ensemble_only"]["micro_cer"], 1 / 3)
            self.assertEqual(aggregate["ensemble_llm"]["micro_cer"], 0)
            self.assertEqual(aggregate["ensemble_llm"]["micro_character_accuracy"], 1)
            self.assertAlmostEqual(
                aggregate["baseline_to_ensemble_llm_improvement"]["relative_error_reduction"],
                1,
            )
            report_paths = save_report(report, root / "reports")
            self.assertTrue(all(path.is_file() for path in report_paths))
            self.assertIn(
                "Relative error reduction: 100.00%",
                report_paths[2].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Accuracy",
                report_paths[2].read_text(encoding="utf-8"),
            )
            csv_report = report_paths[1].read_text(encoding="utf-8-sig")
            self.assertIn("cer_percent", csv_report)
            self.assertIn("character_accuracy_percent", csv_report)

    def test_rejects_empty_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ground_truth = root / "ground_truth"
            ground_truth.mkdir()
            (ground_truth / "page.txt").write_text(" \n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Ground truth is empty"):
                evaluate_directories(
                    ground_truth,
                    root / "baseline",
                    root / "ensemble",
                    root / "llm",
                )


if __name__ == "__main__":
    unittest.main()
