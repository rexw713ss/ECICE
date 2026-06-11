import unittest
from unittest.mock import patch

from llm_correction import (
    build_correction_messages,
    build_reconstruction_messages,
    chat_completion,
    correct_text,
    normalize_ocr_noise,
)
from ocr_recognition import extract_candidates
from traditional_chinese import to_traditional_chinese


class TraditionalChineseTests(unittest.TestCase):
    def test_converts_simplified_chinese_to_taiwan_traditional(self):
        self.assertEqual(
            to_traditional_chinese("中华人民共和国与软件"),
            "中華人民共和國與軟體",
        )

    def test_preserves_special_symbols_formulas_and_private_use_characters(self):
        source = "笔记①：A→B，x²＋△與\ue000"
        converted = to_traditional_chinese(source)

        self.assertEqual(converted, "筆記①：A→B，x²＋△與\ue000")

    def test_ocr_cleanup_preserves_symbols_and_converts_chinese(self):
        self.assertEqual(normalize_ocr_noise("笔记①→软件"), "筆記①→軟體")

    def test_ocr_cleanup_normalizes_unicode_and_punctuation_whitespace(self):
        self.assertEqual(
            normalize_ocr_noise("e\u0301\u00a0， \r\n 软件"),
            "é，\n軟體",
        )

    def test_rule_based_correction_metadata_lists_normalization_steps(self):
        result = correct_text("软件  ，  筆記", use_llm=False)

        rule_based = result["correction_pipeline"]["rule_based_normalization"]
        llm_based = result["correction_pipeline"]["llm_based_correction"]
        self.assertTrue(rule_based["applied"])
        self.assertIn("Unicode NFC normalization", rule_based["steps"])
        self.assertIn("Punctuation and whitespace normalization", rule_based["steps"])
        self.assertFalse(llm_based["applied"])
        self.assertTrue(llm_based["grounded_no_new_information"])

    def test_correction_prompts_preserve_uncertain_text_without_adding_markers(self):
        correction_prompt = build_correction_messages("原文", 1, 1)[0]["content"]
        reconstruction_prompt = build_reconstruction_messages("原文")[0]["content"]

        self.assertIn("必須原樣保留", correction_prompt)
        self.assertIn("不得新增說明標記", reconstruction_prompt)
        self.assertNotIn("【待核對", correction_prompt)
        self.assertNotIn("【待核對", reconstruction_prompt)

    def test_ocr_candidates_are_traditional_before_ensemble_merge(self):
        candidates = extract_candidates(
            {
                "rec_texts": ["简体软件①"],
                "rec_scores": [0.9],
                "rec_boxes": [[0, 0, 100, 20]],
            },
            "document_full",
        )

        self.assertEqual(candidates[0]["text"], "簡體軟體①")

    @patch("llm_correction.ollama_chat_completion")
    def test_all_llm_calls_receive_instruction_and_return_traditional(self, completion):
        completion.return_value = "简体答案①"

        output = chat_completion(
            [{"role": "system", "content": "校對"}, {"role": "user", "content": "內容"}]
        )

        self.assertEqual(output, "簡體答案①")
        sent_messages = completion.call_args.args[0]
        self.assertIn("所有中文輸出一律使用臺灣繁體中文", sent_messages[0]["content"])
        self.assertIn("特殊符號", sent_messages[0]["content"])

    @patch("llm_correction.ollama_chat_completion")
    def test_grounded_correction_calls_forbid_unsupported_information(self, completion):
        completion.return_value = "校正"

        chat_completion(
            [{"role": "system", "content": "校對"}],
            grounded_correction=True,
        )

        sent_messages = completion.call_args.args[0]
        self.assertIn("不得補充、推論或新增", sent_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
